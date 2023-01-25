import random
import tempfile
import time
import gradio as gr
import numpy as np
import torch

from gradio import inputs
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
)
from modules.model_pww import CrossAttnProcessor, StableDiffusionPipeline
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
from pathlib import Path
from safetensors.torch import load_file

models = [
    ("AbyssOrangeMix_Base", "/root/workspace/storage/models/orangemix"),
    ("AbyssOrangeMix2", "/root/models/AbyssOrangeMix2"),
    ("Stable Diffuison 1.5", "/root/models/stable-diffusion-v1-5"),
    ("AnimeSFW", "/root/workspace/animesfw")
]

samplers_k_diffusion = [
    ("Euler a", "sample_euler_ancestral", {}),
    ("Euler", "sample_euler", {}),
    ("LMS", "sample_lms", {}),
    ("Heun", "sample_heun", {}),
    ("DPM2", "sample_dpm_2", {"discard_next_to_last_sigma": True}),
    ("DPM2 a", "sample_dpm_2_ancestral", {"discard_next_to_last_sigma": True}),
    ("DPM++ 2S a", "sample_dpmpp_2s_ancestral", {}),
    ("DPM++ 2M", "sample_dpmpp_2m", {}),
    ("DPM++ SDE", "sample_dpmpp_sde", {}),
    ("DPM fast", "sample_dpm_fast", {}),
    ("DPM adaptive", "sample_dpm_adaptive", {}),
    ("LMS Karras", "sample_lms", {"scheduler": "karras"}),
    (
        "DPM2 Karras",
        "sample_dpm_2",
        {"scheduler": "karras", "discard_next_to_last_sigma": True},
    ),
    (
        "DPM2 a Karras",
        "sample_dpm_2_ancestral",
        {"scheduler": "karras", "discard_next_to_last_sigma": True},
    ),
    ("DPM++ 2S a Karras", "sample_dpmpp_2s_ancestral", {"scheduler": "karras"}),
    ("DPM++ 2M Karras", "sample_dpmpp_2m", {"scheduler": "karras"}),
    ("DPM++ SDE Karras", "sample_dpmpp_sde", {"scheduler": "karras"}),
]

start_time = time.time()

scheduler = DDIMScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="scheduler",
)

vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float16
)

text_encoder = CLIPTextModel.from_pretrained(
    "/root/workspace/storage/models/orangemix",
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)

tokenizer = CLIPTokenizer.from_pretrained(
    "/root/workspace/storage/models/orangemix",
    subfolder="tokenizer",
    torch_dtype=torch.float16,
)

unet = UNet2DConditionModel.from_pretrained(
    "/root/workspace/storage/models/orangemix",
    subfolder="unet",
    torch_dtype=torch.float16,
)

pipe = StableDiffusionPipeline(
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    vae=vae,
    scheduler=scheduler,
)

unet.set_attn_processor(CrossAttnProcessor)
# hook_unet(tokenizer, unet, scheduler)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")

def get_model_list():
    model_available = []
    for model in models:
        if Path(model[1]).is_dir():
           model_available.append(model)
    return model_available 

unet_cache = dict()
def get_model(name):
    keys = [k[0] for k in models]
    if name not in unet_cache:
        if name not in keys:
            raise ValueError(name)
        else:
            unet = UNet2DConditionModel.from_pretrained(
                models[keys.index(name)][1],
                subfolder="unet",
                torch_dtype=torch.float16,
            )
            unet.set_attn_processor(CrossAttnProcessor)
            unet_cache[name] = unet
    return unet_cache[name]
    

def error_str(error, title="Error"):
    return (
        f"""#### {title}
            {error}"""
        if error
        else ""
    )


te_base_weight = text_encoder.get_input_embeddings().weight.data.detach().clone()


def restore_all():
    global te_base_weight, tokenizer
    text_encoder.get_input_embeddings().weight.data = te_base_weight
    tokenizer = CLIPTokenizer.from_pretrained(
        "/root/workspace/storage/models/orangemix",
        subfolder="tokenizer",
        torch_dtype=torch.float16,
    )


def inference(
    prompt,
    guidance,
    steps,
    width=512,
    height=512,
    seed=0,
    neg_prompt="",
    state=None,
    g_strength=0.4,
    img_input=None,
    i2i_scale=0.5,
    hr_enabled=False,
    hr_method="Latent",
    hr_scale=1.5,
    hr_denoise=0.8,
    sampler="DPM++ 2M Karras",
    embs=None,
    model=None,
):
    global pipe, unet, tokenizer, text_encoder
    pipe.unet = get_model(model)
    seed = int(seed)
    if seed == 0 or seed is None:
        seed = int(random.randrange(6894327513))
    generator = torch.Generator("cuda").manual_seed(seed) 
    sampler_name, sampler_opt = None, None
    for label, funcname, options in samplers_k_diffusion:
        if label == sampler:
            sampler_name, sampler_opt = funcname, options

    if embs is not None and len(embs) > 0:
        delta_weight = []
        for name, file in embs.items():
            if str(file).endswith(".pt"):
                loaded_learned_embeds = torch.load(file, map_location="cpu")
            else:
                loaded_learned_embeds = load_file(file, device="cpu")
            loaded_learned_embeds = loaded_learned_embeds["string_to_param"]["*"]
            tokenizer.add_tokens(name)
            delta_weight.append(loaded_learned_embeds)

        delta_weight = torch.cat(delta_weight, dim=0)
        text_encoder.resize_token_embeddings(len(tokenizer))
        text_encoder.get_input_embeddings().weight.data[
            -delta_weight.shape[0] :
        ] = delta_weight

    config = {
        "negative_prompt": neg_prompt,
        "num_inference_steps": int(steps),
        "guidance_scale": guidance,
        "generator": generator,
        "sampler_name": sampler_name,
        "sampler_opt": sampler_opt,
        "pww_state": state,
        "pww_attn_weight": g_strength,
    }

    if img_input is not None:
        ratio = min(height / img_input.height, width / img_input.width)
        img_input = img_input.resize(
            (int(img_input.width * ratio), int(img_input.height * ratio)), Image.LANCZOS
        )
        result = pipe.img2img(prompt, image=img_input, strength=i2i_scale, **config)
    elif hr_enabled:
        result = pipe.txt2img(
            prompt,
            width=width,
            height=height,
            upscale=True,
            upscale_x=hr_scale,
            upscale_denoising_strength=hr_denoise,
            **config,
            **latent_upscale_modes[hr_method],
        )
    else:
        result = pipe.txt2img(prompt, width=width, height=height, **config)

    # restore
    if embs is not None and len(embs) > 0:
        restore_all()
    return gr.Image.update(result[0][0], label=f"Image Seed: {seed}")


color_list = []


def get_color(n):
    for _ in range(n - len(color_list)):
        color_list.append(tuple(np.random.random(size=3) * 256))
    return color_list


def create_mixed_img(current, state, w=512, h=512):
    w, h = int(w), int(h)
    image_np = np.full([h, w, 4], 255)
    colors = get_color(len(state))
    idx = 0

    for key, item in state.items():
        if item["map"] is not None:
            m = item["map"] < 255
            alpha = 150
            if current == key:
                alpha = 200
            image_np[m] = colors[idx] + (alpha,)
        idx += 1

    return image_np


# width.change(apply_new_res, inputs=[width, height, global_stats], outputs=[global_stats, sp, rendered])
def apply_new_res(w, h, state):
    w, h = int(w), int(h)

    for key, item in state.items():
        if item["map"] is not None:
            item["map"] = resize(item["map"], w, h)

    update_img = gr.Image.update(value=create_mixed_img("", state, w, h))
    return state, update_img


def detect_text(text, state, width, height):

    t = text.split(",")
    new_state = {}

    for item in t:
        item = item.strip()
        if item == "":
            continue
        if item in state:
            new_state[item] = {
                "map": state[item]["map"],
                "weight": state[item]["weight"],
            }
        else:
            new_state[item] = {
                "map": None,
                "weight": 0.5,
            }
    update = gr.Radio.update(choices=[key for key in new_state.keys()], value=None)
    update_img = gr.update(value=create_mixed_img("", new_state, width, height))
    update_sketch = gr.update(value=None, interactive=False)
    return new_state, update_sketch, update, update_img


def resize(img, w, h):
    trs = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(min(h, w)),
            transforms.CenterCrop((h, w)),
        ]
    )
    result = np.array(trs(img), dtype=np.uint8)
    return result


def switch_canvas(entry, state, width, height):
    if entry == None:
        return None, 0.5, create_mixed_img("", state, width, height)
    return (
        gr.update(value=None, interactive=True),
        gr.update(value=state[entry]["weight"]),
        create_mixed_img(entry, state, width, height),
    )


def apply_canvas(selected, draw, state, w, h):
    w, h = int(w), int(h)
    state[selected]["map"] = resize(draw, w, h)
    return state, gr.Image.update(value=create_mixed_img(selected, state, w, h))


def apply_weight(selected, weight, state):
    state[selected]["weight"] = weight
    return state


# sp2, radio, width, height, global_stats
def apply_image(image, selected, w, h, strgength, state):
    if selected is not None:
        state[selected] = {"map": resize(image, w, h), "weight": strgength}
    return state, gr.Image.update(value=create_mixed_img(selected, state, w, h))


def add_ti(embs: list[tempfile._TemporaryFileWrapper], state):
    if embs is None:
        return state, ""
    
    state = dict()
    for emb in embs:
        item = Path(emb.name)
        stripedname = str(item.stem).strip()
        state[stripedname] = emb.name

    return state, gr.Text.update(f"{[key for key in state.keys()]}")


# def add_lora(loras: list[tempfile._TemporaryFileWrapper], state):
#     for lora in loras:
#         item = Path(lora.name)
#         stripedname = str(item.stem).strip()
#         state[stripedname] = lora.name

#     return state, gr.Text.update(f"{[key for key in state.keys()]}")


latent_upscale_modes = {
    "Latent": {"upscale_method": "bilinear", "upscale_antialias": False},
    "Latent (antialiased)": {"upscale_method": "bilinear", "upscale_antialias": True},
    "Latent (bicubic)": {"upscale_method": "bicubic", "upscale_antialias": False},
    "Latent (bicubic antialiased)": {
        "upscale_method": "bicubic",
        "upscale_antialias": True,
    },
    "Latent (nearest)": {"upscale_method": "nearest", "upscale_antialias": False},
    "Latent (nearest-exact)": {
        "upscale_method": "nearest-exact",
        "upscale_antialias": False,
    },
}

css = """
.finetuned-diffusion-div div{
    display:inline-flex;
    align-items:center;
    gap:.8rem;
    font-size:1.75rem;
    padding-top:2rem;
}
.finetuned-diffusion-div div h1{
    font-weight:900;
    margin-bottom:7px
}
.finetuned-diffusion-div p{
    margin-bottom:10px;
    font-size:94%
}
.box {
  float: left;
  height: 20px;
  width: 20px;
  margin-bottom: 15px;
  border: 1px solid black;
  clear: both;
}
a{
    text-decoration:underline
}
.tabs{
    margin-top:0;
    margin-bottom:0
}
#gallery{
    min-height:20rem
}
.no-border {
    border: none !important;
}
 """
with gr.Blocks(css=css) as demo:
    gr.HTML(
        f"""
            <div class="finetuned-diffusion-div">
              <div>
                <h1>Demo for diffusion models</h1>
              </div>
              <p>Hso @ nyanko.sketch2img.pww_gradio</p>
            </div>
        """
    )
    global_stats = gr.State(value={})

    with gr.Row():

        with gr.Column(scale=55):
                        model = gr.Dropdown(
                            choices=[k[0] for k in get_model_list()],
                            label="Model",
                            value="AbyssOrangeMix_Base",
                        )
                        image_out = gr.Image(height=512)
            # gallery = gr.Gallery(
            #     label="Generated images", show_label=False, elem_id="gallery"
            # ).style(grid=[1], height="auto")

        with gr.Column(scale=45):

            with gr.Group():

                with gr.Row():
                    with gr.Column(scale=70):


                        prompt = gr.Textbox(
                            label="Prompt",
                            value="loli cat girl, blue eyes, flat chest, solo, long messy silver hair, blue capelet, garden, cat ears, cat tail, upper body",
                            show_label=True,
                            max_lines=2,
                            placeholder="Enter prompt.",
                        )
                        neg_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="bad quality, low quality, jpeg artifact, cropped",
                            show_label=True,
                            max_lines=2,
                            placeholder="Enter negative prompt.",
                        )

                    generate = gr.Button(value="Generate").style(
                        rounded=(False, True, True, False)
                    )

            with gr.Tab("Options"):

                with gr.Group():

                    # n_images = gr.Slider(label="Images", value=1, minimum=1, maximum=4, step=1)
                    with gr.Row():
                        guidance = gr.Slider(
                            label="Guidance scale", value=7.5, maximum=15
                        )
                        steps = gr.Slider(
                            label="Steps", value=25, minimum=2, maximum=75, step=1
                        )

                    with gr.Row():
                        width = gr.Slider(
                            label="Width", value=512, minimum=64, maximum=2048, step=64
                        )
                        height = gr.Slider(
                            label="Height", value=512, minimum=64, maximum=2048, step=64
                        )

                    sampler = gr.Dropdown(
                        value="DPM++ 2M Karras",
                        label="Sampler",
                        choices=[s[0] for s in samplers_k_diffusion],
                    )
                    seed = gr.Number(label="Seed (0 = random)", value=0)

            with gr.Tab("Image to image"):
                with gr.Group():

                    inf_image = gr.Image(
                        label="Image", height=256, tool="editor", type="pil"
                    )
                    inf_strength = gr.Slider(
                        label="Transformation strength",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.5,
                    )

            def res_cap(g, w, h, x):
                if g:
                    return f"Enable upscaler: {w}x{h} to {int(w*x)}x{int(h*x)}"
                else:
                    return "Enable upscaler"

            with gr.Tab("Hires fix"):
                with gr.Group():

                    hr_enabled = gr.Checkbox(label="Enable upscaler", value=False)
                    hr_method = gr.Dropdown(
                        [key for key in latent_upscale_modes.keys()],
                        value="Latent",
                        label="Upscale method",
                    )
                    hr_scale = gr.Slider(
                        label="Upscale factor",
                        minimum=1.0,
                        maximum=3,
                        step=0.1,
                        value=1.5,
                    )
                    hr_denoise = gr.Slider(
                        label="Denoising strength",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=0.8,
                    )

                    hr_scale.change(
                        lambda g, x, w, h: gr.Checkbox.update(
                            label=res_cap(g, w, h, x)
                        ),
                        inputs=[hr_enabled, hr_scale, width, height],
                        outputs=hr_enabled,
                    )
                    hr_enabled.change(
                        lambda g, x, w, h: gr.Checkbox.update(
                            label=res_cap(g, w, h, x)
                        ),
                        inputs=[hr_enabled, hr_scale, width, height],
                        outputs=hr_enabled,
                    )

            with gr.Tab("Embeddings"):

                ti_state = gr.State(dict())
                
                with gr.Group():
                    with gr.Row():
                        with gr.Column(scale=90):
                            ti_vals = gr.Text(label="Avaliable")
                            
                        btn = gr.Button(value="Update").style(
                            rounded=(False, True, True, False)
                        )
                        
                ti_uploads = gr.Files(
                        label="Upload new embeddings",
                        file_types=[".pt", ".safetensors"],
                )
                btn.click(
                    add_ti, inputs=[ti_uploads, ti_state], outputs=[ti_state, ti_vals]
                )

            # with gr.Tab("Loras"):

            #         lora_state = gr.State(dict())
            #         lora_vals = gr.Text(label="Avaliable")
            
            #         lora_uploads = gr.Files(label="Upload new loras", file_types=[".pt", ".safetensors"])

            #         btn2 = gr.Button(value='Upload')
            #         btn2.click(add_lora, inputs=[lora_uploads, lora_state], outputs=[lora_state, lora_vals])

        # error_output = gr.Markdown()

    gr.HTML(
        f"""
            <div class="finetuned-diffusion-div">
              <div>
                <h1>Paint with words</h1>
              </div>
              <p>
                Will use the following formula: w = scale * token_weight_martix * log(1 + sigma) * max(qk).
              </p>
            </div>
        """
    )

    with gr.Row():

        with gr.Column(scale=55):

            rendered = gr.Image(
                invert_colors=True,
                source="canvas",
                interactive=False,
                image_mode="RGBA",
            )

        with gr.Column(scale=45):

            with gr.Group():
                with gr.Row():
                    with gr.Column(scale=70):
                        g_strength = gr.Slider(
                            label="Weight scaling",
                            minimum=0,
                            maximum=0.8,
                            step=0.01,
                            value=0.4,
                        )

                        text = gr.Textbox(
                            lines=2,
                            interactive=True,
                            label="Token to Draw: (Separate by comma)",
                        )

                        radio = gr.Radio([], label="Tokens")

                    sk_update = gr.Button(value="Update").style(
                        rounded=(False, True, True, False)
                    )

                # g_strength.change(lambda b: gr.update(f"Scaled additional attn: $w = {b} \log (1 + \sigma) \std (Q^T K)$."), inputs=g_strength, outputs=[g_output])

            with gr.Tab("SketchPad"):

                sp = gr.Image(
                    image_mode="L",
                    tool="sketch",
                    source="canvas",
                    interactive=False,
                )

                strength = gr.Slider(
                    label="Token strength",
                    minimum=0,
                    maximum=0.8,
                    step=0.01,
                    value=0.5,
                )

                sk_update.click(
                    detect_text,
                    inputs=[text, global_stats, width, height],
                    outputs=[global_stats, sp, radio, rendered],
                )
                radio.change(
                    switch_canvas,
                    inputs=[radio, global_stats, width, height],
                    outputs=[sp, strength, rendered],
                )
                sp.edit(
                    apply_canvas,
                    inputs=[radio, sp, global_stats, width, height],
                    outputs=[global_stats, rendered],
                )
                strength.change(
                    apply_weight,
                    inputs=[radio, strength, global_stats],
                    outputs=[global_stats],
                )

            with gr.Tab("UploadFile"):

                sp2 = gr.Image(
                    image_mode="L",
                    source="upload",
                    shape=(512, 512),
                )

                strength2 = gr.Slider(
                    label="Token strength",
                    minimum=0,
                    maximum=0.8,
                    step=0.01,
                    value=0.5,
                )

                apply_style = gr.Button(value="Apply")
                apply_style.click(
                    apply_image,
                    inputs=[sp2, radio, width, height, strength2, global_stats],
                    outputs=[global_stats, rendered],
                )

            width.change(
                apply_new_res,
                inputs=[width, height, global_stats],
                outputs=[global_stats, rendered],
            )
            height.change(
                apply_new_res,
                inputs=[width, height, global_stats],
                outputs=[global_stats, rendered],
            )

    # color_stats = gr.State(value={})
    # text.change(detect_color, inputs=[sp, text, color_stats], outputs=[color_stats, rendered])
    # sp.change(detect_color, inputs=[sp, text, color_stats], outputs=[color_stats, rendered])

    inputs = [
        prompt,
        guidance,
        steps,
        width,
        height,
        seed,
        neg_prompt,
        global_stats,
        g_strength,
        inf_image,
        inf_strength,
        hr_enabled,
        hr_method,
        hr_scale,
        hr_denoise,
        sampler,
        ti_state,
        model,
    ]
    outputs = [image_out]
    prompt.submit(inference, inputs=inputs, outputs=outputs)
    generate.click(inference, inputs=inputs, outputs=outputs)

print(f"Space built in {time.time() - start_time:.2f} seconds")
# demo.launch(share=True)
demo.launch(share=True, enable_queue=True)