import argparse
import contextlib
import itertools
import math
import os
import tempfile

import bitsandbytes as bnb
from einops import rearrange
import torch
import torchtext
import torchvision
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from anime2sketch.model import create_model
from modules.dataset import ImageStore
from modules.latent_predictor import LatentEdgePredictor, hook_unet
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import CLIPTokenizer

import diffusers
from diffusers import DDIMScheduler, StableDiffusionPipeline

# ref: https://github.com/ogkalu2/Sketch-Guided-Stable-Diffusion/blob/main/train_LGP.py
# ref: https://sketch-guided-diffusion.github.io

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/root/workspace/sketch2img/train.yaml")
    parser.add_argument("--network_weights", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    return args

def generate_sketch(sketch_generator, img, fixed=1024, method=torchvision.transforms.InterpolationMode.BICUBIC):
    org_size = (img.shape[-2], img.shape[-1])
    transformed = torchvision.transforms.Resize((fixed, fixed), method)(img)
    val = 1-sketch_generator(transformed)
    val[val<0.5] = 0
    val[val>=0.5] = 1
    tiled = torch.tile(val, (3, 1, 1))
    resized = torchvision.transforms.Resize(org_size, method)(tiled)
    return resized

def encode_tokens(tokenizer, text_encoder, input_ids):
    z = []
    if input_ids.shape[1] > 77:  
        # todo: Handle end-of-sentence truncation
        while max(map(len, input_ids)) != 0:
            rem_tokens = [x[75:] for x in input_ids]
            tokens = []
            for j in range(len(input_ids)):
                tokens.append(input_ids[j][:75] if len(input_ids[j]) > 0 else [tokenizer.eos_token_id] * 75)

            rebuild = [[tokenizer.bos_token_id] + list(x[:75]) + [tokenizer.eos_token_id] for x in tokens]
            if hasattr(torch, "asarray"):
                z.append(torch.asarray(rebuild))
            else:
                z.append(torch.IntTensor(rebuild))
            input_ids = rem_tokens
    else:
        z.append(input_ids)

    # Get the text embedding for conditioning
    encoder_hidden_states = None
    for tokens in z:
        state = text_encoder(tokens.to(text_encoder.device), output_hidden_states=True)
        state = text_encoder.text_model.final_layer_norm(state['hidden_states'][-1])
        encoder_hidden_states = state if encoder_hidden_states is None else torch.cat((encoder_hidden_states, state), axis=-2)
        
    return encoder_hidden_states

def train():

    args = parse_args()
    config = OmegaConf.load(args.config)
    # get_world_size = lambda: int(os.environ.get("WORLD_SIZE", 1))
    get_local_rank = lambda: int(os.environ.get("LOCAL_RANK", -1))
    set_seed(config.seed)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    dataset = ImageStore(
        size=config.resolution,
        seed=config.seed,
        rank=get_local_rank(),
        tokenizer=tokenizer,
        **config.dataset
    )
    
    ddp_scaler = DistributedDataParallelKwargs(bucket_cap_mb=15, find_unused_parameters=False)
    metrics = []
    if config.monitor.wandb_id != "" and get_local_rank() in [0, -1]:
        import wandb
        wandb.init(project=config.monitor.wandb_id, reinit=False)
        metrics.append("wandb")
        
    accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[ddp_scaler], log_with=metrics,)
    # accelerator = Accelerator(mixed_precision="fp16", log_with=metrics,)

    # モデルを読み込む
    pipe = StableDiffusionPipeline.from_pretrained(config.model_path, tokenizer=None, safety_checker=None)
    text_encoder, vae, unet = pipe.text_encoder, pipe.vae, pipe.unet
    del pipe

    # モデルに xformers とか memory efficient attention を組み込む
    unet.enable_xformers_memory_efficient_attention()

    # prepare network
    feature_blocks = hook_unet(unet)
    edge_predictor = LatentEdgePredictor(9320, 4, 9)
        
    # generator
    torchtext.utils.download_from_url("https://huggingface.co/datasets/nyanko7/tmp-public/resolve/main/netG.pth", root="./weights/")
    sketch_generator =  create_model()
    sketch_generator.eval()

    optimizer = bnb.optim.AdamW8bit(
        edge_predictor.parameters(),
        **config.optimizer.params
    )

    # dataloaderを準備する
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        num_workers=config.dataset.num_workers,
        batch_size=config.batch_size,
        shuffle=True,
        persistent_workers=True,
    )

    # 学習ステップ数を計算する
    max_train_steps = config.train_epochs * len(train_dataloader)
    
    # lr schedulerを用意する
    lr_scheduler = diffusers.optimization.get_scheduler(
        "constant_with_warmup",
        optimizer,
        num_warmup_steps=150,
        num_training_steps=max_train_steps,
    )

    unet, edge_predictor, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, edge_predictor, optimizer, train_dataloader, lr_scheduler
    )


    unet.to(accelerator.device, dtype=torch.float16)
    text_encoder.to(accelerator.device, dtype=torch.float16)
    vae.to(accelerator.device, dtype=torch.float32)
    sketch_generator.to(accelerator.device, dtype=torch.float32)
    edge_predictor.to(accelerator.device, dtype=torch.float32)
    
    unet.eval()
    vae.eval()
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    edge_predictor.requires_grad_(True)
    
    if config.monitor.huggingface_repo and get_local_rank() in [0, -1]:
        from huggingface_hub import Repository
        from huggingface_hub.constants import ENDPOINT
        repo = Repository(
            tempfile.TemporaryDirectory().name,
            clone_from=f"{ENDPOINT}/{config.monitor.huggingface_repo}",
            use_auth_token=config.monitor.huggingface_token,
            revision=None, 
        )

    # resumeする
    if args.resume is not None:
        print(f"resume training from state: {args.resume}")
        accelerator.load_state(args.resume)

    # epoch数を計算する
    num_update_steps_per_epoch = len(train_dataloader)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    progress_bar = tqdm(
        range(max_train_steps),
        smoothing=0,
        disable=not accelerator.is_local_main_process,
        desc="steps",
    )
    global_step = 0

    noise_scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False,
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("network_train")
        
    def get_noise_level(noise, noise_scheduler, timesteps):
        sqrt_one_minus_alpha_prod = (1 - noise_scheduler.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(noise.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
        noise_level = sqrt_one_minus_alpha_prod * noise
        return noise_level

    for epoch in range(num_train_epochs):
        progress_bar.set_description_str(f"Epoch {epoch+1}/{num_train_epochs}")

        loss_total = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(edge_predictor):
                
                input_ids, px = batch[0], batch[1]
                with torch.no_grad():
                    input_ids = input_ids.to(accelerator.device)
                    encoder_hidden_states = encode_tokens(tokenizer, text_encoder, input_ids)
                    latents = vae.encode(px).latent_dist.sample() * 0.18215
                    sketchs = vae.encode(generate_sketch(sketch_generator, px)).latent_dist.sample() * 0.18215

                # Sample noise that we'll add to the latents 
                noise = torch.randn_like(latents, device=latents.device)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), dtype=torch.int64, device=latents.device)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                noise_level = get_noise_level(noise, noise_scheduler, timesteps)

                # Predict the noise residual
                unet(noisy_latents, timesteps, encoder_hidden_states)
                
                intermediate_result = []
                for block in feature_blocks:
                    resized = torch.nn.functional.interpolate(block.output, size=latents.shape[2], mode="bilinear") 
                    intermidiate_result.append(resized)
                    # free vram
                    del block.output
                    
                intermidiate_result = torch.cat(intermediate_result, dim=1)
                result = edge_predictor(intermediate_result, noise_level)
                result = rearrange(result, "(b w h) c -> b c h w", b=bsz, h=sketchs.shape[2], w=sketchs.shape[3])
                loss = torch.nn.functional.mse_loss(result, sketchs, reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            current_loss = loss.detach().item()
            logs = {"loss": current_loss, "lr": lr_scheduler.get_last_lr()[0]}
            accelerator.log(logs, step=global_step)

            loss_total += current_loss
            avr_loss = loss_total / (step + 1)
            logs = {"loss": avr_loss}  
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            ctx = contextlib.nullcontext()
            if config.monitor.huggingface_repo and accelerator.is_main_process:
                ctx = repo.commit(f"add/update model: epoch {epoch}", blocking=False, auto_lfs_prune=True)
                
            with ctx:
                accelerator.save(accelerator.unwrap_model(edge_predictor).state_dict(), "sketch_encoder_model.pt")

        # end of epoch
    accelerator.end_training()


if __name__ == "__main__":
    train()
