import torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from modules.pipeline import AntiGradientPipeline
from modules.latent_predictor import LatentEdgePredictor
from diffusers.image_processor import VaeImageProcessor
import os

# Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Fix the scheduler configuration
scheduler = DPMSolverMultistepScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    thresholding=False,
    algorithm_type="dpmsolver++",
    solver_type="midpoint",
    lower_order_final=True,
)

# Load the VAE model
vae = AutoencoderKL.from_pretrained(
    "benjamin-paine/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float16
).to("cuda")

# Load the main pipeline
pipe = AntiGradientPipeline.from_pretrained(
    "./AbyssOrangeMix_directory",
    vae=vae,
    torch_dtype=torch.float16,
    scheduler=scheduler,
).to("cuda")

# Preprocessing for the PNG sketch
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Load and preprocess the PNG sketch
image_path = "sketch1.png"  # Replace with your sketch file path
try:
    sketch = Image.open(image_path).convert("RGB").resize((512, 512))  # Ensure 512x512 size
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find the sketch file: {image_path}")

tensor_img = transform(sketch).unsqueeze(0).to(vae.device, dtype=vae.dtype)

# Encode the sketch into latent space
latent_sketch = vae.encode(tensor_img).latent_dist.sample() * 0.18215

# Load LatentEdgePredictor
lgp = LatentEdgePredictor(9320, 4, 9)
lgp.load_state_dict(torch.load("./edge_predictor.pt"))
lgp.to(pipe.unet.device, dtype=pipe.unet.dtype)
pipe.setup_lgp(lgp)

# Define the prompt for image generation
prompt = "A futuristic cityscape inspired by the sketch"
negative_prompt = ""  # Add any negative prompts here, if needed

# Generate an image
try:
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,  # Number of diffusion steps
        guidance_scale=7.5,      # Guidance scale
        sketch_image=latent_sketch,
        width=512,               # Width of output image
        height=512,              # Height of output image
    )
    generated_image = result[0]
    generated_image.save("output.png")  # Save the generated image
    print("Image generated successfully and saved as 'output.png'.")
except Exception as e:
    print(f"An error occurred during image generation: {e}")
