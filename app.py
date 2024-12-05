import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import gradio as gr
from PIL import Image
from huggingface_hub import hf_hub_download
import spaces

hf_hub_download(repo_id="black-forest-labs/FLUX.1-Redux-dev", filename="flux1-redux-dev.safetensors", local_dir="models/style_models")
hf_hub_download(repo_id="black-forest-labs/FLUX.1-Depth-dev", filename="flux1-depth-dev.safetensors", local_dir="models/diffusion_models")
hf_hub_download(repo_id="Comfy-Org/sigclip_vision_384", filename="sigclip_vision_patch14_384.safetensors", local_dir="models/clip_vision")
hf_hub_download(repo_id="Kijai/DepthAnythingV2-safetensors", filename="depth_anything_v2_vitl_fp32.safetensors", local_dir="models/depthanything")
hf_hub_download(repo_id="black-forest-labs/FLUX.1-dev", filename="ae.safetensors", local_dir="models/vae/FLUX1")
hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", filename="clip_l.safetensors", local_dir="models/text_encoders")
t5_path = hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", filename="t5xxl_fp16.safetensors", local_dir="models/text_encoders/t5")

# Import all the necessary functions from the original script
def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

# Add all the necessary setup functions from the original script
def find_path(name: str, path: str = None) -> str:
    if path is None:
        path = os.getcwd()
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name
    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None
    return find_path(name, parent_directory)

def add_comfyui_directory_to_sys_path() -> None:
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")

def add_extra_model_paths() -> None:
    try:
        from main import load_extra_path_config
    except ImportError:
        from utils.extra_config import load_extra_path_config
    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")

# Initialize paths
add_comfyui_directory_to_sys_path()
add_extra_model_paths()

def import_custom_nodes() -> None:
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    init_extra_nodes()

# Import all necessary nodes
from nodes import (
    StyleModelLoader,
    VAEEncode,
    NODE_CLASS_MAPPINGS,
    LoadImage,
    CLIPVisionLoader,
    SaveImage,
    VAELoader,
    CLIPVisionEncode,
    DualCLIPLoader,
    EmptyLatentImage,
    VAEDecode,
    UNETLoader,
    CLIPTextEncode,
)

# Initialize all constant nodes and models in global context
import_custom_nodes()

# Global variables for preloaded models and constants
#with torch.inference_mode():
    # Initialize constants
intconstant = NODE_CLASS_MAPPINGS["INTConstant"]()
CONST_1024 = intconstant.get_value(value=1024)

# Load CLIP
dualcliploader = DualCLIPLoader()
CLIP_MODEL = dualcliploader.load_clip(
    clip_name1="t5/t5xxl_fp16.safetensors",
    clip_name2="clip_l.safetensors",
    type="flux",
)

# Load VAE
vaeloader = VAELoader()
VAE_MODEL = vaeloader.load_vae(vae_name="FLUX1/ae.safetensors")

# Load UNET
unetloader = UNETLoader()
UNET_MODEL = unetloader.load_unet(
    unet_name="flux1-depth-dev.safetensors", weight_dtype="default"
)

# Load CLIP Vision
clipvisionloader = CLIPVisionLoader()
CLIP_VISION_MODEL = clipvisionloader.load_clip(
    clip_name="sigclip_vision_patch14_384.safetensors"
)

# Load Style Model
stylemodelloader = StyleModelLoader()
STYLE_MODEL = stylemodelloader.load_style_model(
    style_model_name="flux1-redux-dev.safetensors"
)

# Initialize samplers
ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
SAMPLER = ksamplerselect.get_sampler(sampler_name="euler")

# Initialize depth model
cr_clip_input_switch = NODE_CLASS_MAPPINGS["CR Clip Input Switch"]()
downloadandloaddepthanythingv2model = NODE_CLASS_MAPPINGS["DownloadAndLoadDepthAnythingV2Model"]()
DEPTH_MODEL = downloadandloaddepthanythingv2model.loadmodel(
    model="depth_anything_v2_vitl_fp32.safetensors"
)
cliptextencode = CLIPTextEncode()
loadimage = LoadImage()
vaeencode = VAEEncode()
fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
instructpixtopixconditioning = NODE_CLASS_MAPPINGS["InstructPixToPixConditioning"]()
clipvisionencode = CLIPVisionEncode()
stylemodelapplyadvanced = NODE_CLASS_MAPPINGS["StyleModelApplyAdvanced"]()
emptylatentimage = EmptyLatentImage()
basicguider = NODE_CLASS_MAPPINGS["BasicGuider"]()
basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()        
randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
vaedecode = VAEDecode()
cr_text = NODE_CLASS_MAPPINGS["CR Text"]()
saveimage = SaveImage()
getimagesizeandcount = NODE_CLASS_MAPPINGS["GetImageSizeAndCount"]()
depthanything_v2 = NODE_CLASS_MAPPINGS["DepthAnything_V2"]()
imageresize = NODE_CLASS_MAPPINGS["ImageResize+"]()

@spaces.GPU
def generate_image(prompt, structure_image, style_image, depth_strength=15, style_strength=0.5, progress=gr.Progress(track_tqdm=True)) -> str:
    """Main generation function that processes inputs and returns the path to the generated image."""
    with torch.inference_mode():
        # Set up CLIP
        clip_switch = cr_clip_input_switch.switch(
            Input=1,
            clip1=get_value_at_index(CLIP_MODEL, 0),
            clip2=get_value_at_index(CLIP_MODEL, 0),
        )
        
        # Encode text
        text_encoded = cliptextencode.encode(
            text=prompt,
            clip=get_value_at_index(clip_switch, 0),
        )
        empty_text = cliptextencode.encode(
            text="",
            clip=get_value_at_index(clip_switch, 0),
        )
        
        # Process structure image
        structure_img = loadimage.load_image(image=structure_image)
        
        # Resize image
        resized_img = imageresize.execute(
            width=get_value_at_index(CONST_1024, 0),
            height=get_value_at_index(CONST_1024, 0),
            interpolation="bicubic",
            method="keep proportion",
            condition="always",
            multiple_of=16,
            image=get_value_at_index(structure_img, 0),
        )
        
        # Get image size
        size_info = getimagesizeandcount.getsize(
            image=get_value_at_index(resized_img, 0)
        )
        
        # Encode VAE
        vae_encoded = vaeencode.encode(
            pixels=get_value_at_index(size_info, 0),
            vae=get_value_at_index(VAE_MODEL, 0),
        )
        
        # Process depth
        depth_processed = depthanything_v2.process(
            da_model=get_value_at_index(DEPTH_MODEL, 0),
            images=get_value_at_index(size_info, 0),
        )
        
        # Apply Flux guidance
        flux_guided = fluxguidance.append(
            guidance=depth_strength,
            conditioning=get_value_at_index(text_encoded, 0),
        )
        
        # Process style image
        style_img = loadimage.load_image(image=style_image)
        
        # Encode style with CLIP Vision
        style_encoded = clipvisionencode.encode(
            crop="center",
            clip_vision=get_value_at_index(CLIP_VISION_MODEL, 0),
            image=get_value_at_index(style_img, 0),
        )
        
        # Set up conditioning
        conditioning = instructpixtopixconditioning.encode(
            positive=get_value_at_index(flux_guided, 0),
            negative=get_value_at_index(empty_text, 0),
            vae=get_value_at_index(VAE_MODEL, 0),
            pixels=get_value_at_index(depth_processed, 0),
        )
        
        # Apply style
        style_applied = stylemodelapplyadvanced.apply_stylemodel(
            strength=style_strength,
            conditioning=get_value_at_index(conditioning, 0),
            style_model=get_value_at_index(STYLE_MODEL, 0),
            clip_vision_output=get_value_at_index(style_encoded, 0),
        )
        
        # Set up empty latent
        empty_latent = emptylatentimage.generate(
            width=get_value_at_index(resized_img, 1),
            height=get_value_at_index(resized_img, 2),
            batch_size=1,
        )
        
        # Set up guidance
        guided = basicguider.get_guider(
            model=get_value_at_index(UNET_MODEL, 0),
            conditioning=get_value_at_index(style_applied, 0),
        )
        
        # Set up scheduler
        schedule = basicscheduler.get_sigmas(
            scheduler="simple",
            steps=28,
            denoise=1,
            model=get_value_at_index(UNET_MODEL, 0),
        )
        
        # Generate random noise
        noise = randomnoise.get_noise(noise_seed=random.randint(1, 2**64))
        
        # Sample
        sampled = samplercustomadvanced.sample(
            noise=get_value_at_index(noise, 0),
            guider=get_value_at_index(guided, 0),
            sampler=get_value_at_index(SAMPLER, 0),
            sigmas=get_value_at_index(schedule, 0),
            latent_image=get_value_at_index(empty_latent, 0),
        )
        
        # Decode VAE
        decoded = vaedecode.decode(
            samples=get_value_at_index(sampled, 0),
            vae=get_value_at_index(VAE_MODEL, 0),
        )
        
        # Save image
        prefix = cr_text.text_multiline(text="Flux_BFL_Depth_Redux")
        
        saved = saveimage.save_images(
            filename_prefix=get_value_at_index(prefix, 0),
            images=get_value_at_index(decoded, 0),
        )
        saved_path = f"output/{saved['ui']['images'][0]['filename']}"
        return saved_path

# Create Gradio interface

examples = [
    ["", "mona.png", "receita-tacos.webp", 15, 0.6],
    ["a woman looking at a house catching fire on the background", "disaster_girl.png", "abaporu.jpg", 15, 0.15],
    ["istanbul aerial, dramatic photography", "natasha.png", "istambul.jpg", 15, 0.5],
]

output_image = gr.Image(label="Generated Image")

with gr.Blocks() as app:
    gr.Markdown("# FLUX Style Shaping")
    gr.Markdown("Flux[dev] Redux + Flux[dev] Depth ComfyUI workflow by [Nathan Shipley](https://x.com/CitizenPlain) running directly on Gradio. [workflow](https://gist.github.com/nathanshipley/7a9ac1901adde76feebe58d558026f68) - [how to convert your any comfy workflow to gradio (soon)](#)")
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
            with gr.Row():
                with gr.Group():
                    structure_image = gr.Image(label="Structure Image", type="filepath")
                    depth_strength = gr.Slider(minimum=0, maximum=50, value=15, label="Depth Strength")
                with gr.Group():
                    style_image = gr.Image(label="Style Image", type="filepath")
                    style_strength = gr.Slider(minimum=0, maximum=1, value=0.5, label="Style Strength")
            generate_btn = gr.Button("Generate")
            
            gr.Examples(
                examples=examples,
                inputs=[prompt_input, structure_image, style_image, depth_strength, style_strength],
                outputs=[output_image],
                fn=generate_image,
                cache_examples=True,
                cache_mode="lazy"
            )
        
        with gr.Column():
            output_image.render()
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt_input, structure_image, style_image, depth_strength, style_strength],
        outputs=[output_image]
    )

if __name__ == "__main__":
    app.launch(share=True)