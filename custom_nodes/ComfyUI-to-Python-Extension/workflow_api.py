import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


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


def main():
    import_custom_nodes()
    with torch.inference_mode():
        intconstant = NODE_CLASS_MAPPINGS["INTConstant"]()
        intconstant_83 = intconstant.get_value(value=1024)

        intconstant_84 = intconstant.get_value(value=1024)

        dualcliploader = DualCLIPLoader()
        dualcliploader_357 = dualcliploader.load_clip(
            clip_name1="t5/t5xxl_fp16.safetensors",
            clip_name2="clip_l.safetensors",
            type="flux",
        )

        cr_clip_input_switch = NODE_CLASS_MAPPINGS["CR Clip Input Switch"]()
        cr_clip_input_switch_319 = cr_clip_input_switch.switch(
            Input=1,
            clip1=get_value_at_index(dualcliploader_357, 0),
            clip2=get_value_at_index(dualcliploader_357, 0),
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_174 = cliptextencode.encode(
            text="a girl looking at a house on fire",
            clip=get_value_at_index(cr_clip_input_switch_319, 0),
        )

        cliptextencode_175 = cliptextencode.encode(
            text="", clip=get_value_at_index(cr_clip_input_switch_319, 0)
        )

        loadimage = LoadImage()
        loadimage_429 = loadimage.load_image(
            image="7038548d-d204-4810-bb74-d1dea277200a.png"
        )

        imageresize = NODE_CLASS_MAPPINGS["ImageResize+"]()
        imageresize_72 = imageresize.execute(
            width=get_value_at_index(intconstant_83, 0),
            height=get_value_at_index(intconstant_84, 0),
            interpolation="bicubic",
            method="keep proportion",
            condition="always",
            multiple_of=16,
            image=get_value_at_index(loadimage_429, 0),
        )

        getimagesizeandcount = NODE_CLASS_MAPPINGS["GetImageSizeAndCount"]()
        getimagesizeandcount_360 = getimagesizeandcount.getsize(
            image=get_value_at_index(imageresize_72, 0)
        )

        vaeloader = VAELoader()
        vaeloader_359 = vaeloader.load_vae(vae_name="FLUX1/ae.safetensors")

        vaeencode = VAEEncode()
        vaeencode_197 = vaeencode.encode(
            pixels=get_value_at_index(getimagesizeandcount_360, 0),
            vae=get_value_at_index(vaeloader_359, 0),
        )

        unetloader = UNETLoader()
        unetloader_358 = unetloader.load_unet(
            unet_name="flux1-depth-dev.safetensors", weight_dtype="default"
        )

        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        ksamplerselect_363 = ksamplerselect.get_sampler(sampler_name="euler")

        randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        randomnoise_365 = randomnoise.get_noise(noise_seed=random.randint(1, 2**64))

        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        fluxguidance_430 = fluxguidance.append(
            guidance=15, conditioning=get_value_at_index(cliptextencode_174, 0)
        )

        downloadandloaddepthanythingv2model = NODE_CLASS_MAPPINGS[
            "DownloadAndLoadDepthAnythingV2Model"
        ]()
        downloadandloaddepthanythingv2model_437 = (
            downloadandloaddepthanythingv2model.loadmodel(
                model="depth_anything_v2_vitl_fp32.safetensors"
            )
        )

        depthanything_v2 = NODE_CLASS_MAPPINGS["DepthAnything_V2"]()
        depthanything_v2_436 = depthanything_v2.process(
            da_model=get_value_at_index(downloadandloaddepthanythingv2model_437, 0),
            images=get_value_at_index(getimagesizeandcount_360, 0),
        )

        instructpixtopixconditioning = NODE_CLASS_MAPPINGS[
            "InstructPixToPixConditioning"
        ]()
        instructpixtopixconditioning_431 = instructpixtopixconditioning.encode(
            positive=get_value_at_index(fluxguidance_430, 0),
            negative=get_value_at_index(cliptextencode_175, 0),
            vae=get_value_at_index(vaeloader_359, 0),
            pixels=get_value_at_index(depthanything_v2_436, 0),
        )

        clipvisionloader = CLIPVisionLoader()
        clipvisionloader_438 = clipvisionloader.load_clip(
            clip_name="sigclip_vision_patch14_384.safetensors"
        )

        loadimage_440 = loadimage.load_image(
            image="2013_CKS_01180_0005_000(the_court_of_pir_budaq_shiraz_iran_circa_1455-60074106).jpg"
        )

        clipvisionencode = CLIPVisionEncode()
        clipvisionencode_439 = clipvisionencode.encode(
            crop="center",
            clip_vision=get_value_at_index(clipvisionloader_438, 0),
            image=get_value_at_index(loadimage_440, 0),
        )

        stylemodelloader = StyleModelLoader()
        stylemodelloader_441 = stylemodelloader.load_style_model(
            style_model_name="flux1-redux-dev.safetensors"
        )

        cr_text = NODE_CLASS_MAPPINGS["CR Text"]()
        cr_text_456 = cr_text.text_multiline(text="Flux_BFL_Depth_Redux")

        emptylatentimage = EmptyLatentImage()
        cr_conditioning_input_switch = NODE_CLASS_MAPPINGS[
            "CR Conditioning Input Switch"
        ]()
        cr_model_input_switch = NODE_CLASS_MAPPINGS["CR Model Input Switch"]()
        stylemodelapplyadvanced = NODE_CLASS_MAPPINGS["StyleModelApplyAdvanced"]()
        basicguider = NODE_CLASS_MAPPINGS["BasicGuider"]()
        basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
        samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        vaedecode = VAEDecode()
        saveimage = SaveImage()
        imagecrop = NODE_CLASS_MAPPINGS["ImageCrop+"]()

        for q in range(10):
            emptylatentimage_10 = emptylatentimage.generate(
                width=get_value_at_index(imageresize_72, 1),
                height=get_value_at_index(imageresize_72, 2),
                batch_size=1,
            )

            cr_conditioning_input_switch_271 = cr_conditioning_input_switch.switch(
                Input=1,
                conditioning1=get_value_at_index(instructpixtopixconditioning_431, 0),
                conditioning2=get_value_at_index(instructpixtopixconditioning_431, 0),
            )

            cr_conditioning_input_switch_272 = cr_conditioning_input_switch.switch(
                Input=1,
                conditioning1=get_value_at_index(instructpixtopixconditioning_431, 1),
                conditioning2=get_value_at_index(instructpixtopixconditioning_431, 1),
            )

            cr_model_input_switch_320 = cr_model_input_switch.switch(
                Input=1,
                model1=get_value_at_index(unetloader_358, 0),
                model2=get_value_at_index(unetloader_358, 0),
            )

            stylemodelapplyadvanced_442 = stylemodelapplyadvanced.apply_stylemodel(
                strength=0.5,
                conditioning=get_value_at_index(instructpixtopixconditioning_431, 0),
                style_model=get_value_at_index(stylemodelloader_441, 0),
                clip_vision_output=get_value_at_index(clipvisionencode_439, 0),
            )

            basicguider_366 = basicguider.get_guider(
                model=get_value_at_index(cr_model_input_switch_320, 0),
                conditioning=get_value_at_index(stylemodelapplyadvanced_442, 0),
            )

            basicscheduler_364 = basicscheduler.get_sigmas(
                scheduler="simple",
                steps=28,
                denoise=1,
                model=get_value_at_index(cr_model_input_switch_320, 0),
            )

            samplercustomadvanced_362 = samplercustomadvanced.sample(
                noise=get_value_at_index(randomnoise_365, 0),
                guider=get_value_at_index(basicguider_366, 0),
                sampler=get_value_at_index(ksamplerselect_363, 0),
                sigmas=get_value_at_index(basicscheduler_364, 0),
                latent_image=get_value_at_index(emptylatentimage_10, 0),
            )

            vaedecode_321 = vaedecode.decode(
                samples=get_value_at_index(samplercustomadvanced_362, 0),
                vae=get_value_at_index(vaeloader_359, 0),
            )

            saveimage_327 = saveimage.save_images(
                filename_prefix=get_value_at_index(cr_text_456, 0),
                images=get_value_at_index(vaedecode_321, 0),
            )

            fluxguidance_382 = fluxguidance.append(
                guidance=4,
                conditioning=get_value_at_index(cr_conditioning_input_switch_272, 0),
            )

            imagecrop_447 = imagecrop.execute(
                width=2000,
                height=2000,
                position="top-center",
                x_offset=0,
                y_offset=0,
                image=get_value_at_index(loadimage_440, 0),
            )


if __name__ == "__main__":
    main()
