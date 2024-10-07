"""
This module handles loading the model pipelines, preprocessing images,
and generating images.
"""

import torch
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
)
from PIL import Image
import cv2
import numpy as np
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from functools import lru_cache
from utils.device_management import get_device, cleanup_memory
from config import FORCE_CPU, setup_logging
import logging
from typing import Optional, List

# Ensure logging is set up
setup_logging()


def load_pipeline(
    checkpoint: str,
    sampling_method: str,
    use_controlnet: bool = False,
    use_canny: bool = False,
    use_depth: bool = False,
) -> StableDiffusionXLControlNetPipeline:
    """
    Loads the appropriate Stable Diffusion pipeline based on the checkpoint
    and optional ControlNet settings.

    Args:
        checkpoint: The name of the checkpoint to load.
        sampling_method: The sampling method to use.
        use_controlnet: Whether to enable ControlNet.
        use_canny: Whether to enable Canny-based ControlNet.
        use_depth: Whether to enable Depth-based ControlNet.

    Returns:
        The configured pipeline object.
    """

    if checkpoint == "JuggernautXL_v9Rundiffusionphoto2":
        model_path = (
            "./assets/Juggernaut_XL/juggernautXL_v9Rundiffusionphoto2.safetensors"
        )
    elif checkpoint == "YamerMIX":
        model_path = "./assets/YamerMIX/sdxlUnstableDiffusers_nihilmania.safetensors"
    else:
        model_path = "./assets/SDXL_base_1_0/model.safetensors"

    controlnet = []
    if use_controlnet:
        if use_canny:
            canny_controlnet = ControlNetModel.from_pretrained(
                "./assets/Controlnets/canny_XL",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
            controlnet.append(canny_controlnet)
        if use_depth:
            depth_controlnet = ControlNetModel.from_pretrained(
                "./assets/Controlnets/depth_XL",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
            controlnet.append(depth_controlnet)

        pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
    else:
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

    device = get_device()
    if FORCE_CPU:
        device = torch.device(
            "cpu"
        )  # TODO ENSURE IT"S NOT RUNNING ON CPU -> NOT SUPPORTED DUE TO float16 FORMAT
    pipe = pipe.to(device)

    if device.type == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()

    if sampling_method.lower() == "DPM++ 2M Karras".lower():
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
        )
    elif sampling_method.lower() == "DPM++ SDE Karras".lower():
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )
    elif sampling_method.lower() == "Euler a".lower():
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    return pipe


def preprocess_image(
    image: Image.Image, target_width: int, target_height: int
) -> Image.Image:
    """
    Preprocesses an image by cropping it to the target aspect ratio and resizing it.

    Args:
        image: The input image to be processed.
        target_width: The target width after resizing.
        target_height: The target height after resizing.

    Returns:
        The processed image.
    """
    # Calculate the aspect ratio
    aspect_ratio = image.width / image.height
    target_aspect_ratio = target_width / target_height

    if aspect_ratio > target_aspect_ratio:
        # Image is wider, crop the width
        new_width = int(target_aspect_ratio * image.height)
        left = (image.width - new_width) // 2
        image = image.crop((left, 0, left + new_width, image.height))
    elif aspect_ratio < target_aspect_ratio:
        # Image is taller, crop the height
        new_height = int(image.width / target_aspect_ratio)
        top = (image.height - new_height) // 2
        image = image.crop((0, top, image.width, top + new_height))

    # Resize the image to the target dimensions
    image = image.resize((target_width, target_height), Image.LANCZOS)
    return image


def preprocess_canny(
    image: Image.Image, low_threshold: int = 100, high_threshold: int = 200
) -> Image.Image:
    """
    Applies Canny edge detection to the given image.

    Args:
        image: The input image to process.
        low_threshold: The lower bound for the Canny edge detection.
        high_threshold: The upper bound for the Canny edge detection.

    Returns:
        The processed image with Canny edges.
    """
    canny_image = cv2.Canny(np.array(image), low_threshold, high_threshold)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    return Image.fromarray(canny_image)


def preprocess_depth(image: Image.Image) -> Image.Image:
    """
    Estimates the depth of the image using a depth estimation model.

    Args:
        image: The input image for depth estimation.

    Returns:
        The processed depth image.
    """
    depth_estimator = DPTForDepthEstimation.from_pretrained(
        "Intel/dpt-hybrid-midas"
    ).to("cuda")
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    return Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))


def generate_image(
    prompt: str,
    negative_prompt: str,
    sampling_method: str = "Euler a",
    sampling_steps: int = 50,
    cfg_scale: float = 7.5,
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
    controlnet_enabled: bool = False,
    controlnet_option: Optional[str] = "Canny",
    controlnet_image: Optional[str] = None,
    controlnet_conditioning_scale: float = 0.8,
    checkpoint: str = "SDXL1.0_Base",
) -> Optional[Image.Image]:
    """
    Generates an image using the specified parameters and optional ControlNet settings.

    Args:
        prompt: The text prompt for image generation.
        negative_prompt: The negative text prompt to avoid in generation.
        sampling_method: The sampling method to use.
        sampling_steps: Number of inference steps.
        cfg_scale: Classifier-free guidance scale.
        width: Width of the generated image.
        height: Height of the generated image.
        seed: Random seed for generation (-1 for random).
        controlnet_enabled: Whether ControlNet is enabled.
        controlnet_option: ControlNet option to use (e.g., 'Canny', 'Depth').
        controlnet_image: Path to the image used for ControlNet conditioning.
        controlnet_conditioning_scale: Scale for ControlNet conditioning.
        checkpoint: The model checkpoint to use.

    Returns:
        The generated image, or None if an error occurred.
    """

    use_canny = controlnet_enabled and "Canny" in controlnet_option
    use_depth = controlnet_enabled and "Depth" in controlnet_option

    logging.debug(
        f"""Checkpoint: {checkpoint}, Prompt: {prompt}, Negative Prompt: {negative_prompt}, Sampling Method: {sampling_method}, 
    Sampling Steps: {sampling_steps}, CFG Scale: {cfg_scale}, Width: {width}, Height: {height}, 
    Seed: {seed}, ControlNet Enabled: {controlnet_enabled}, ControlNet Option: {controlnet_option}, 
    ControlNet Image: {controlnet_image}, ControlNet Conditioning Scale: {controlnet_conditioning_scale}, use canny/depth {use_canny}/{use_depth}"""
    )

    pipe = load_pipeline(
        checkpoint,
        sampling_method,
        use_controlnet=controlnet_enabled,
        use_canny=use_canny,
        use_depth=use_depth,
    )

    generator = torch.manual_seed(
        seed if seed != -1 else torch.randint(0, 1000000, (1,)).item()
    )

    controlnet_images = []
    if controlnet_enabled and controlnet_image:
        input_image = Image.open(controlnet_image).convert("RGB")
        input_image = preprocess_image(input_image, width, height)

        if use_canny:
            canny_image = preprocess_canny(input_image)
            controlnet_images.append(canny_image)
        if use_depth:
            depth_image = preprocess_depth(input_image)
            controlnet_images.append(depth_image)

    with torch.no_grad():
        try:
            if controlnet_images and controlnet_enabled:
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=controlnet_images,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    num_inference_steps=sampling_steps,
                    guidance_scale=cfg_scale,
                    height=height,
                    width=width,
                    generator=generator,
                ).images[0]
            else:
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=sampling_steps,
                    guidance_scale=cfg_scale,
                    height=height,
                    width=width,
                    generator=generator,
                ).images[0]
        except torch.cuda.OutOfMemoryError:
            logging.error(
                "CUDA Out of Memory! Please reduce the image resolution, number of inference steps, or disable controlnets."
            )
            torch.cuda.empty_cache()
            return None
        finally:
            cleanup_memory()

    return image
