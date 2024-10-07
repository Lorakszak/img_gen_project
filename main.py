"""Main Gradio App defining the interface, callbacks, getting everything together"""

import gradio as gr
import json
from utils.helpers import (
    add_to_prompt,
    get_resolution,
    swap_resolution,
    validate_dimension,
)
from utils.model_inference import generate_image
import logging
import traceback
from utils.device_management import get_device
from config import DEBUG, PORT, SHARE, SSL_KEYFILE, SSL_CERTFILE, setup_logging
from typing import Optional, Tuple, List, Dict

# Ensure logging is set up
setup_logging()

# Load sample styles (for img gen)
with open("assets/sample_styles.json", "r") as f:
    sample_styles = json.load(f)

# Global variable to store the Style (for img gen)
active_style = gr.State(None)


def create_interface(include_face_upload: bool) -> None:
    """
    Creates the Gradio interface for the image generation app.

    Args:
        include_face_upload: Whether to include face upload functionality in the interface.
    """
    with gr.Row():
        with gr.Column(scale=3):

            # Generating image
            with gr.Row():
                checkpoint_dropdown = gr.Dropdown(
                    choices=[
                        "SDXL1.0_Base",
                        "JuggernautXL_v9Rundiffusionphoto2",
                        "YamerMIX",
                    ],
                    label="Checkpoint",
                    value="SDXL1.0_Base",
                )
                generate_button = gr.Button("Generate")

            # Positive Prompt
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here")

            # Negative Prompt
            negative_prompt = gr.Textbox(
                label="Negative Prompt", placeholder="Enter negative prompt here"
            )

            # Styles Dropdown
            style_names = ["None"] + [style["name"] for style in sample_styles]
            style_dropdown = gr.Dropdown(
                choices=style_names, label="Style", value="None"
            )

            # Sampling Method
            sampling_methods = [
                "Euler a",
                "DPM++ 2M Karras",
                "DPM++ SDE Karras",
            ]
            sampling_method = gr.Dropdown(
                sampling_methods, label="Sampling Method", value="Euler a"
            )

            # Sampling Steps
            sampling_steps = gr.Slider(
                minimum=1, maximum=100, step=1, value=30, label="Sampling Steps"
            )

            # CFG Scale
            cfg_scale = gr.Slider(
                minimum=1, maximum=20, step=0.1, value=7.0, label="CFG Scale"
            )

            # Width and Height
            width_slider = gr.Slider(
                minimum=64, maximum=2048, step=8, value=1024, label="Width"
            )
            height_slider = gr.Slider(
                minimum=64, maximum=2048, step=8, value=1024, label="Height"
            )

            # Aspect Ratio and Reverse Resolution
            with gr.Row():
                aspect_ratio_buttons = [
                    gr.Button(ratio, size="sm")
                    for ratio in ["1:1", "2:3", "4:5", "9:16"]
                ]
            reverse_button = gr.Button("Reverse Resolution", size="sm")

            # Seed
            seed = gr.Number(value=-1, label="Seed (-1 for random)")

            # Face Upload (Only for Avatars)
            if include_face_upload:
                face_image = gr.Image(label="Upload Your Face", type="filepath")
            else:
                face_image = None

            # ControlNet Options
            with gr.Accordion("ControlNet Options", open=False):
                controlnet_enabled = gr.Checkbox(label="Enable ControlNet", value=False)
                controlnet_option = gr.Dropdown(
                    ["Canny", "Depth"],
                    label="ControlNet Type",
                    value="Canny",
                    interactive=True,
                    multiselect=True,
                )
                controlnet_image = gr.Image(label="ControlNet Image", type="filepath")
                controlnet_conditioning_scale = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    step=0.01,
                    value=0.6,
                    label="ControlNet Conditioning Scale",
                )

            # Generate Button
            generate_inputs = [
                prompt,
                negative_prompt,
                style_dropdown,
                sampling_method,
                sampling_steps,
                cfg_scale,
                width_slider,
                height_slider,
                seed,
                controlnet_enabled,
                controlnet_option if controlnet_enabled else None,
                controlnet_image if controlnet_enabled else None,
                controlnet_conditioning_scale if controlnet_enabled else None,
                checkpoint_dropdown,
            ]

        with gr.Column(scale=2):
            output_image = gr.Image(label="Generated Image")
            discard_button = gr.Button("Discard Image")

    ### Callbacks Definitions ###

    def update_dimensions(aspect_ratio: str) -> Tuple[int, int]:
        """
        Updates the width and height sliders based on the selected aspect ratio.

        Args:
            aspect_ratio: The selected aspect ratio (e.g., '1:1', '2:3').

        Returns:
            Tuple of (width, height) corresponding to the selected aspect ratio.
        """
        w, h = get_resolution(aspect_ratio)
        return w, h

    def reverse_dimensions(w: int, h: int) -> Tuple[int, int]:
        """
        Reverses the width and height dimensions.

        Args:
            w: The current width value.
            h: The current height value.

        Returns:
            Tuple of (height, width) to swap the resolution.
        """
        return h, w

    def generate_with_error_handling(
        prompt_text: str,
        negative_text: str,
        style_name: str,
        method: str,
        steps: int,
        cfg: float,
        w: int,
        h: int,
        s: int,
        cnet_enabled: bool,
        cnet_option: Optional[List[str]],
        cnet_image: Optional[str],
        cnet_conditioning_scale: Optional[float],
        checkpoint: str,
    ) -> Optional[str]:
        """
        Generates an image using the provided inputs.

        Args:
            prompt_text: The text prompt for image generation.
            negative_text: Negative prompt to avoid certain features.
            style_name: The selected style for image generation.
            method: The sampling method to use.
            steps: The number of sampling steps.
            cfg: Classifier-free guidance scale.
            w: Image width in pixels.
            h: Image height in pixels.
            s: Random seed for reproducibility.
            cnet_enabled: Whether ControlNet is enabled.
            cnet_option: ControlNet options, e.g., 'Canny' or 'Depth' or both.
            cnet_image: The image used for ControlNet conditioning.
            cnet_conditioning_scale: ControlNet conditioning scale.
            checkpoint: The selected model checkpoint.

        Returns:
            The generated image as a file path or None in case of an error.
        """
        try:
            # Validate dimensions
            w = validate_dimension(int(w))
            h = validate_dimension(int(h))

            # Apply style if selected
            if style_name != "None":
                selected_style = next(
                    (style for style in sample_styles if style["name"] == style_name),
                    None,
                )
                if selected_style:
                    prompt_text = selected_style["prompt"].replace(
                        "{prompt}", prompt_text
                    )
                    negative_text = selected_style["negative_prompt"].replace(
                        "{prompt}", negative_text
                    )

            # Prepare inputs,
            inputs = {
                "prompt": prompt_text,
                "negative_prompt": negative_text,
                "sampling_method": method,
                "sampling_steps": steps,
                "cfg_scale": cfg,
                "width": w,
                "height": h,
                "seed": s,
                "controlnet_enabled": cnet_enabled,
                "controlnet_option": cnet_option,
                "controlnet_image": cnet_image,
                "controlnet_conditioning_scale": cnet_conditioning_scale,
                "checkpoint": checkpoint,
            }

            # Call model inference
            image = generate_image(**inputs)
            return image
        except Exception as e:
            logging.error(f"Error during image generation: {str(e)}")
            logging.error(traceback.format_exc())
            return None

    # Aspect Ratio Buttons
    for button in aspect_ratio_buttons:
        button.click(
            update_dimensions,
            inputs=button,
            outputs=[width_slider, height_slider],
        )

    # Reverse Resolution Button
    reverse_button.click(
        reverse_dimensions,
        inputs=[width_slider, height_slider],
        outputs=[width_slider, height_slider],
    )

    # Generate Button
    generate_button.click(
        generate_with_error_handling, inputs=generate_inputs, outputs=output_image
    )

    # Discard Buttons
    discard_button.click(lambda: None, inputs=None, outputs=output_image)


def main() -> None:
    """Main point for the Gradio app, creating tabs for different modes, running the server"""
    with gr.Blocks(css=".gradio-container { width: 100vw; }") as demo:
        with gr.Tabs():
            with gr.TabItem("Style Transfer"):
                create_interface(include_face_upload=False)
            with gr.TabItem("Avatars", visible=False):
                create_interface(include_face_upload=True)
        logging.debug(f"MAIN APP run, Debug:{DEBUG}, Server_Port:{PORT}, SHARE:{SHARE}")
        if DEBUG:
            demo.launch(
                debug=DEBUG,
                server_port=PORT,
                share=SHARE,
            )
        else:
            demo.launch(
                debug=DEBUG,
                server_port=PORT,
                share=SHARE,
                ssl_verify=False,
                server_name="0.0.0.0",
                ssl_certfile=SSL_CERTFILE,
                ssl_keyfile=SSL_KEYFILE,
            )


if __name__ == "__main__":
    main()
