"""
This module provides helper functions for modifying prompts and managing image resolutions.
"""

from typing import Tuple


def add_to_prompt(prompt: str, addition: str, check_existing: bool = False) -> str:
    """
    Adds a specified addition to the prompt, optionally checking if the addition already exists.

    Args:
        prompt: The original prompt.
        addition: The string to add to the prompt.
        check_existing: If True, ensures the addition is not already in the prompt.

    Returns:
        The updated prompt with the addition appended.
    """
    if check_existing:
        # Split the prompt into individual keywords
        existing_keywords = [k.strip() for k in prompt.split(",") if k.strip()]
        # Check if the addition already exists in the prompt
        if addition in existing_keywords:
            return prompt  # Return the original prompt without changes

    if prompt:
        return f"{prompt}, {addition}"
    else:
        return addition


def get_resolution(aspect_ratio: str) -> Tuple[int, int]:
    """
    Returns the width and height corresponding to the given aspect ratio.

    Args:
        aspect_ratio: The aspect ratio as a string (e.g., '1:1', '2:3').

    Returns:
        A tuple of (width, height) for the given aspect ratio. Defaults to (1024, 1024).
    """
    resolutions = {
        "1:1": (1024, 1024),
        "2:3": (832, 1216),
        "4:5": (896, 1152),
        "9:16": (768, 1344),
    }
    return resolutions.get(aspect_ratio, (1024, 1024))


def swap_resolution(width: int, height: int) -> Tuple[int, int]:
    """
    Swaps the width and height values.

    Args:
        width: The width value.
        height: The height value.

    Returns:
        A tuple with swapped width and height.
    """
    return height, width


def validate_dimension(value: int) -> int:
    """
    Validates the dimension value by ensuring it is divisible by 8.

    Args:
        value: The dimension value to validate.

    Returns:
        The validated dimension value, adjusted to be divisible by 8.
    """
    return value - (value % 8)
