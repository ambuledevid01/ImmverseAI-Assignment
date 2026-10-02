"""
I/O utility module for file searching and JSON metadata export.
"""

import os
import json
from typing import List, Dict, Any


def get_image_files(input_path: str) -> List[str]:
    valid_extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if os.path.isfile(input_path):
        if input_path.lower().endswith(valid_extensions):
            return [input_path]
        else:
            raise ValueError(f"File '{input_path}' is not a supported image format.")

    image_paths = []
    for root, _, files in os.walk(input_path):
        for file in sorted(files):
            if file.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, file))

    return image_paths


def save_json_metadata(data: Dict[str, Any], output_path: str):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
