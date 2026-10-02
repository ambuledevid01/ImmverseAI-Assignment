"""
Dataset loader module for historical manuscript images and annotations.
"""

import os
from typing import List, Dict, Any


class ManuscriptDatasetLoader:
    def __init__(self, data_dir: str = "./data/raw", annotations_dir: str = "./data/annotations"):
        self.data_dir = data_dir
        self.annotations_dir = annotations_dir
        self.classes = ["header", "footer", "main_text", "side_text", "filler"]

    def list_image_files(self) -> List[str]:
        if not os.path.exists(self.data_dir):
            return []
        valid_extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
        image_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_files.append(os.path.join(root, file))
        return image_files

    def load_annotations(self, annotation_file: str) -> Dict[str, Any]:
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        return {"images": [], "annotations": [], "categories": self.classes}
