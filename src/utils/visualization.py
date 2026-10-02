"""
Visualization utility module for rendering bounding boxes and labels.
"""

import cv2
import numpy as np
from typing import List, Dict, Any

CLASS_COLORS_BGR = {
    "header": (71, 99, 255),
    "footer": (255, 144, 30),
    "main_text": (50, 205, 50),
    "side_text": (0, 165, 255),
    "filler": (211, 85, 186)
}


def draw_bounding_boxes(
    image: np.ndarray,
    predictions: List[Dict[str, Any]],
    thickness: int = 2
) -> np.ndarray:
    if image is None:
        raise ValueError("Cannot draw bounding boxes on a None image.")

    annotated_image = image.copy()
    for pred in predictions:
        label = pred.get("label", "unknown")
        bbox = pred.get("bbox", [0, 0, 0, 0])
        score = pred.get("confidence", 0.0)

        x_min, y_min, x_max, y_max = [int(v) for v in bbox]
        color = CLASS_COLORS_BGR.get(label, (0, 255, 255))

        cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), color, thickness)
        label_text = f"{label}: {score:.2f}"

        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = max(y_min - 5, text_height + 5)

        cv2.rectangle(
            annotated_image,
            (x_min, text_y - text_height - 4),
            (x_min + text_width + 4, text_y + baseline),
            color,
            -1
        )
        cv2.putText(
            annotated_image,
            label_text,
            (x_min + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return annotated_image
