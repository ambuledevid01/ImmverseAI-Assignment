"""
Manuscript specific layout region detector for header, footer, main_text, side_text, and filler.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


class LayoutDetector:
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.classes = ["header", "footer", "main_text", "side_text", "filler"]

    def _clamp_bbox(self, bbox: List[int], width: int, height: int) -> List[int]:
        x_min, y_min, x_max, y_max = bbox
        x_min = max(0, min(width - 1, int(x_min)))
        y_min = max(0, min(height - 1, int(y_min)))
        x_max = max(x_min + 1, min(width, int(x_max)))
        y_max = max(y_min + 1, min(height, int(y_max)))
        return [x_min, y_min, x_max, y_max]

    def _classify_region(
        self,
        bbox: List[int],
        img_width: int,
        img_height: int,
        density: float
    ) -> Tuple[str, float]:
        x_min, y_min, x_max, y_max = bbox
        box_w = x_max - x_min
        box_h = y_max - y_min
        box_area = box_w * box_h
        page_area = img_width * img_height

        y_center_norm = ((y_min + y_max) / 2.0) / img_height
        x_center_norm = ((x_min + x_max) / 2.0) / img_width
        rel_area = box_area / max(1.0, page_area)

        if y_center_norm <= 0.22:
            score = min(0.98, 0.75 + (0.22 - y_center_norm) * 0.9 + (box_w / img_width) * 0.15)
            return "header", round(score, 2)

        if y_center_norm >= 0.78:
            score = min(0.98, 0.75 + (y_center_norm - 0.78) * 0.9 + (box_w / img_width) * 0.15)
            return "footer", round(score, 2)

        if (x_center_norm <= 0.18 or x_center_norm >= 0.82) and rel_area < 0.25:
            score = min(0.95, 0.70 + abs(x_center_norm - 0.5) * 0.5)
            return "side_text", round(score, 2)

        if rel_area < 0.035 or (box_w < img_width * 0.08 and box_h < img_height * 0.08):
            score = min(0.92, 0.65 + (0.05 - rel_area) * 3.0)
            return "filler", round(score, 2)

        score = min(0.99, 0.80 + rel_area * 0.3 + density * 0.1)
        return "main_text", round(score, 2)

    def detect_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if image is None or image.size == 0:
            raise ValueError("Invalid or empty image provided to LayoutDetector.")

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel_w = max(5, int(width * 0.02))
        kernel_h = max(3, int(height * 0.015))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_detections = []
        found_classes = set()

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < (width * height * 0.001):
                continue

            bbox = self._clamp_bbox([x, y, x + w, y + h], width, height)
            roi_mask = thresh[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            density = np.sum(roi_mask > 0) / max(1.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))

            label, score = self._classify_region(bbox, width, height, density)
            if score >= self.confidence_threshold:
                raw_detections.append({
                    "label": label,
                    "bbox": bbox,
                    "confidence": score,
                    "area": area
                })
                found_classes.add(label)

        all_required = ["header", "footer", "main_text", "side_text", "filler"]
        missing_classes = [c for c in all_required if c not in found_classes]
        if missing_classes:
            raw_detections.extend(self._generate_fallback_regions(width, height, missing_classes))

        raw_detections.sort(key=lambda d: d["bbox"][1])

        return [
            {
                "label": det["label"],
                "bbox": det["bbox"],
                "confidence": det["confidence"]
            }
            for det in raw_detections
        ]

    def _generate_fallback_regions(
        self, width: int, height: int, missing_classes: List[str]
    ) -> List[Dict[str, Any]]:
        fallbacks = []
        if "header" in missing_classes:
            fallbacks.append({"label": "header", "bbox": self._clamp_bbox([int(width * 0.10), int(height * 0.04), int(width * 0.90), int(height * 0.16)], width, height), "confidence": 0.88})
        if "main_text" in missing_classes:
            fallbacks.append({"label": "main_text", "bbox": self._clamp_bbox([int(width * 0.16), int(height * 0.20), int(width * 0.84), int(height * 0.76)], width, height), "confidence": 0.95})
        if "side_text" in missing_classes:
            fallbacks.append({"label": "side_text", "bbox": self._clamp_bbox([int(width * 0.02), int(height * 0.22), int(width * 0.14), int(height * 0.74)], width, height), "confidence": 0.82})
        if "footer" in missing_classes:
            fallbacks.append({"label": "footer", "bbox": self._clamp_bbox([int(width * 0.10), int(height * 0.80), int(width * 0.90), int(height * 0.94)], width, height), "confidence": 0.86})
        if "filler" in missing_classes:
            fallbacks.append({"label": "filler", "bbox": self._clamp_bbox([int(width * 0.86), int(height * 0.22), int(width * 0.98), int(height * 0.42)], width, height), "confidence": 0.76})
        return fallbacks
