"""
CLI Inference Script for Manuscript Layout Region Detection.
"""

import os
import sys
import argparse
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.image_preprocessing import preprocess_image
from src.inference.detector import LayoutDetector
from src.utils.visualization import draw_bounding_boxes
from src.utils.io_utils import get_image_files, save_json_metadata
from src.utils.logger import setup_logger

logger = setup_logger("InferenceCLI")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manuscript Layout Region Detection CLI tool."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="./data/test_images",
        help="Relative path to input image file or directory containing images."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./outputs",
        help="Relative path to destination folder for prediction outputs."
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.5,
        help="Confidence threshold between 0.0 and 1.0."
    )
    return parser.parse_args()


def process_batch(input_path: str, output_dir: str, confidence_threshold: float):
    logger.info(f"Starting inference pipeline...")
    logger.info(f"Input path : {input_path}")
    logger.info(f"Output path: {output_dir}")
    logger.info(f"Confidence : {confidence_threshold}")

    try:
        image_files = get_image_files(input_path)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)

    if not image_files:
        logger.warning(f"No valid image files found in path: {input_path}")
        return

    logger.info(f"Found {len(image_files)} image(s) to process.")

    annotated_dir = os.path.join(output_dir, "annotated")
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    detector = LayoutDetector(confidence_threshold=confidence_threshold)

    for index, image_path in enumerate(image_files, start=1):
        filename = os.path.basename(image_path)
        base_name, _ = os.path.splitext(filename)

        logger.info(f"[{index}/{len(image_files)}] Processing: {filename}")

        original_image, processed_image = preprocess_image(image_path)
        predictions = detector.detect_regions(original_image)

        json_output_path = os.path.join(predictions_dir, f"{base_name}_prediction.json")
        metadata = {
            "image_filename": filename,
            "relative_path": image_path,
            "image_dimensions": {
                "height": original_image.shape[0],
                "width": original_image.shape[1],
                "channels": original_image.shape[2]
            },
            "num_regions_detected": len(predictions),
            "regions": predictions
        }
        save_json_metadata(metadata, json_output_path)

        annotated_image = draw_bounding_boxes(original_image, predictions)
        annotated_output_path = os.path.join(annotated_dir, f"{base_name}_annotated.png")
        cv2.imwrite(annotated_output_path, annotated_image)

        logger.info(f" -> Saved annotated image: {annotated_output_path}")
        logger.info(f" -> Saved JSON metadata:   {json_output_path}")

    logger.info("Batch inference completed successfully.")


def main():
    args = parse_arguments()
    process_batch(args.input, args.output, args.conf)


if __name__ == "__main__":
    main()
