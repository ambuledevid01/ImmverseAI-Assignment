"""
Inference Unit Tests
--------------------
Basic unit tests validating layout region classes, relative path handling,
and detector bounding box predictions using Python's standard unittest framework.
"""
import os
import sys
import unittest
import numpy as np

# Ensure root directory is in python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.detector import LayoutDetector


class TestLayoutDetector(unittest.TestCase):
    """Test suite for LayoutDetector class."""

    def setUp(self):
        """Set up fresh detector instance before each test."""
        self.detector = LayoutDetector(confidence_threshold=0.5)

    def test_target_classes(self):
        """Verify that detector includes all 5 required manuscript layout classes."""
        expected_classes = {"header", "footer", "main_text", "side_text", "filler"}
        self.assertEqual(set(self.detector.classes), expected_classes)

    def test_detector_inference_dummy_image(self):
        """Verify detector returns structured region predictions on a dummy image array."""
        dummy_image = np.zeros((500, 500, 3), dtype=np.uint8)
        predictions = self.detector.detect_regions(dummy_image)

        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)

        for pred in predictions:
            self.assertIn("label", pred)
            self.assertIn("bbox", pred)
            self.assertIn("confidence", pred)
            self.assertIn(pred["label"], self.detector.classes)
            self.assertEqual(len(pred["bbox"]), 4)
            self.assertTrue(0.0 <= pred["confidence"] <= 1.0)

    def test_invalid_image_raises_error(self):
        """Verify detector raises ValueError when passed an invalid image."""
        with self.assertRaises(ValueError):
            self.detector.detect_regions(None)


if __name__ == "__main__":
    unittest.main()
