"""
Streamlit Web Dashboard for Manuscript Layout Region Detection.
"""

import os
import sys
import json
import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import load_config
from src.preprocessing.image_preprocessing import apply_clahe, denoise_image, deskew_image
from src.inference.detector import LayoutDetector
from src.utils.visualization import draw_bounding_boxes
from src.utils.io_utils import get_image_files

st.set_page_config(
    page_title="Manuscript Layout Region Detector",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_project_config():
    try:
        return load_config()
    except Exception:
        return {
            "target_classes": ["header", "footer", "main_text", "side_text", "filler"],
            "version": "0.1.0"
        }


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def main():
    st.title("📜 Manuscript Specific Layout Region Detection")
    st.markdown(
        "Analyze historical manuscript pages to detect and classify 5 distinct layout regions: "
        "**Header**, **Footer**, **Main Text**, **Side Text**, and **Filler**."
    )

    config = load_project_config()
    all_classes = config.get("target_classes", ["header", "footer", "main_text", "side_text", "filler"])

    st.sidebar.header("⚙️ Detection Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Cutoff",
        min_value=0.10,
        max_value=1.00,
        value=0.50,
        step=0.05
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Filter Classes")
    selected_classes = [
        cls_name for cls_name in all_classes
        if st.sidebar.checkbox(f"Show `{cls_name}`", value=True, key=f"cb_{cls_name}")
    ]

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Image Preprocessing")
    enable_enhancement = st.sidebar.checkbox("Enable CLAHE & Contrast", value=True)
    enable_deskew = st.sidebar.checkbox("Enable Page Deskewing", value=True)

    tab1, tab2, tab3 = st.tabs(["📷 Single Page Analysis", "📂 Batch Directory Processing", "ℹ️ About & Classes"])

    with tab1:
        st.subheader("Upload & Analyze Manuscript Image")
        uploaded_file = st.file_uploader(
            "Choose a manuscript page image...",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
        )

        sample_path = os.path.join(PROJECT_ROOT, "data", "raw", "Sample Test Data", "Screenshot 2026-10-02 003806.png")

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_name = uploaded_file.name
        elif os.path.exists(sample_path):
            original_img = cv2.imread(sample_path)
            image_name = "Sample Manuscript Page (Default)"
            st.info(f"Displaying sample manuscript image: `{image_name}`.")
        else:
            original_img = None
            st.warning("Please upload a manuscript image to begin detection.")

        if original_img is not None:
            processed_img = original_img.copy()
            if enable_deskew:
                processed_img = deskew_image(processed_img)
            if enable_enhancement:
                processed_img = apply_clahe(processed_img)
                processed_img = denoise_image(processed_img)

            detector = LayoutDetector(confidence_threshold=confidence_threshold)
            raw_predictions = detector.detect_regions(processed_img)
            filtered_predictions = [p for p in raw_predictions if p["label"] in selected_classes]

            annotated_bgr = draw_bounding_boxes(original_img, filtered_predictions)

            col1, col2 = st.columns(2)
            with col1:
                st.image(convert_bgr_to_rgb(original_img), caption="Raw Input Manuscript Image", use_column_width=True)
            with col2:
                st.image(convert_bgr_to_rgb(annotated_bgr), caption=f"Detected Regions ({len(filtered_predictions)} Found)", use_column_width=True)

            st.markdown("---")
            st.subheader("📊 Region Detection Results")
            if filtered_predictions:
                res_col1, res_col2 = st.columns([3, 2])
                with res_col1:
                    st.dataframe([
                        {
                            "Region #": idx,
                            "Class Label": pred["label"],
                            "Confidence": f"{pred['confidence']:.2f}",
                            "Bounding Box": str(pred["bbox"])
                        }
                        for idx, pred in enumerate(filtered_predictions, start=1)
                    ], use_container_width=True)

                with res_col2:
                    metadata_dict = {
                        "image_name": image_name,
                        "dimensions": {
                            "height": original_img.shape[0],
                            "width": original_img.shape[1],
                            "channels": original_img.shape[2]
                        },
                        "num_regions": len(filtered_predictions),
                        "regions": filtered_predictions
                    }
                    st.json(metadata_dict)
                    st.download_button(
                        label="📥 Download JSON Metadata",
                        data=json.dumps(metadata_dict, indent=4),
                        file_name=f"{os.path.splitext(image_name)[0]}_predictions.json",
                        mime="application/json"
                    )

    with tab2:
        st.subheader("Batch Folder Processing")
        input_dir_path = st.text_input("Relative directory path:", value="./data/raw/Sample Test Data")
        output_dir_path = st.text_input("Relative output directory:", value="./outputs")

        if st.button("🚀 Run Batch Processing", type="primary"):
            if not os.path.exists(input_dir_path):
                st.error(f"Directory not found: `{input_dir_path}`")
            else:
                try:
                    img_files = get_image_files(input_dir_path)
                    st.info(f"Processing {len(img_files)} image(s)...")

                    batch_detector = LayoutDetector(confidence_threshold=confidence_threshold)
                    progress_bar = st.progress(0)

                    annotated_out_dir = os.path.join(output_dir_path, "annotated")
                    predictions_out_dir = os.path.join(output_dir_path, "predictions")
                    os.makedirs(annotated_out_dir, exist_ok=True)
                    os.makedirs(predictions_out_dir, exist_ok=True)

                    for idx, img_p in enumerate(img_files):
                        raw_img = cv2.imread(img_p)
                        if raw_img is None:
                            continue

                        preds = batch_detector.detect_regions(raw_img)
                        ann_bgr = draw_bounding_boxes(raw_img, preds)
                        fname = os.path.basename(img_p)
                        base, _ = os.path.splitext(fname)

                        cv2.imwrite(os.path.join(annotated_out_dir, f"{base}_annotated.png"), ann_bgr)
                        with open(os.path.join(predictions_out_dir, f"{base}_prediction.json"), "w") as jf:
                            json.dump({"image": fname, "predictions": preds}, jf, indent=4)

                        progress_bar.progress((idx + 1) / len(img_files))

                    st.success(f"Batch processing completed! Saved outputs to `{output_dir_path}`.")
                except Exception as e:
                    st.error(f"Error during batch processing: {e}")

    with tab3:
        st.subheader("Layout Region Descriptions")
        st.markdown("""
        - **`header`**: Top margin text, running headers, section titles, folio numbers.
        - **`footer`**: Bottom margin text, catchwords, page numbers, signatures.
        - **`main_text`**: Main body text block.
        - **`side_text`**: Marginalia, commentary, handwritten notes.
        - **`filler`**: Decorative motifs, English text, stamps, pencil marks.
        """)


if __name__ == "__main__":
    main()
