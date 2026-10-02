Manuscript Specific Layout Region Detection
  An end-to-end modular Python pipeline and computer vision framework built to automatically detect and classify layout regions in historical manuscript images (palm-leaf, paper documents, multi-column pages, and degraded scans).

📜 Project Overview & Target Regions
  Historical manuscripts contain distinct visual zones that separate body text from annotations, running titles, page numbers, and decorative motifs. This system detects and localizes five target layout region classes using bounding boxes, labels, and confidence scores:

  1.header : Top margin text, running headers, section titles, or folio numbers.
  2.footer : Bottom margin text, catchwords, page numbers, or signatures.
  3.main_text : Main body text block of the historical manuscript page.
  4.side_text : Marginalia, commentary, or handwritten annotations along page margins.
  5.filler : Decorative motifs, English text, stamps, pencil marks, and non-script visual elements.


📂 Project Directory Structure (I used the Releative Paths Here)

  manuscript-layout-detector/
  │
  ├── config/                     # Configuration management package
  │   ├── __init__.py             # Config package loader
  │   └── config.yaml             # Layout classes, colors, confidence thresholds, relative paths
  │
  ├── data/                       # Dataset directories
  │   ├── raw/                    # Raw untouched manuscript images
  │   │   └── reference/          # Reference manuscript pages (untouched)
  │   ├── processed/              # Cached preprocessed images
  │   ├── test_images/            # Sample images for testing CLI batch execution
  │   └── annotations/            # JSON/COCO ground-truth annotation files
  │
  ├── models/                     # Model weights and checkpoints
  │   ├── pretrained/             # Pre-trained base model weights
  │   └── trained/                # Fine-tuned model checkpoints
  │
  ├── src/                        # Modular source code package
  │   ├── __init__.py
  │   ├── preprocessing/          # Image transformations (resizing, CLAHE, binarization, deskew)
  │   │   ├── __init__.py
  │   │   └── image_preprocessing.py
  │   ├── dataset/                # Dataset loaders and annotation parsers
  │   │   ├── __init__.py
  │   │   └── dataset_loader.py
  │   ├── training/               # Model training loops and evaluation logic
  │   │   ├── __init__.py
  │   │   └── trainer.py
  │   ├── inference/              # Core region detection engine & classifier
  │   │   ├── __init__.py
  │   │   └── detector.py
  │   └── utils/                  # Helper utilities
  │       ├── __init__.py
  │       ├── visualization.py    # Drawing bounding box overlays & class labels
  │       ├── io_utils.py         # File searching & JSON metadata export
  │       └── logger.py           # Standardized console logging
  │
  ├── outputs/                    # Processed output directory
  │   ├── predictions/            # Generated JSON prediction metadata
  │   ├── annotated/              # Output images with rendered bounding boxes
  │   └── metrics/                # Evaluation plots and metric logs
  │
  ├── app/                        # Application interface package
  │   ├── __init__.py
  │   ├── main_app.py             # Streamlit interactive web dashboard
  │   └── server/                 # Node.js Express REST API server
  │       ├── server.js
  │       ├── package.json
  │       └── README.md
  │
  ├── notebooks/                  # Jupyter notebooks for EDA and experimentation
  │   └── README.md
  │
  ├── tests/                      # Automated unit tests
  │   ├── __init__.py
  │   └── test_inference.py       # Pytest / Unittest test suite
  │
  ├── inference.py                # Command-line interface (CLI) for batch processing
  ├── main.py                     # Primary entry point to display project status
  ├── requirements.txt            # Python dependencies
  ├── .gitignore                  # Git ignore definitions
  └── README.md                   # Complete documentation



🛠️ Installation & Requirements

  1. Prerequisites
  - Python 3.8+ installed on Windows, macOS, or Linux.
  - Node.js (v18+) for running the optional Express REST API server.

  2. Install Python Dependencies
    Terminal 
      pip install -r requirements.txt




🚀 Execution & Usage Guide

  1. CLI Batch Inference Script (`inference.py`)
    Process single manuscript images or an entire folder of images in batch mode.

  Terminal 
    python inference.py --input ./data/raw/Sample\ Test\ Data --output ./outputs


CLI Flags
  - `--input`, `-i`: Relative path to input image file or folder containing images (default: `./data/test_images`).
  - `--output`, `-o`: Relative destination folder for predictions (default: `./outputs`).
  - `--conf`, `-c`: Confidence threshold cutoff between `0.0` and `1.0` (default: `0.5`).

Output Deliverables:
  - Annotated Images (`./outputs/annotated/`) : Rendered bounding boxes, class labels, and confidence scores overlaid on manuscript pages.
  - JSON Prediction Metadata (`./outputs/predictions/`) : Structured JSON files containing image dimensions, total region count, and coordinates `[x_min, y_min, x_max, y_max]`.



Interactive Streamlit Web Dashboard (`app/main_app.py`)
  Launch the interactive web application to upload manuscript pages and visualize predictions:

Terminal 
  streamlit run app/main_app.py
  Or:
  python -m streamlit run app/main_app.py


- Access URL: `http://localhost:8501`  //DEFAULT Frontend PORT FOR LOCALHOST (I HAVENT DEPLOYED THE PROJECT)
- Features:
  - Drag-and-drop single image upload.
  - Interactive slider controls for confidence score thresholds.
  - Class filtering checkboxes (`header`, `footer`, `main_text`, `side_text`, `filler`).
  - Side-by-side view (Raw Manuscript vs. Region Overlays).
  - Live JSON metadata inspector & download button.


3. Node.js Express REST API Server (`app/server/`)
  Start the lightweight Express server:

Terminal 
  cd app/server
  npm install
  npm start


  - Access URL: `http://localhost:3000`  // Backed default port 
  - REST API Endpoints:
    - `GET /`: Health check.
    - `GET /api/status`: Configuration status.
    - `POST /api/detect`: Triggers Python CLI inference and returns JSON response.



4. Automated Unit Testing
  Execute the test suite to verify class specs and detection structures:

Terminal 
  python -m unittest discover tests




🔬 Image Preprocessing & Layout Detection Method

  - CLAHE (Contrast Limited Adaptive Histogram Equalization) : Enhances local contrast to reveal faded handwritten ink.
  - Denoising : Bilateral filtering preserves crisp text stroke edges while eliminating paper stains and ink bleed-through.
  - Page Deskewing : Automatically estimates rotation angle using minimum area bounding boxes and deskews the scan.
  - Morphological Aggregation : Rectangular dilation kernels merge text characters into block regions.
  - Spatial Projection Classification : Categorizes regions based on vertical positioning (`y_center`), horizontal margins (`x_center`), aspect ratio, and ink density into `header`, `footer`, `main_text`, `side_text`, and `filler`.
  - Safety Clamping : Ensures all bounding box coordinates remain strictly within page image dimensions.



Note : I made the half of the project using the AI tools .
       I Optimised the Pyhton code Code. 
       Used Node & Express for the Backend
       for frontend I used the Streamlit UI 


       Thank You ! 