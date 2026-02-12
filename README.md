# Image Dehazing (Traditional)

This project provides a classic, training-free image dehazing pipeline based on the Dark Channel Prior,
plus a Tkinter GUI for image preview, metric comparison, and chart export.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

## GUI Workflow

1. Choose an image folder and load an image.
2. Click `Run Dehaze` to generate output.
3. Open the `Metrics` tab to view:
   - grouped bar chart (input vs dehazed),
   - radar chart (normalized profile),
   - numeric table for four metrics:
     - Tenengrad
     - Laplacian Variance
     - Entropy
     - RMS Contrast
4. Click `Export Metric Charts` to save paper-ready PNG figures.

## Notes

- The algorithm is traditional (no training/dataset needed).
- Large images may take longer; try smaller images for faster results.
