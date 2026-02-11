# Image Dehazing (Traditional)

This project provides a classic, training-free image dehazing pipeline based on the Dark Channel Prior,
plus a simple Tkinter UI to browse a folder, preview the input/output, and save results.

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

## Notes

- The algorithm is traditional (no training/dataset needed).
- Large images may take longer; try smaller images for faster results.
