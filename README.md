# HITL Fake News Detection Platform

This repository contains a hybrid text and image forensics pipeline for human-in-the-loop claim extraction and manipulation detection.

## Setup

1. Create and activate a virtual environment:
   - PowerShell:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

2. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root containing:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   GOOGLE_FACTCHECK_API_KEY=your_factcheck_api_key
   ```

## Run

- Run the text pipeline:
  ```powershell
  .\.venv\Scripts\python.exe app.py
  ```

- Run the image forensics test suite:
  ```powershell
  .\.venv\Scripts\python.exe test_phase3.py
  ```

## Notes

- `app.py` uses `google-genai` and `requests`.
- `image_forensics.py` uses `numpy`, `opencv-python`, `Pillow`, and `ExifRead`.
- Forensic output is saved to `forensic_output/`.
