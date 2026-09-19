# Backend

FastAPI API + SQLite persistence + computer-vision pipeline.

- `main.py` — API routes, page serving, upload analysis, tow records, and detection history.
- `cv/tow_cv_only.py` — FastALPR/OpenCV pipeline for plate detection, OCR, tracking, tow-motion logic, voting, and snapshots.
- `requirements.txt` — Python dependencies.

Run the full project from the repository root with `python start_towtrace.py`.
