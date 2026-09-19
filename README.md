# Have I Been Towed? (TowTrace)

A rear-facing camera on a tow truck captures the vehicle being towed. AI detects and reads the license plate, sends the plate and tow details to the backend, and the public website lets a driver search their plate to find where the car was taken.

## Project structure

```
have-i-been-towed/
├── ai-model/       # Python license plate detection
├── backend/        # FastAPI + SQLite API
├── frontend/       # HTML search page
└── README.md
```

## Running the backend

```
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1      # Mac/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Backend runs at `http://127.0.0.1:8000`. Interactive API docs at `http://127.0.0.1:8000/docs`.

## API routes

- `POST /tows` — create a new tow record
- `GET /tows` — list all tow records
- `GET /tows/{plate}` — search for a tow record by plate

### Tow record shape

```json
{
  "plate": "ABC123",
  "detected_at": "2026-09-19T08:15:00",
  "truck_id": "TRUCK-01",
  "destination": "Demo Tow Yard",
  "status": "towed",
  "confidence": 0.95
}
```

## Running the frontend

Open `frontend/index.html` directly in a browser while the backend is running. It searches a plate against `GET /tows/{plate}`.

**Note:** the frontend currently points to `http://127.0.0.1:8000`, so the backend must be running on the same machine you're viewing the page from.

## Team

| Owner | Role |
|---|---|
| Teammate 1 | AI / Computer Vision |
| Teammate 2 | Frontend / UX |
| Teammate 3 | Backend / Database |
| Teammate 4 (Ward) | Integration / Full Stack / Demo |
