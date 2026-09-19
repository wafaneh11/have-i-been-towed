import re
import shutil
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import Field, Session, SQLModel, create_engine, select

from backend.cv import tow_cv_only as cv_pipeline


# Clean a plate and reject invalid characters.
def normalize_plate(plate: str) -> str:
    cleaned = plate.strip().upper().replace(" ", "").replace("-", "")

    if not re.fullmatch(r"[A-Z0-9]{1,10}", cleaned):
        raise HTTPException(
            status_code=422,
            detail="Plate must contain 1–10 letters or numbers.",
        )

    return cleaned


# Database table: each row represents one tow.
class Tow(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    plate: str = Field(index=True)
    detected_at: datetime
    truck_id: str
    destination: str
    status: str
    confidence: float
    image_url: str | None = None


# Database table: each row is one plate the CV pipeline accepted from an
# uploaded test video, alongside its best snapshot image.
class Detection(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    plate: str = Field(index=True)
    confidence: float
    votes: int
    created_at: datetime
    snapshot_filename: str | None = None
    source_filename: str


# Input form: the information needed to create a tow.
class TowCreate(SQLModel):
    plate: str = Field(min_length=1, max_length=20)
    detected_at: datetime
    truck_id: str = Field(min_length=1, max_length=50)
    destination: str = Field(min_length=1, max_length=200)
    status: str = Field(default="towed", min_length=1, max_length=30)
    confidence: float = Field(ge=0, le=1)
    image_url: str | None = None


# GitHub-friendly project layout.
backend_dir = Path(__file__).resolve().parent
project_root = backend_dir.parent
frontend_dir = project_root / "frontend"
data_dir = project_root / "data"

database_path = data_dir / "towtrace.db"
frontend_path = frontend_dir / "index.html"
detection_page_path = frontend_dir / "detection.html"
evaluation_page_path = frontend_dir / "evaluation.html"

# Uploaded videos are temporary; accepted snapshots are retained for evaluation.
uploads_dir = data_dir / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)

snapshot_dir = data_dir / "snapshots"
snapshot_dir.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    f"sqlite:///{database_path}",
    connect_args={"check_same_thread": False},
)


# Create any missing database tables when the backend starts.
@asynccontextmanager
async def lifespan(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    yield


app = FastAPI(title="TowTrace AI", lifespan=lifespan)

# Serve saved detection snapshots as plain files, e.g. /snapshots/ABC123_....jpg
app.mount("/snapshots", StaticFiles(directory=snapshot_dir), name="snapshots")


@app.get("/", include_in_schema=False)
def home():
    """Serve the TowTrace frontend from the same FastAPI app."""
    if not frontend_path.exists():
        raise HTTPException(status_code=500, detail="Frontend index.html is missing")
    return FileResponse(frontend_path)


@app.get("/detection", include_in_schema=False)
def detection_page():
    """Serve the page for uploading and test-running a video through the CV pipeline."""
    if not detection_page_path.exists():
        raise HTTPException(status_code=500, detail="detection.html is missing")
    return FileResponse(detection_page_path)


@app.get("/evaluation", include_in_schema=False)
def evaluation_page():
    """Serve the page that shows past plate-detection snapshots."""
    if not evaluation_page_path.exists():
        raise HTTPException(status_code=500, detail="evaluation.html is missing")
    return FileResponse(evaluation_page_path)


@app.get("/health")
def health():
    return {"message": "TowTrace backend is running!"}


# Clean the plate and check for a duplicate before saving.
@app.post("/tows", response_model=Tow, status_code=201)
def create_tow(tow_data: TowCreate):
    tow_data.plate = normalize_plate(tow_data.plate)

    with Session(engine) as session:
        statement = select(Tow).where(
            Tow.plate == tow_data.plate,
            Tow.truck_id == tow_data.truck_id,
            Tow.detected_at == tow_data.detected_at,
        )

        existing_tow = session.exec(statement).first()

        if existing_tow is not None:
            raise HTTPException(
                status_code=409,
                detail="This tow detection is already saved",
            )

        tow = Tow.model_validate(tow_data)
        session.add(tow)
        session.commit()
        session.refresh(tow)
        return tow


# Return the 50 most recent tow records.
@app.get("/tows", response_model=list[Tow])
def list_tows():
    with Session(engine) as session:
        statement = (
            select(Tow)
            .order_by(Tow.detected_at.desc(), Tow.id.desc())
            .limit(50)
        )

        tows = session.exec(statement).all()
        return tows


# Clean the search plate and find its most recent tow.
@app.get("/tows/{plate}", response_model=Tow)
def search_tow(plate: str):
    plate = normalize_plate(plate)

    with Session(engine) as session:
        statement = (
            select(Tow)
            .where(Tow.plate == plate)
            .order_by(Tow.detected_at.desc(), Tow.id.desc())
            .limit(1)
        )

        tow = session.exec(statement).first()

        if tow is None:
            raise HTTPException(
                status_code=404,
                detail="No tow record found for this plate",
            )

        return tow


ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Same defaults the CV pipeline uses when run from the command line
# (see tow_cv_only.py's argparse defaults / test_cv.bat).
CV_EVERY = 5
CV_MIN_CONF = 0.80
CV_MIN_VOTES = 2
CV_MIN_LEN = 5
CV_MAX_LEN = 8
CV_MIN_TOWED_SECONDS = 3.0
CV_MIN_RIGID_FRAC = 0.15
CV_MOVE_THR = 0.5
CV_RIGID_RATIO = 0.3
CV_MAX_TOWED = 1


# Run an uploaded video through the same "towed" detection pipeline as
# tow_cv_only.py, saving a snapshot and a Detection row for every plate
# it accepts.
@app.post("/detections/analyze")
async def analyze_video(
    video: UploadFile = File(...),
    truck_id: str = Form("TRUCK-01"),
    destination: str = Form("Demo Tow Yard"),
):
    suffix = Path(video.filename or "").suffix.lower()

    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=422,
            detail="Please upload a video file (mp4, mov, avi, mkv, or webm).",
        )

    with tempfile.NamedTemporaryFile(dir=uploads_dir, suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(video.file, tmp)
        temp_video_path = Path(tmp.name)

    try:
        probe = cv2.VideoCapture(str(temp_video_path))
        opened = probe.isOpened()
        probe.release()

        if not opened:
            raise HTTPException(
                status_code=422,
                detail="Could not open the uploaded file as a video.",
            )

        reads, total_frames, tracks, fps = cv_pipeline.read_video(
            temp_video_path,
            every=CV_EVERY,
            min_conf=CV_MIN_CONF,
            min_len=CV_MIN_LEN,
            max_len=CV_MAX_LEN,
            debug=False,
            roi=None,
            target="towed",
            min_width_frac=0.0,
            move_thr=CV_MOVE_THR,
            rigid_ratio=CV_RIGID_RATIO,
        )

        chosen, rivals, _rows = cv_pipeline.select_towed(
            tracks,
            fps,
            CV_EVERY,
            CV_MIN_TOWED_SECONDS,
            CV_MIN_RIGID_FRAC,
            CV_MIN_VOTES,
            CV_MAX_TOWED,
        )

        ambiguous = bool(rivals)
        if ambiguous:
            chosen = []

        merged = cv_pipeline.merge_reads(chosen)
        plates = cv_pipeline.pick_plates(merged, CV_MIN_VOTES)

        accepted = []

        with Session(engine) as session:
            for text in plates:
                data = merged[text]
                votes = len(data["confs"])
                confidence = sum(data["confs"]) / votes
                plate = normalize_plate(text)
                detected_at = datetime.now()

                snapshot_filename = None
                if data["best_frame"] is not None:
                    snapshot_filename = f"{text}_{int(detected_at.timestamp())}.jpg"
                    cv2.imwrite(str(snapshot_dir / snapshot_filename), data["best_frame"])

                image_url = f"/snapshots/{snapshot_filename}" if snapshot_filename else None

                # Record the raw detection for the evaluation gallery ...
                detection = Detection(
                    plate=plate,
                    confidence=confidence,
                    votes=votes,
                    created_at=detected_at,
                    snapshot_filename=snapshot_filename,
                    source_filename=video.filename or "uploaded video",
                )
                session.add(detection)

                # ... and, exactly like the standalone CV pipeline does for a
                # real truck-camera detection, save it as a Tow record so the
                # plate shows up as towed on the homepage lookup.
                tow = Tow(
                    plate=plate,
                    detected_at=detected_at,
                    truck_id=truck_id.strip() or "TRUCK-01",
                    destination=destination.strip() or "Demo Tow Yard",
                    status="towed",
                    confidence=confidence,
                    image_url=image_url,
                )
                session.add(tow)

                session.commit()
                session.refresh(detection)

                accepted.append(
                    {
                        "plate": plate,
                        "confidence": round(confidence, 4),
                        "votes": votes,
                        "snapshot_url": image_url,
                    }
                )

        return {
            "analyzed_frames": total_frames // CV_EVERY,
            "ambiguous": ambiguous,
            "accepted": accepted,
        }
    finally:
        temp_video_path.unlink(missing_ok=True)


# Return the 50 most recent plate-detection snapshots, newest first.
@app.get("/detections", response_model=list[Detection])
def list_detections():
    with Session(engine) as session:
        statement = (
            select(Detection)
            .order_by(Detection.created_at.desc(), Detection.id.desc())
            .limit(50)
        )

        return session.exec(statement).all()


# Delete every saved snapshot and its Detection row. Does not touch the
# Tow table, so plates already found towed stay searchable.
@app.delete("/detections")
def clear_detections():
    with Session(engine) as session:
        detections = session.exec(select(Detection)).all()
        count = len(detections)

        for detection in detections:
            if detection.snapshot_filename:
                (snapshot_dir / detection.snapshot_filename).unlink(missing_ok=True)
            session.delete(detection)

        session.commit()

    return {"cleared": count}