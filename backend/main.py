import re
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from sqlmodel import Field, Session, SQLModel, create_engine, select
from fastapi.middleware.cors import CORSMiddleware

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


# Input form: the information needed to create a tow.
class TowCreate(SQLModel):
    plate: str = Field(min_length=1, max_length=20)
    detected_at: datetime
    truck_id: str = Field(min_length=1, max_length=50)
    destination: str = Field(min_length=1, max_length=200)
    status: str = Field(default="towed", min_length=1, max_length=30)
    confidence: float = Field(ge=0, le=1)
    image_url: str | None = None


# Keep the database beside this Python file.
database_path = Path(__file__).resolve().parent / "towtrace.db"

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
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