from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import router
from app.triage_service import TriageService
from triageguard.predict import TriagePipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models ONCE at startup
    model_dir = Path("models")
    pipeline = TriagePipeline.load(model_dir=model_dir)
    app.state.triage_service = TriageService(pipeline)
    yield


app = FastAPI(
    title="TriageGuard",
    description="Clinical Decision Support System for Emergency Triage",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS enabled
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Mount figures directory
figures_dir = Path("results/figures")
if figures_dir.exists():
    app.mount("/figures", StaticFiles(directory=str(figures_dir)), name="figures")

# Mount web frontend if built
dist_dir = Path("web/dist")
if dist_dir.exists():
    app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="static")
