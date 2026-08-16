"""FastAPI Web Application and REST API for AI Text Detector Playground."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..engine import (
    ClozeCongruenceDetector,
    NvidiaNIMClient,
    NVIDIA_MODELS,
    TextHumanizer,
    HUMANIZER_MODES,
)
from ..security.encryption import (
    EncryptionError,
    decrypt_api_key,
    encrypt_api_key,
    generate_fernet_key,
    mask_api_key,
    resolve_api_credentials,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Cloze Congruence AI Text Detector",
    description="Statistical Cloze Infilling Congruence Detector powered by NVIDIA NIM & Fernet Security",
    version="1.0.0",
)

# Enable CORS for local playground flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request & Response Schemas
class DetectRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Input text to evaluate")
    mask_rate: float = Field(0.30, ge=0.10, le=0.50, description="Cloze masking percentage")
    num_passes: int = Field(2, ge=1, le=5, description="Monte Carlo passes count")
    model_name: Optional[str] = Field("meta/llama-3.3-70b-instruct", description="NVIDIA NIM model")
    temperature: float = Field(0.0, ge=0.0, le=1.0, description="Infill temperature")
    raw_api_key: Optional[str] = Field(None, description="Optional plaintext NVIDIA NIM API key")
    encrypted_token: Optional[str] = Field(None, description="Encrypted Fernet token")
    fernet_key: Optional[str] = Field(None, description="Fernet Secret key")


class HumanizeRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Text to humanize")
    domain: str = Field("academic", description="Target domain profile")
    model_name: Optional[str] = Field("meta/llama-3.3-70b-instruct", description="NVIDIA NIM model")
    temperature: float = Field(0.75, ge=0.1, le=1.2, description="Generation temperature")
    raw_api_key: Optional[str] = Field(None, description="Optional plaintext API key")
    encrypted_token: Optional[str] = Field(None, description="Encrypted Fernet token")
    fernet_key: Optional[str] = Field(None, description="Fernet Secret key")


class EncryptRequest(BaseModel):
    api_key: str = Field(..., min_length=1, description="Raw API key to encrypt")
    fernet_key: Optional[str] = Field(None, description="Optional custom Fernet secret key")


class PromptGenerateRequest(BaseModel):
    domain: str = Field("academic", description="Domain profile (academic, conversational, technical, etc.)")
    target_audience: str = Field("General Audience", description="Target readership")
    additional_notes: str = Field("", description="Custom stylistic instructions")


def create_detector_for_request(
    raw_api_key: Optional[str] = None,
    encrypted_token: Optional[str] = None,
    fernet_key: Optional[str] = None,
) -> ClozeCongruenceDetector:
    """Helper to instantiate detector with request-specific or system credentials."""
    nim_client = NvidiaNIMClient(
        api_key=raw_api_key,
        encrypted_token=encrypted_token,
        fernet_key=fernet_key,
    )
    return ClozeCongruenceDetector(nim_client=nim_client)


@app.get("/api/status")
async def get_system_status() -> Dict[str, Any]:
    """Retrieve system configuration, security status, and active backend mode."""
    nim_client = NvidiaNIMClient()
    status = nim_client.get_status()
    return {
        "status": "online",
        "nvidia_client": status,
        "security": {
            "encryption_algorithm": "Fernet (AES-128-CBC + HMAC-SHA256)",
            "credentials_configured": status["is_live"],
        },
        "available_modes": list(HUMANIZER_MODES.keys()),
        "available_models": NVIDIA_MODELS,
    }


@app.get("/api/models")
async def get_models() -> List[Dict[str, Any]]:
    """List supported NVIDIA NIM models."""
    return NVIDIA_MODELS


@app.post("/api/detect")
async def detect_ai_text(req: DetectRequest) -> Dict[str, Any]:
    """Run randomized Cloze Congruence AI detection on input text."""
    try:
        detector = create_detector_for_request(
            raw_api_key=req.raw_api_key,
            encrypted_token=req.encrypted_token,
            fernet_key=req.fernet_key,
        )
        results = detector.analyze(
            text=req.text,
            mask_rate=req.mask_rate,
            num_passes=req.num_passes,
            model_name=req.model_name,
            temperature=req.temperature,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/api/humanize")
async def humanize_text(req: HumanizeRequest) -> Dict[str, Any]:
    """Rewrite text with high burstiness and anti-detection styling."""
    try:
        nim_client = NvidiaNIMClient(
            api_key=req.raw_api_key,
            encrypted_token=req.encrypted_token,
            fernet_key=req.fernet_key,
        )
        humanizer = TextHumanizer(nim_client=nim_client)
        result = humanizer.humanize(
            text=req.text,
            domain=req.domain,
            model_name=req.model_name,
            temperature=req.temperature,
        )
        
        # Also run cloze detection on the newly humanized text for instant before/after comparison
        detector = ClozeCongruenceDetector(nim_client=nim_client)
        humanized_detection = detector.analyze(result["humanized_text"]) if result["humanized_text"] else {}
        result["humanized_detection"] = humanized_detection

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Humanization failed: {str(e)}")


@app.post("/api/prompt")
async def generate_prompt(req: PromptGenerateRequest) -> Dict[str, Any]:
    """Generate anti-detection humanizer prompt template."""
    humanizer = TextHumanizer()
    bundle = humanizer.generate_humanize_prompt(
        domain=req.domain,
        target_audience=req.target_audience,
        additional_notes=req.additional_notes,
    )
    return bundle


@app.post("/api/encrypt")
async def encrypt_credentials_endpoint(req: EncryptRequest) -> Dict[str, Any]:
    """Encrypt an API key using Fernet key encryption."""
    try:
        fernet_key = req.fernet_key or generate_fernet_key()
        encrypted_token = encrypt_api_key(req.api_key, fernet_key)
        return {
            "status": "success",
            "fernet_secret_key": fernet_key,
            "encrypted_token": encrypted_token,
            "masked_key": mask_api_key(req.api_key),
            "instructions": {
                "dotenv": f"FERNET_SECRET_KEY={fernet_key}\nFERNET_ENCRYPTED_NVIDIA_API_KEY={encrypted_token}",
                "github_actions": "Add FERNET_SECRET_KEY and FERNET_ENCRYPTED_NVIDIA_API_KEY to Repository Secrets",
            },
        }
    except EncryptionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encryption failed: {str(e)}")


# Serve static files and frontend
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "AI Text Detector API is active. Static frontend not yet compiled."}
