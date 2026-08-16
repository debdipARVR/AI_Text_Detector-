"""Integration tests for FastAPI Web Application endpoints and static assets."""

import pytest
from fastapi.testclient import TestClient
from src.web.app import app

client = TestClient(app)


def test_status_endpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "nvidia_client" in data
    assert "security" in data
    assert len(data["available_models"]) >= 1


def test_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    assert any(m["id"] == "z-ai/glm-5.2" for m in models)
    assert any(m["id"] == "thinkingmachines/inkling" for m in models)


def test_encrypt_endpoint():
    response = client.post("/api/encrypt", json={"api_key": "nvapi-sample-key-12345"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "fernet_secret_key" in data
    assert "encrypted_token" in data
    assert data["masked_key"].startswith("nvapi")


def test_detect_endpoint():
    sample_text = (
        "Furthermore, large language models are transforming technological innovation. "
        "Moreover, they represent a pivotal advancement across multiple industries. "
        "In conclusion, adopting these architectures is crucial for continuous progression."
    )
    response = client.post(
        "/api/detect",
        json={
            "text": sample_text,
            "mask_rate": 0.30,
            "num_passes": 1,
            "model_name": "z-ai/glm-5.2",
            "temperature": 0.0,
            "raw_api_key": "",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "ai_probability" in data
    assert "verdict" in data
    assert "metrics" in data
    assert "spans" in data
    assert "highlighted_html" in data
    assert "deepeval_evaluation" in data


def test_humanize_endpoint():
    sample_text = "Furthermore, it is a testament to the crucial role of AI in our ecosystem."
    response = client.post(
        "/api/humanize",
        json={
            "text": sample_text,
            "domain": "academic",
            "raw_api_key": "",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "humanized_text" in data
    assert "ai_markers_before" in data
    assert "humanized_detection" in data


def test_prompt_endpoint():
    response = client.post(
        "/api/prompt",
        json={
            "domain": "academic",
            "target_audience": "University Professors",
            "additional_notes": "Emphasize empirical evidence",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "system_prompt" in data
    assert "user_prompt_template" in data
    assert "University Professors" in data["user_prompt_template"]


def test_static_and_index():
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "ClozeCongruence" in res_index.text

    res_css = client.get("/static/app.css")
    assert res_css.status_code == 200
    assert "ClozeCongruence" in res_css.text

    res_js = client.get("/static/app.js")
    assert res_js.status_code == 200
    assert "renderDetectionResults" in res_js.text
