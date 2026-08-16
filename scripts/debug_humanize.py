"""Debug /api/humanize."""

import os
import sys
import traceback
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.web.app import app

client = TestClient(app)
try:
    res = client.post(
        "/api/humanize",
        json={
            "text": "Furthermore, it is a testament to the crucial role of AI in our ecosystem.",
            "domain": "academic",
            "raw_api_key": "",
        },
    )
    print("Status:", res.status_code)
    print("Response:", res.json())
except Exception as e:
    traceback.print_exc()
