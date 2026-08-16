"""Verification script to test the live server endpoints."""

import urllib.request
import json

def test_endpoints():
    print("Testing GET /api/status...")
    req = urllib.request.Request("http://127.0.0.1:8000/api/status")
    with urllib.request.urlopen(req) as response:
        status_data = json.loads(response.read().decode())
        print(f"Status: {status_data['status']}, Mode: {status_data['nvidia_client']['mode']}")

    print("\nTesting POST /api/detect...")
    detect_payload = json.dumps({
        "text": "Furthermore, large language models are transforming technological paradigms. Moreover, they play a crucial role in modern computational ecosystems.",
        "mask_rate": 0.30,
        "num_passes": 1,
        "model_name": "meta/llama-3.3-70b-instruct",
        "temperature": 0.0
    }).encode("utf-8")
    
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/detect",
        data=detect_payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        detect_data = json.loads(response.read().decode())
        print(f"Verdict: {detect_data['verdict']}, AI Probability: {detect_data['ai_probability']}%")
        print(f"Congruence: {detect_data['metrics']['congruence_avg']}%, Spans: {len(detect_data['spans'])}")

    print("\nTesting POST /api/humanize...")
    hum_payload = json.dumps({
        "text": "Furthermore, it is a testament to the crucial role of AI in navigating this landscape.",
        "domain": "academic"
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/humanize",
        data=hum_payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        hum_data = json.loads(response.read().decode())
        print(f"Humanized Output: {hum_data['humanized_text']}")

    print("\nAll live endpoints verified successfully!")

if __name__ == "__main__":
    test_endpoints()
