import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.engine.nim_client import NvidiaNIMClient

client = NvidiaNIMClient()
status = client.get_status()
print("Client Mode:", status["mode"])
print("Is Live NIM:", status["is_live"])
print("Masked Key:", status["masked_key"])
print("Default Model:", status["default_model"])
