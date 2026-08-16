#!/usr/bin/env python3
"""Runner script to launch the Streamlit Playground."""

import os
import subprocess
import sys

def main():
    print("\n=======================================================")
    print(" 🕵️‍♂️ Starting ClozeCongruence Streamlit Playground")
    print(" Server: http://localhost:8501")
    print("=======================================================\n")

    app_path = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path, "--server.port=8501", "--server.headless=true"])

if __name__ == "__main__":
    main()
