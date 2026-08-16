#!/usr/bin/env python3
"""Runner script to launch the Streamlit Playground."""

import os
import subprocess
import sys

def main():
    print("\n=======================================================")
    print(" Starting ClozeCongruence Streamlit Playground")
    print(" Server: http://localhost:8501")
    print("=======================================================\n")

    app_path = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
    
    # Check if .venv python exists in project directory
    venv_python = os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe")
    python_exec = venv_python if os.path.exists(venv_python) else sys.executable

    subprocess.run([python_exec, "-m", "streamlit", "run", app_path, "--server.port=8501"])

if __name__ == "__main__":
    main()
