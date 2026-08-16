#!/usr/bin/env python3
"""Run script to start the ClozeCongruence AI Text Detector Playground."""

import argparse
import sys
import uvicorn

def main():
    parser = argparse.ArgumentParser(description="Start ClozeCongruence Web Playground")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on file change")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f" 🕵️‍♂️ Starting ClozeCongruence AI Text Detector Playground")
    print(f" URL: http://{args.host}:{args.port}")
    print(f" Docs: http://{args.host}:{args.port}/docs")
    print(f"=======================================================\n")

    uvicorn.run("src.web.app:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
