"""Script to verify all required dependencies and libraries are installed."""

def check_all_libraries():
    modules = [
        "cryptography", "fastapi", "uvicorn", "streamlit", "openai",
        "dotenv", "rapidfuzz", "scipy", "jinja2", "pydantic",
        "pytest", "httpx", "pandas", "altair", "starlette", "cffi"
    ]
    print("=" * 60)
    print(" INSTALLED LIBRARIES VERIFICATION")
    print("=" * 60)
    all_ok = True
    for m in modules:
        try:
            mod = __import__(m)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK] {m:<16} : {ver}")
        except Exception as e:
            print(f"  [FAIL] {m:<16} : Error ({e})")
            all_ok = False
    print("=" * 60)
    if all_ok:
        print("ALL LIBRARIES DOWNLOADED AND VERIFIED SUCCESSFULLY!")
    else:
        print("Some libraries failed to load.")
    print("=" * 60)

if __name__ == "__main__":
    check_all_libraries()
