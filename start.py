"""Launch PaperPilot."""
import uvicorn

if __name__ == "__main__":
    port = 8000
    print(f"PaperPilot running at: http://localhost:{port}")
    uvicorn.run("paperpilot.app:app", host="0.0.0.0", port=port, reload=True)
