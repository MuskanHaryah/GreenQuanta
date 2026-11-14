"""python -m backend: serve the demo API (and frontend/dist if built) on http://127.0.0.1:5000."""
from backend.app import create_app

if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, use_reloader=False, threaded=True)
