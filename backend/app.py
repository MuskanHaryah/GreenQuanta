"""App factory: create_app(config=None) mounts the /api blueprint, the live-run manager and, if FRONTEND_DIST holds
an index.html (frontend/dist, built by Guide 3), the frontend: its files, and index.html for every other non-/api path."""
import sys

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException, NotFound

from backend import OPTIMIZER, ROOT
from backend.api import bp
from backend.jobs import JobManager

DEFAULTS = {
    "RESULTS_ROOT": ROOT / "results" / "mids",
    "FRONTEND_DIST": ROOT / "frontend" / "dist",
    "PYTHON": sys.executable,             # start the backend with .venv-gq/Scripts/python.exe
    "RUN_TIMEOUT_S": 900,
}


def create_app(config=None):
    app = Flask(__name__, static_folder=None)
    app.config.update(DEFAULTS)
    app.config.update(config or {})
    app.json.sort_keys = False
    app.extensions["jobs"] = JobManager(app.config["PYTHON"], OPTIMIZER, app.config["RUN_TIMEOUT_S"],
                                        app.config["RESULTS_ROOT"])
    app.register_blueprint(bp, url_prefix="/api")

    @app.errorhandler(HTTPException)
    def http_error(e):
        if request.path.startswith("/api"):
            return jsonify(error={"code": e.name.lower().replace(" ", "_"), "message": e.description}), e.code
        return e

    @app.get("/", defaults={"path": ""})
    @app.get("/<path:path>")
    def frontend(path):
        dist = app.config["FRONTEND_DIST"]
        if path.startswith("api/") or not (dist / "index.html").is_file():
            raise NotFound()
        return send_from_directory(dist, path if path and (dist / path).is_file() else "index.html")

    return app
