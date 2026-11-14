"""App factory: create_app(config=None) mounts the /api blueprint and the live-run manager."""
import sys

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from backend import OPTIMIZER, ROOT
from backend.api import bp
from backend.jobs import JobManager

DEFAULTS = {
    "RESULTS_ROOT": ROOT / "results" / "mids",
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

    return app
