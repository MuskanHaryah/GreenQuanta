"""App factory: create_app(config=None) mounts the /api blueprint."""
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from backend import ROOT
from backend.api import bp

DEFAULTS = {
    "RESULTS_ROOT": ROOT / "results" / "mids",
}


def create_app(config=None):
    app = Flask(__name__, static_folder=None)
    app.config.update(DEFAULTS)
    app.config.update(config or {})
    app.json.sort_keys = False
    app.register_blueprint(bp, url_prefix="/api")

    @app.errorhandler(HTTPException)
    def http_error(e):
        if request.path.startswith("/api"):
            return jsonify(error={"code": e.name.lower().replace(" ", "_"), "message": e.description}), e.code
        return e

    return app
