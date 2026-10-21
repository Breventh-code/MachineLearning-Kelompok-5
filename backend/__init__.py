"""Flask application factory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_app(test_config=None):
    from flask import Flask
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "frontend" / "templates"),
        static_folder=str(PROJECT_ROOT / "frontend" / "static"),
        static_url_path="/static",
    )
    app.config.update(
        JSON_AS_ASCII=False,
        MAX_CONTENT_LENGTH=1024 * 1024,  # batas JSON: 1 MB
    )
    if test_config:
        app.config.update(test_config)

    from backend.routes import web
    app.register_blueprint(web)
    return app
