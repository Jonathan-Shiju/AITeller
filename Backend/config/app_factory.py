from quart import Quart
from Backend.config.register_routes import register_all_routes
from Backend.config.app_logger import setup_logging
import logging
import os

def create_app():
    setup_logging()
    # Absolute path to the templates directory
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../app/templates"))
    print("Quart template folder:", template_dir)
    app = Quart(__name__, template_folder=template_dir)
    register_all_routes(app)
    return app
