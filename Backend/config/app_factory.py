from flask import Flask
from flask_sockets import Sockets
from Backend.config.register_routes import register_all_routes
from Backend.config.app_logger import setup_logging
from Backend.app.routes.backendMain_route import twilio_media_ws
import logging
import os

def create_app():
    setup_logging()
    # Absolute path to the templates directory
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../app/templates"))
    print("Flask template folder:", template_dir)
    app = Flask(__name__, template_folder=template_dir)
    register_all_routes(app)
    return app
