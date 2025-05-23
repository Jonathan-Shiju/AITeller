from flask import Flask
from config.register_routes import register_all_routes
from config.app_logger import setup_logging

def create_app():
    setup_logging()
    app = Flask(__name__)
    register_all_routes(app)
    return app