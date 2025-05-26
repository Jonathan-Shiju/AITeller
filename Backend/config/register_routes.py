from Backend.app.routes.backendMain_route import backendMain, twilio_media_ws
from flask_sockets import Sockets

def register_all_routes(app):
    """
    Register all routes for the backend application.

    This function registers the main backend route, Twilio webhook,
    WebSocket handler, and custom error handlers.

    :param app: The Flask application instance.
    """
    app.register_blueprint(backendMain)

    # Setup WebSocket handling
    sockets = Sockets(app)
    sockets.route('/twilio-media-ws')(twilio_media_ws)

    return app
