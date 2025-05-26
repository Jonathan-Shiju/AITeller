from Backend.app import routes

def register_all_routes(app):
    """
    Register all routes for the backend application.

    This function registers the main backend route, Twilio webhook,
    WebSocket handler, and custom error handlers.

    :param app: The Flask application instance.
    """
    app.register_blueprint(routes.backendMain)
    return app
