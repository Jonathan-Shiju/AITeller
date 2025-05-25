from app import routes

def register_routes(app):
    """
    Register all routes for the backend application.

    This function registers the main backend route, Twilio webhook,
    WebSocket handler, and custom error handlers.

    :param app: The Flask application instance.
    """
   app.include_router(routes.backendMain)
   return app
