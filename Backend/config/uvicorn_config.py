import uvicorn
from Backend.config.app_factory import create_app
from asgiref.wsgi import WsgiToAsgi

def run_uvicorn():
    app = create_app()
    asgi_app = WsgiToAsgi(app)
    uvicorn.run(asgi_app, host='0.0.0.0', port=8000)

if __name__ == "__main__":
    run_uvicorn()
