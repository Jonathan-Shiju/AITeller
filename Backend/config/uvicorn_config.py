import uvicorn
from Backend.config.app_factory import create_app

def run_uvicorn():
    app = create_app()
    uvicorn.run(app, host='0.0.0.0', port=8000)

def return_app():
    app = create_app()
    return app

if __name__ == "__main__":
    run_uvicorn()
