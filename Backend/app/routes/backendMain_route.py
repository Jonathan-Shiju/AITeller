from quart import Blueprint, render_template, request, websocket
import logging
import os
from dotenv import load_dotenv
from Backend.services.twilio_voip import media_ws_helper

# Load environment variables from dev.env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../env/dev.env'))

logger = logging.getLogger(__name__)

backendMain = Blueprint('backendMain', __name__)

@backendMain.route('/', methods=['GET'])
async def backend_main():
    """
    Main route for the backend application.
    This route can be used to check if the backend is running.
    """
    return "Backend is running", 200

@backendMain.route('/twilio-webhook', methods=['POST'])
async def twilio_webhook():
    """
    Webhook endpoint for Twilio to handle incoming audio streams.
    This route processes audio data, transcribes it, generates a reply,
    and sends back the audio response.
    """
    logger.info("Received Twilio webhook request")
    ngrok_url = os.environ.get("NGROK_URL")
    return await render_template('twilio_response.xml', ngrok_url=ngrok_url), 200, {'Content-Type': 'application/xml'}

async def twilio_media_ws():
    """
    WebSocket endpoint for Twilio Media Streams.
    """
    await media_ws_helper(websocket)


