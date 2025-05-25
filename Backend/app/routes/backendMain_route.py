from flask import Flask, render_template, request
from flask_sockets import Sockets
import logging
import os
from services.twilio_voip import media_ws_helper

logging = logging.getLogger(__name__)
backendMain = Flask(__name__)
sockets = Sockets(backendMain)

@backendMain.route('', methods=['GET'])
def backend_main():
    """
    Main route for the backend application.
    This route can be used to check if the backend is running.
    """
    return "Backend is running", 200

@backendMain.route('/twilio-webhook', methods=['POST'])
def twilio_webhook():
    """
    Webhook endpoint for Twilio to handle incoming audio streams.
    This route processes audio data, transcribes it, generates a reply,
    and sends back the audio response.
    """
    logger.info("Received Twilio webhook request")
    ngrok_url = os.environ.get('NGROK_URL')
    return render_template('twilio_response.xml', ngrok_url=ngrok_url), 200, {'Content-Type': 'application/xml'}

@sockets.route('/ws')
def websocket_handler(ws):
   '''
   WebSocket handler for real-time communication.
   This route can be used to handle real-time data exchange
   between the client and server.
   '''
   logger.info("WebSocket connection established")
   media_ws_helper(ws)
   logger.info("WebSocket connection closed")
   return "WebSocket connection closed", 200

@backendMain.errorhandler(404)
def not_found_error(error):
    """
    Custom error handler for 404 Not Found errors.
    Returns a JSON response with an error message.
    """
    return {"error": "Not Found"}, 404


