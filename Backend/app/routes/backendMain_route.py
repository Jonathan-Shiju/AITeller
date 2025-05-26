from flask import Flask, render_template
import logging

logging = logging.getLogger(__name__)

backendMain = Flask(__name__)

@backendMain.route('/', methods=['GET'])
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
    return render_template('twilio_response.xml'), 200, {'Content-Type': 'application/xml'}


