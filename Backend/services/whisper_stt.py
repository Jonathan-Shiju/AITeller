import whisper
import os

model = None

def initialize_model():
    """Initialize the Whisper model"""
    global model
    if model is None:
        # Load the Whisper model
        model = whisper.load_model("base")

def transcribe_audio(audio_data):
    """Transcribe audio data to text"""
    global model

    # Initialize model if not already loaded
    if model is None:
        initialize_model()

    try:
        # Convert audio data to a format suitable for Whisper
        if isinstance(audio_data, str) and os.path.exists(audio_data):
            # If audio_data is a file path
            audio = whisper.load_audio(audio_data)
        else:
            # If audio_data is raw audio data
            audio = whisper.load_audio(audio_data)

        audio = whisper.pad_or_trim(audio)

        # Make a prediction
        result = model.transcribe(audio)

        # Return the transcription
        return result["text"]

    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""

# Initialize model on import
initialize_model()
