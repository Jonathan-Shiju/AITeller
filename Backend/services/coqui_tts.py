import os
import io
import tempfile
import logging
from TTS.api import TTS

logger = logging.getLogger(__name__)

# Configuration for Coqui TTS
DEFAULT_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"  # You can change this to your preferred model
DEFAULT_VOICE = None  # For models that support voice selection

def text_to_speech(text, model_name=DEFAULT_MODEL, voice=DEFAULT_VOICE):
    """
    Converts text to speech using Coqui TTS and returns the audio content as bytes.

    :param text: The text to convert to speech
    :param model_name: The Coqui TTS model to use
    :param voice: The voice to use (if the model supports it)
    :return: Audio content as bytes
    """
    try:
        # Initialize TTS
        tts = TTS(model_name=model_name)

        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filepath = temp_file.name

        # Generate speech
        if voice:
            tts.tts_to_file(text=text, file_path=temp_filepath, speaker=voice)
        else:
            tts.tts_to_file(text=text, file_path=temp_filepath)

        # Read the generated audio file
        with open(temp_filepath, "rb") as audio_file:
            audio_content = audio_file.read()

        # Clean up temporary file
        try:
            os.remove(temp_filepath)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file: {e}")

        return audio_content

    except Exception as e:
        logger.error(f"Coqui TTS error: {e}")
        raise

def save_to_file(text, filename, model_name=DEFAULT_MODEL, voice=DEFAULT_VOICE):
    """
    Converts the given text to speech using Coqui TTS and saves it as a WAV file.

    :param text: The text to convert to speech
    :param filename: The output filename for the audio file
    :param model_name: The Coqui TTS model to use
    :param voice: The voice to use (if the model supports it)
    """
    try:
        # Initialize TTS
        tts = TTS(model_name=model_name)

        # Generate speech
        if voice:
            tts.tts_to_file(text=text, file_path=filename, speaker=voice)
        else:
            tts.tts_to_file(text=text, file_path=filename)

    except Exception as e:
        logger.error(f"Coqui TTS error: {e}")
        raise

def list_available_models():
    """
    Lists all available Coqui TTS models.

    :return: List of available model names
    """
    try:
        tts = TTS()
        return tts.list_models()
    except Exception as e:
        logger.error(f"Failed to list TTS models: {e}")
        return []

def list_available_voices(model_name=DEFAULT_MODEL):
    """
    Lists all available voices for the given model.

    :param model_name: The model name to check voices for
    :return: List of available voices or None if the model doesn't support voice selection
    """
    try:
        tts = TTS(model_name=model_name)
        if hasattr(tts, "speakers"):
            return tts.speakers
        return None
    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        return None
