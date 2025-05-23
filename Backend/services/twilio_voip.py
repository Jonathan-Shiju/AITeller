import tempfile
import base64
import logging
import json
import numpy as np
import io
import soundfile as sf
from .whisper_stt import transcribe_audio
from .llm import generate_reply
from .google_tts import text_to_speech

logger = logging.getLogger(__name__)


def media_ws_helper(ws):
    has_seen_media = False
    buffer = bytearray()
    stream_sid = None

    while not ws.close:
        try:
            message = ws.receive()
            if message is None:
                break

            data = json.loads(message)
            if data["event"] == "connected":
                logger.info("Connected: ", message)
            
            elif data["event"] == "start":
                logger.info("Start: ", message)
                stream_sid = data['start']['streamSid']
            
            elif data["event"] == "media":
                if not has_seen_media:
                    logger.info("Media message received")
                    has_seen_media = True
                
                payload = data['media']['payload']
                chunk = base64.b64decode(payload)
                buffer.extend(chunk)
                
                # Check if the buffer size exceeds 16KB - Change if sufficient context not included 
                if len(buffer) > 16000:
                    logger.info("Buffer exceeded 16KB, processing...")
                    # Here you would transcribe and generate a reply, then TTS and stream back
                    # Transcribe buffer using whisper
                    # Convert buffer to numpy array (assuming 16-bit PCM, mono, 16kHz)
                    audio_bytes = bytes(buffer)
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                    # Save to temp wav file for whisper
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                        sf.write(tmp_wav, audio_np, 16000, subtype='PCM_16')
                        tmp_wav_path = tmp_wav.name

                    result = transcribe_audio(tmp_wav_path)
                    transcription = result["text"]
                    logger.info(f"Transcription: {transcription}")

                    

                    # TTS (using pyttsx3 or any TTS library, here just as a placeholder)
                    # Save TTS to wav and stream back as base64-encoded media events
                    # For production, use a streaming TTS API or library

                    # For brevity, just clear the buffer
                    buffer.clear()
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            break
        finally:
            if ws.close:
                break
