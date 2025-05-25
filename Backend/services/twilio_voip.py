import tempfile
import base64
import logging
import json
import numpy as np
import io
import soundfile as sf
import os
from datetime import datetime
from scipy.io import wavfile
from .whisper_stt import transcribe_audio
from .llm import generate_reply
from .coqui_tts import text_to_speech  # Changed from google_tts to coqui_tts

logger = logging.getLogger(__name__)

def process_buffer(buffer):
    """
    Process the audio buffer to transcribe and generate a reply.

    :param buffer: Audio buffer as bytearray
    :return: (tts_audio_bytes, transcription_text, response_text)
    """
    # Convert buffer to numpy array (assuming 8-bit mu-law, mono, 8kHz from Twilio)
    audio_bytes = bytes(buffer)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Save to temp wav file for whisper (Whisper works better with 16kHz)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        # Using 8kHz sample rate to match Twilio's format
        sf.write(tmp_wav, audio_np, 8000, subtype='PCM_16')
        tmp_wav_path = tmp_wav.name

    result = transcribe_audio(tmp_wav_path)
    transcription = result["text"]
    logger.info(f"Transcription: {transcription}")

    # Generate a reply using LLM
    reply = generate_reply(transcription)
    logger.info(f"Generated reply: {reply}")

    # Convert reply to speech using Coqui TTS (self-hosted)
    tts_audio = text_to_speech(reply)
    logger.info("Generated TTS audio using Coqui TTS")

    return tts_audio, transcription, reply

def convert_to_mulaw(audio_data, target_sample_rate=8000):
    """
    Convert audio to 8kHz mu-law format required by Twilio Media Streams.

    :param audio_data: Input audio bytes
    :param target_sample_rate: Target sample rate (8000 Hz for Twilio)
    :return: Audio bytes in mu-law format
    """
    try:
        # Load audio into memory buffer
        with io.BytesIO(audio_data) as audio_buffer:
            # Read audio with original sample rate
            sample_rate, audio = wavfile.read(audio_buffer)

            # Resample to 8kHz if needed
            if sample_rate != target_sample_rate:
                # Simple resampling - for production use a better resampling algorithm
                import librosa
                audio = librosa.resample(
                    audio.astype(np.float32),
                    orig_sr=sample_rate,
                    target_sr=target_sample_rate
                )

            # Convert to mu-law encoding
            import librosa.util
            mulaw_audio = librosa.util.mu_compress(audio, mu=255, quantize=True)

            # Convert to bytes
            mulaw_bytes = np.array(mulaw_audio, dtype=np.uint8).tobytes()

            return mulaw_bytes
    except Exception as e:
        logger.error(f"Error converting audio to mu-law: {e}")
        # Return original audio as fallback
        return audio_data

def send_audio_response(ws, audio_data, stream_sid):
    """
    Send audio response back to the client over WebSocket.

    :param ws: WebSocket connection
    :param audio_data: Audio data bytes to send
    :param stream_sid: Stream session ID
    """
    # Convert to 8kHz mu-law format required by Twilio
    mulaw_audio = convert_to_mulaw(audio_data)

    response_data = {
        "event": "media",
        "media": {
            "payload": base64.b64encode(mulaw_audio).decode('utf-8'),
            "streamSid": stream_sid
        }
    }

    ws.send(json.dumps(response_data))
    logger.info("Sent audio response to client")

    # Send a mark event after sending audio to track when it's completed playing
    mark_label = f"audio_complete_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    send_mark_event(ws, stream_sid, mark_label)
    return mark_label

def send_mark_event(ws, stream_sid, mark_name):
    """
    Send a mark event to Twilio.

    :param ws: WebSocket connection
    :param stream_sid: Stream session ID
    :param mark_name: Name of the mark for tracking
    """
    mark_event = {
        "event": "mark",
        "streamSid": stream_sid,
        "mark": {
            "name": mark_name
        }
    }
    ws.send(json.dumps(mark_event))
    logger.info(f"Sent mark event with name: {mark_name}")

def save_call_logs(stream_sid, call_sid, conversation_history):
    """
    Save call logs to a separate file.

    :param stream_sid: Stream session ID
    :param call_sid: Call SID
    :param conversation_history: List of (transcription, response) tuples from the call
    """
    try:
        log_dir = os.path.join(os.path.dirname(__file__), "../logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{log_dir}/call_{call_sid or stream_sid}_{timestamp}.log"

        with open(filename, "w") as f:
            f.write(f"Call Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Stream SID: {stream_sid}\n")
            f.write(f"Call SID: {call_sid or 'Not provided'}\n\n")
            f.write("Conversation:\n")
            for i, (trans, resp) in enumerate(conversation_history, 1):
                f.write(f"{i}. User: {trans}\n")
                f.write(f"   AITeller: {resp}\n\n")

        logger.info(f"Call logs saved to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Failed to save call logs: {e}")
        return None

def media_ws_helper(ws):
    has_seen_media = False
    buffer = bytearray()
    stream_sid = None
    call_sid = None
    conversation_history = []  # Store (transcription, response) pairs
    pending_marks = set()  # Track mark events we're waiting for
    awaiting_audio_confirmation = False  # Flag to track if we're waiting for audio confirmation
    queued_audio = None  # Store an audio response that's waiting to be sent
    queued_transcription = None  # Store the transcription for the queued audio
    queued_response = None  # Store the response text for the queued audio

    while not ws.closed:
        try:
            message = ws.receive()
            if message is None:
                break

            data = json.loads(message)
            if data["event"] == "connected":
                logger.info(f"Connected: {message}")

            elif data["event"] == "start":
                stream_sid = data['start']['streamSid']
                call_sid = data['start'].get('callSid')
                logger.info("Stream started:")
                logger.info(f"  Stream SID: {stream_sid}")
                logger.info(f"  Account SID: {data['start'].get('accountSid', 'Not provided')}")
                logger.info(f"  Call SID: {call_sid or 'Not provided'}")

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

                    # Process the buffer and get audio response
                    tts_audio, transcription, response = process_buffer(buffer)

                    # Track conversation history
                    if transcription:
                        conversation_history.append((transcription, response))

                    # Only send audio if we're not waiting for a previous confirmation
                    if not awaiting_audio_confirmation:
                        # Send the TTS audio back to the client and track the mark
                        mark_label = send_audio_response(ws, tts_audio, stream_sid)
                        pending_marks.add(mark_label)
                        awaiting_audio_confirmation = True
                    else:
                        # Queue the audio for sending later
                        logger.info("Queuing audio response while waiting for previous confirmation")
                        queued_audio = tts_audio
                        queued_transcription = transcription
                        queued_response = response

                    # Clear the buffer after processing
                    buffer.clear()
                else:
                    logger.info("Buffer size is below threshold, waiting for more data...")

            elif data["event"] == "mark":
                # Handle mark events received from Twilio
                mark_name = data["mark"]["name"]
                logger.info(f"Received mark event: {mark_name}")

                if mark_name in pending_marks:
                    logger.info(f"Audio playback confirmed for mark: {mark_name}")
                    pending_marks.remove(mark_name)

                    # Now that audio is confirmed, we can send any queued audio
                    awaiting_audio_confirmation = False

                    if queued_audio:
                        logger.info("Sending previously queued audio response")
                        mark_label = send_audio_response(ws, queued_audio, stream_sid)
                        pending_marks.add(mark_label)
                        awaiting_audio_confirmation = True

                        # Clear the queue
                        queued_audio = None
                        queued_transcription = None
                        queued_response = None
                else:
                    logger.warning(f"Received unexpected mark: {mark_name}")

            elif data["event"] == "stop":
                logger.info("Stop event received, processing final buffer")
                if has_seen_media:
                    # Process any remaining data in the buffer
                    processed_audio, transcription, response = process_buffer(buffer)

                    # Add final transcription to history
                    if transcription:
                        conversation_history.append((transcription, response))

                    # Send the final TTS audio back to the client
                    # For the final audio, we'll send it regardless of pending confirmations
                    mark_label = send_audio_response(ws, processed_audio, stream_sid)

                    logger.info(f"Final transcription: {transcription}")
                    buffer.clear()

                    # Save call logs to file
                    log_file = save_call_logs(stream_sid, call_sid, conversation_history)
                    logger.info(f"Call logs saved to: {log_file}")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Save any logs we have so far if there's an error
            if stream_sid and conversation_history:
                save_call_logs(stream_sid, call_sid, conversation_history)
            break

    # Final cleanup
    if stream_sid and conversation_history:
        save_call_logs(stream_sid, call_sid, conversation_history)
