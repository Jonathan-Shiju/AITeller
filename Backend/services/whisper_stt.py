import whisper

model = None    

def __init__():
    global model
    # Load the Whisper model
    model = whisper.load_model("base")

def transcribe_audio(audio_data):
    global model
    # Convert audio data to a format suitable for Whisper
    audio = whisper.load_audio(audio_data)
    audio = whisper.pad_or_trim(audio)

    # Make a prediction
    result = model.transcribe(audio)
    
    # Return the transcription
    return result["text"]
