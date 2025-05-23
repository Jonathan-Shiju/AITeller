from google.cloud import texttospeech as tts

def google_tts(text, filename):
    """
    Converts the given text to speech using Google Cloud TTS and saves it as an MP3 file.

    :param text: The text to convert to speech.
    :param filename: The output filename for the MP3 audio.
    """
    client = tts.TextToSpeechClient()
    synthesis_input = tts.SynthesisInput(text=text)

    voice = tts.VoiceSelectionParams(
        language_code="en-IN",
        ssml_gender=tts.SsmlVoiceGender.NEUTRAL,
        name="en-IN-Chirp3-HD-Laomedeia"  # Sampled the audio @https://cloud.google.com/text-to-speech?hl=en
    )

    audio_config = tts.AudioConfig(
        audio_encoding=tts.AudioEncoding.MP3  # MP3 Format to stream back to the user
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open(filename, "wb") as out:
        out.write(response.audio_content)
