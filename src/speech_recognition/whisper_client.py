import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile

class WhisperTranscriber:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)
        self.sample_rate = 16000

    def record_audio(self, duration=5):
        """Record audio from microphone"""
        print("Recording...")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1
        )
        sd.wait()
        return audio

    def transcribe_audio(self, audio):
        """Transcribe audio using Whisper"""
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
            write(temp_audio.name, self.sample_rate, audio)
            
            # Transcribe using Whisper
            result = self.model.transcribe(temp_audio.name)
            return result["text"]

    def transcribe_realtime(self, duration=5):
        """Record and transcribe in one step"""
        audio = self.record_audio(duration)
        return self.transcribe_audio(audio) 