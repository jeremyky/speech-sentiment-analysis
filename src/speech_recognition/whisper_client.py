import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
from src.utils.config_manager import ConfigManager

class WhisperTranscriber:
    def __init__(self):
        self.config = ConfigManager()
        model_size = self.config.main_config['model']['whisper_model_size']
        self.model = whisper.load_model(model_size)
        
        audio_config = self.config.get_audio_config()
        self.sample_rate = audio_config['sample_rate']
        self.channels = audio_config['channels']
        self.default_duration = audio_config['recording_duration']

    def record_audio(self, duration=None):
        """Record audio from microphone"""
        duration = duration or self.default_duration
        print("Recording...")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels
        )
        sd.wait()
        return audio

    def transcribe_audio(self, audio):
        """Transcribe audio using Whisper"""
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
            write(temp_audio.name, self.sample_rate, audio)
            result = self.model.transcribe(temp_audio.name)
            return result["text"]

    def transcribe_realtime(self, duration=None):
        """Record and transcribe in one step"""
        audio = self.record_audio(duration)
        transcription = self.transcribe_audio(audio)
        return transcription, audio

    def transcribe_file(self, filepath):
        """Transcribe audio from file"""
        try:
            result = self.model.transcribe(filepath)
            return result["text"]
        except Exception as e:
            print(f"Error transcribing file: {e}")
            raise 