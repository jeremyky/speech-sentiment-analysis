import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
from src.utils.config_manager import ConfigManager
import numpy as np
import librosa

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

    def transcribe_realtime(self, audio_data):
        """Transcribe audio in real-time with improved handling"""
        try:
            # Convert to float32 if not already
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)
            
            # Apply voice activity detection with stricter threshold
            intervals = librosa.effects.split(
                audio_data, 
                top_db=30,  # Increased from 20 to 30 for stricter VAD
                frame_length=2048,
                hop_length=512
            )
            
            # If no voice activity detected, return empty
            if len(intervals) == 0:
                return "", audio_data
            
            # Calculate average energy
            energy = np.mean(np.abs(audio_data))
            if energy < 0.01:  # Threshold for silence
                return "", audio_data
            
            # Keep only voice segments
            y_voice = np.concatenate([audio_data[start:end] for start, end in intervals])
            
            # Transcribe with Whisper using better parameters
            result = self.model.transcribe(
                y_voice,
                language='en',
                task='transcribe',
                fp16=False,
                best_of=1,
                beam_size=1,
                condition_on_previous_text=True,
                no_speech_threshold=0.6,  # Increased threshold for "no speech" detection
                initial_prompt=""  # Removed the prompt
            )
            
            # Only return text if confidence is high enough
            if result.get("no_speech_prob", 0) > 0.5:
                return "", audio_data
            
            text = result["text"].strip()
            # Remove the prompt if it somehow appears
            text = text.replace("The following is a transcription of speech:", "").strip()
            
            return text, audio_data
            
        except Exception as e:
            print(f"Error in real-time transcription: {e}")
            return "", None

    def transcribe_file(self, filepath):
        """Transcribe audio from file"""
        try:
            result = self.model.transcribe(filepath)
            return result["text"]
        except Exception as e:
            print(f"Error transcribing file: {e}")
            raise 

    def calculate_wer(self, reference, hypothesis):
        """Calculate Word Error Rate between reference and transcribed text"""
        try:
            # Tokenize into words
            ref_words = reference.lower().split()
            hyp_words = hypothesis.lower().split()
            
            # Calculate Levenshtein distance
            d = self._levenshtein_distance(ref_words, hyp_words)
            
            # Calculate WER
            wer = float(d) / float(len(ref_words))
            
            print("\nTranscription Metrics:")
            print("-" * 40)
            print(f"Word Error Rate: {wer:.2%}")
            print("-" * 40)
            
            return wer
            
        except Exception as e:
            print(f"Error calculating WER: {e}")
            return None
        
    def _levenshtein_distance(self, ref, hyp):
        """Helper function to calculate Levenshtein distance"""
        m = len(ref)
        n = len(hyp)
        dp = [[0] * (n+1) for _ in range(m+1)]
        
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
            
        for i in range(1, m+1):
            for j in range(1, n+1):
                if ref[i-1] == hyp[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                    
        return dp[m][n]