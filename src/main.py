import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
import time
from speech_recognition.whisper_client import WhisperTranscriber
from sentiment_analysis.bert_model import SentimentAnalyzer
import sounddevice as sd

class AudioMeter(tk.Canvas):
    def __init__(self, parent, width=400, height=60, **kwargs):
        super().__init__(parent, width=width, height=height, **kwargs)
        self.width = width
        self.height = height
        self.configure(bg='black')
        self.bars = 30
        self.bar_width = (width - (self.bars + 1)) / self.bars
        self.levels = [0] * self.bars
        
    def update_levels(self, audio_data):
        if audio_data is None:
            self.levels = [0] * self.bars
        else:
            # Convert audio data to levels
            chunk_size = len(audio_data) // self.bars
            new_levels = []
            for i in range(self.bars):
                start = i * chunk_size
                end = start + chunk_size
                chunk = audio_data[start:end]
                level = float(np.abs(chunk).mean())
                new_levels.append(min(1.0, level * 3))  # Amplify for visibility
            self.levels = new_levels
        self.draw_bars()
        
    def draw_bars(self):
        self.delete("all")
        for i, level in enumerate(self.levels):
            x = i * (self.bar_width + 1) + 1
            height = int(level * self.height)
            # Color gradient from green to yellow to red
            if level < 0.5:
                color = '#2ecc71'  # green
            elif level < 0.8:
                color = '#f1c40f'  # yellow
            else:
                color = '#e74c3c'  # red
            
            self.create_rectangle(
                x, self.height - height,
                x + self.bar_width, self.height,
                fill=color, outline=''
            )

class SentimentAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech-to-Sentiment Analyzer")
        self.root.geometry("600x500")
        
        # Initialize components
        self.transcriber = WhisperTranscriber()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.is_recording = False
        self.audio_data = None
        
        self.setup_gui()
        
    def setup_gui(self):
        # Status label
        self.status_label = ttk.Label(self.root, text="Ready to start", font=('Arial', 12))
        self.status_label.pack(pady=10)
        
        # Audio meter
        self.audio_meter = AudioMeter(self.root)
        self.audio_meter.pack(pady=10, padx=20)
        
        # Transcription display
        ttk.Label(self.root, text="Transcription:", font=('Arial', 10, 'bold')).pack(pady=5)
        self.transcription_text = tk.Text(self.root, height=5, width=50)
        self.transcription_text.pack(pady=10, padx=20)
        
        # Sentiment display
        ttk.Label(self.root, text="Sentiment Analysis:", font=('Arial', 10, 'bold')).pack(pady=5)
        self.sentiment_frame = ttk.Frame(self.root)
        self.sentiment_frame.pack(pady=10)
        
        self.sentiment_label = ttk.Label(self.sentiment_frame, text="No sentiment yet", font=('Arial', 12))
        self.sentiment_label.pack()
        
        # Confidence bar
        self.confidence_bar = ttk.Progressbar(self.root, length=200, mode='determinate')
        self.confidence_bar.pack(pady=10)
        
        # Start/Stop button
        self.toggle_button = ttk.Button(self.root, text="Start Recording", command=self.toggle_recording)
        self.toggle_button.pack(pady=20)
        
    def audio_callback(self, indata, frames, time, status):
        self.audio_data = indata[:, 0]
        
    def update_audio_meter(self):
        if self.is_recording:
            self.audio_meter.update_levels(self.audio_data)
            self.root.after(50, self.update_audio_meter)  # Update every 50ms
        else:
            self.audio_meter.update_levels(None)
            
    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.toggle_button.config(text="Stop Recording")
            self.status_label.config(text="Recording...")
            
            # Start audio monitoring
            self.stream = sd.InputStream(
                channels=1,
                callback=self.audio_callback,
                samplerate=16000
            )
            self.stream.start()
            self.update_audio_meter()
            
            # Start recording in a separate thread
            self.record_thread = threading.Thread(target=self.record_and_analyze)
            self.record_thread.daemon = True
            self.record_thread.start()
        else:
            self.is_recording = False
            self.toggle_button.config(text="Start Recording")
            self.status_label.config(text="Stopped")
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()
            
    def record_and_analyze(self):
        while self.is_recording:
            try:
                # Record and transcribe
                transcription = self.transcriber.transcribe_realtime()
                self.root.after(0, self.update_transcription, transcription)
                
                # Analyze sentiment
                sentiment = self.sentiment_analyzer.analyze_text(transcription)
                self.root.after(0, self.update_sentiment, sentiment)
                
            except Exception as e:
                self.root.after(0, self.show_error, str(e))
                break
    
    def update_transcription(self, text):
        self.transcription_text.delete(1.0, tk.END)
        self.transcription_text.insert(tk.END, text)
    
    def update_sentiment(self, sentiment):
        label = sentiment['label']
        score = sentiment['score']
        
        # Update sentiment label with color
        color = '#2ecc71' if label == 'POSITIVE' else '#e74c3c'
        self.sentiment_label.config(
            text=f"{label}",
            foreground=color
        )
        
        # Update confidence bar
        self.confidence_bar['value'] = score * 100
    
    def show_error(self, error_msg):
        self.status_label.config(text=f"Error: {error_msg}")
        self.is_recording = False
        self.toggle_button.config(text="Start Recording")

def main():
    root = tk.Tk()
    app = SentimentAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 