import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
import time
from src.speech_recognition.whisper_client import WhisperTranscriber
from src.sentiment_analysis.bert_model import SentimentAnalyzer
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

class EmotionDisplay(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        # Initialize with a waiting message
        self.setup_initial_display()
        
    def setup_initial_display(self):
        frame = ttk.Frame(self)
        frame.pack(fill='x', pady=2)
        label = ttk.Label(
            frame,
            text="Waiting for speech...",
            font=('Arial', 10)
        )
        label.pack(side='left', padx=5)
        
    def update_emotions(self, emotions):
        # Clear previous emotions
        for widget in self.winfo_children():
            widget.destroy()
        
        if not emotions:
            self.setup_initial_display()
            return
            
        # Create new emotion displays
        for emotion in emotions:
            frame = ttk.Frame(self)
            frame.pack(fill='x', pady=2)
            
            # Emotion label with percentage
            label_text = f"{emotion['label'].capitalize()}: {emotion['score']:.1%}"
            label = ttk.Label(
                frame, 
                text=label_text,
                foreground=emotion['color'],
                font=('Arial', 10, 'bold')
            )
            label.pack(side='left', padx=5)
            
            # Confidence bar
            bar = ttk.Progressbar(frame, length=200, mode='determinate')
            bar.pack(side='right', padx=5)
            bar['value'] = emotion['score'] * 100

class SentimentAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech-to-Sentiment Analyzer")
        self.root.geometry("600x700")
        
        # Initialize components
        self.transcriber = WhisperTranscriber()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.is_recording = False
        self.audio_data = None
        self.cycle_time = 5  # 5 seconds per cycle
        self.current_transcription = ""
        
        self.setup_gui()
        
    def setup_gui(self):
        # Control Panel Frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=5, padx=20, fill='x')
        
        # Single Record Button
        self.record_button = ttk.Button(
            control_frame, 
            text="Start Recording", 
            command=self.toggle_recording
        )
        self.record_button.pack(side='left', padx=5)
        
        # Post-Recording Controls Frame (initially hidden)
        self.post_record_frame = ttk.Frame(control_frame)
        self.post_record_frame.pack(side='left', padx=5)
        
        # New Recording Button (initially hidden)
        self.new_recording_button = ttk.Button(
            self.post_record_frame,
            text="New Recording",
            command=self.start_new_recording
        )
        
        # Continue Recording Button (initially hidden)
        self.continue_recording_button = ttk.Button(
            self.post_record_frame,
            text="Continue Recording",
            command=self.continue_recording
        )
        
        # Timer display
        self.timer_frame = ttk.Frame(self.root)
        self.timer_frame.pack(pady=5)
        self.timer_label = ttk.Label(
            self.timer_frame, 
            text="Time remaining: --",
            font=('Arial', 12)
        )
        self.timer_label.pack(side='left', padx=5)
        
        # Status label
        self.status_label = ttk.Label(
            self.root, 
            text="Click 'Start Recording' to begin", 
            font=('Arial', 12)
        )
        self.status_label.pack(pady=5)
        
        # Audio meter
        self.audio_meter = AudioMeter(self.root)
        self.audio_meter.pack(pady=10, padx=20)
        
        # Transcription display
        ttk.Label(
            self.root, 
            text="Transcription:", 
            font=('Arial', 10, 'bold')
        ).pack(pady=5)
        self.transcription_text = tk.Text(self.root, height=5, width=50)
        self.transcription_text.pack(pady=10, padx=20)
        
        # Emotion display
        ttk.Label(
            self.root, 
            text="Emotion Analysis:", 
            font=('Arial', 10, 'bold')
        ).pack(pady=5)
        self.emotion_display = EmotionDisplay(self.root)
        self.emotion_display.pack(pady=10, padx=20, fill='x')
        
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        self.is_recording = True
        self.record_button.config(text="Stop Recording")
        self.status_label.config(text="Recording...")
        
        # Hide post-recording controls
        self.new_recording_button.pack_forget()
        self.continue_recording_button.pack_forget()
        
        # Start audio monitoring
        self.stream = sd.InputStream(
            channels=1,
            callback=self.audio_callback,
            samplerate=16000
        )
        self.stream.start()
        self.update_audio_meter()
        self.update_timer(self.cycle_time)
        
        # Start recording in a separate thread
        self.record_thread = threading.Thread(target=self.record_and_analyze)
        self.record_thread.daemon = True
        self.record_thread.start()
        
    def stop_recording(self):
        self.is_recording = False
        self.record_button.config(text="Start Recording")
        self.status_label.config(text="Recording complete. Review or start new/continue recording.")
        
        # Show post-recording controls
        self.new_recording_button.pack(side='left', padx=5)
        self.continue_recording_button.pack(side='left', padx=5)
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
            
    def start_new_recording(self):
        self.current_transcription = ""
        self.update_transcription("")
        self.emotion_display.update_emotions([])
        self.start_recording()
        
    def continue_recording(self):
        self.start_recording()  # Continues with existing transcription
            
    def update_timer(self, remaining):
        if remaining >= 0 and self.is_recording:
            self.timer_label.config(text=f"Time remaining: {remaining}s")
            self.root.after(1000, self.update_timer, remaining - 1)
        elif remaining < 0 and self.is_recording:
            self.stop_recording()
            
    def audio_callback(self, indata, frames, time, status):
        self.audio_data = indata[:, 0]
        
    def update_audio_meter(self):
        if self.is_recording:
            self.audio_meter.update_levels(self.audio_data)
            self.root.after(50, self.update_audio_meter)
        else:
            self.audio_meter.update_levels(None)
            
    def record_and_analyze(self):
        try:
            # Record and transcribe
            transcription = self.transcriber.transcribe_realtime()
            if transcription and not transcription.isspace():
                self.current_transcription += f"{transcription}\n"
                self.root.after(0, self.update_transcription, self.current_transcription)
                
                # Analyze sentiment
                emotions = self.sentiment_analyzer.analyze_text(transcription)
                self.root.after(0, self.update_sentiment, emotions)
                
        except Exception as e:
            print(f"Error in record_and_analyze: {e}")
            self.root.after(0, self.show_error, str(e))
    
    def update_transcription(self, text):
        self.transcription_text.delete(1.0, tk.END)
        self.transcription_text.insert(tk.END, text)
        self.transcription_text.see(tk.END)
    
    def update_sentiment(self, emotions):
        try:
            self.emotion_display.update_emotions(emotions)
        except Exception as e:
            print(f"Error updating sentiment display: {e}")
    
    def show_error(self, error_msg):
        self.status_label.config(text=f"Error: {error_msg}")
        self.is_recording = False
        self.record_button.config(text="Start Recording")

def main():
    root = tk.Tk()
    app = SentimentAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 