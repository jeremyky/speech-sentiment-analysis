import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
import time
from src.speech_recognition.whisper_client import WhisperTranscriber
from src.sentiment_analysis.bert_model import SentimentAnalyzer
import sounddevice as sd
from tkinter import filedialog
import os
import wave
import contextlib
import tempfile
from pydub import AudioSegment
from pydub.utils import mediainfo
from datetime import datetime
import soundfile as sf
from src.sentiment_analysis.evaluate import analyze_hierarchical_text

# Try to import video functionality, but make it optional
try:
    from pydub.utils import mediainfo
    VIDEO_SUPPORT = True
except ImportError:
    print("Video support not available. Install moviepy for video file support.")
    VIDEO_SUPPORT = False

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

class AudioMeter(tk.Canvas):
    def __init__(self, parent, width=30, height=200):
        super().__init__(parent, width=width, height=height, bg='black')
        self.configure(highlightthickness=1, highlightbackground='gray')
        self.width = width
        self.height = height
        self.level = 0
        
    def update_levels(self, audio_data):
        if audio_data is None or len(audio_data) == 0:
            self.level = 0
        else:
            # Calculate RMS value
            rms = np.sqrt(np.mean(np.square(audio_data)))
            # Scale to 0-1 range with some headroom
            self.level = min(1.0, rms * 4)
        self.draw_meter()
        
    def draw_meter(self):
        self.delete("all")
        
        # Draw background
        self.create_rectangle(0, 0, self.width, self.height, fill='black')
        
        # Draw level bar
        bar_height = int(self.level * self.height)
        if self.level < 0.3:
            color = '#2ecc71'  # Green
        elif self.level < 0.7:
            color = '#f1c40f'  # Yellow
        else:
            color = '#e74c3c'  # Red
            
        self.create_rectangle(
            2,                    # Left padding
            self.height - bar_height,  # Top
            self.width - 2,       # Right padding
            self.height,          # Bottom
            fill=color,
            outline=''
        )

class SentimentAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Emotion Analysis")
        
        # Initialize components
        self.transcriber = WhisperTranscriber()
        self.analyzer = SentimentAnalyzer()
        
        # Audio recording variables
        self.is_recording = False
        self.recorded_audio = None
        self.current_transcription = ""
        
        # Create recordings directory
        self.recordings_dir = "recordings"
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        # Setup GUI components
        self.setup_gui()
        
        # Initialize audio meter
        self.setup_audio_meter()
        
        # Start audio level monitoring
        self.update_audio_meter()
        
    def setup_gui(self):
        # Add File Upload Frame at the top
        upload_frame = ttk.LabelFrame(self.root, text="File Upload")
        upload_frame.pack(pady=5, padx=20, fill='x')
        
        # Audio File Upload Button
        self.audio_upload_btn = ttk.Button(
            upload_frame,
            text="Upload Audio File",
            command=self.upload_audio
        )
        self.audio_upload_btn.pack(side='left', padx=5, pady=5)
        
        # Video File Upload Button
        self.video_upload_btn = ttk.Button(
            upload_frame,
            text="Upload Video File",
            command=self.upload_video
        )
        self.video_upload_btn.pack(side='left', padx=5, pady=5)
        
        # File info label
        self.file_info_label = ttk.Label(
            upload_frame,
            text="No file selected",
            font=('Arial', 10)
        )
        self.file_info_label.pack(side='left', padx=10)
        
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
        self.transcription_text = tk.Text(
            self.root,
            height=10,
            width=50,
            wrap=tk.WORD,
            font=('Arial', 10)
        )
        self.transcription_text.pack(pady=10, padx=20)
        
        # Emotion analysis display
        ttk.Label(
            self.root, 
            text="Emotion Analysis:", 
            font=('Arial', 10, 'bold')
        ).pack(pady=5)
        self.emotion_text = tk.Text(
            self.root,
            height=10,
            width=50,
            wrap=tk.WORD,
            font=('Arial', 10)
        )
        self.emotion_text.pack(pady=10, padx=20)
        
        # Add Analysis Mode Frame
        analysis_frame = ttk.LabelFrame(self.root, text="Analysis Mode")
        analysis_frame.pack(pady=5, padx=20, fill='x')
        
        # Add radio buttons for analysis mode
        self.analysis_mode = tk.StringVar(value="segment")
        ttk.Radiobutton(
            analysis_frame,
            text="Analyze Current Segment",
            variable=self.analysis_mode,
            value="segment"
        ).pack(side='left', padx=5)
        
        ttk.Radiobutton(
            analysis_frame,
            text="Analyze Full Transcript",
            variable=self.analysis_mode,
            value="full"
        ).pack(side='left', padx=5)
        
        # Recordings Frame
        recordings_frame = ttk.LabelFrame(self.root, text="Recording Controls")
        recordings_frame.pack(pady=5, padx=20, fill='x')
        
        # Playback controls
        self.play_button = ttk.Button(
            recordings_frame,
            text="Play Recording",
            command=self.play_recording,
            state='disabled'
        )
        self.play_button.pack(side='left', padx=5)
        
        self.save_button = ttk.Button(
            recordings_frame,
            text="Save Recording",
            command=self.save_recording,
            state='disabled'
        )
        self.save_button.pack(side='left', padx=5)
        
    def toggle_recording(self):
        """Toggle recording state"""
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.record_button.config(text="Stop Recording")
            self.status_label.config(text="Recording...")
            
            # Start recording thread
            self.recording_thread = threading.Thread(target=self.record_and_analyze)
            self.recording_thread.daemon = True
            self.recording_thread.start()
        else:
            # Stop recording
            self.is_recording = False
            self.record_button.config(text="Start Recording")
            self.status_label.config(text="Recording stopped")
            
            # Show post-recording options
            self.new_recording_button.pack(side='left', padx=5)
            self.continue_recording_button.pack(side='left', padx=5)

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
        """Update audio meter display"""
        try:
            if self.is_recording:
                # Get audio data
                audio_data = sd.rec(
                    frames=1024,
                    samplerate=self.transcriber.sample_rate,
                    channels=1,
                    dtype=np.float32
                )
                sd.wait()
                
                # Update meter
                self.audio_meter.update_levels(audio_data)
            else:
                self.audio_meter.update_levels(None)
                
            # Schedule next update
            self.root.after(50, self.update_audio_meter)
            
        except Exception as e:
            print(f"Error updating audio meter: {e}")
    
    def record_and_analyze(self):
        """Record audio and analyze in real-time"""
        while self.is_recording:
            try:
                # Record and transcribe
                transcription, audio_data = self.transcriber.transcribe_realtime()
                
                if transcription and not transcription.isspace():
                    self.current_transcription += f"{transcription}\n"
                    
                    # Update displays in GUI thread
                    self.root.after(0, lambda: self.update_displays(self.current_transcription))
                    
                    # Store the recorded audio
                    if self.recorded_audio is None:
                        self.recorded_audio = audio_data
                    else:
                        self.recorded_audio = np.concatenate((self.recorded_audio, audio_data))
                    
                    # Enable playback controls
                    self.root.after(0, lambda: self.play_button.config(state='normal'))
                    self.root.after(0, lambda: self.save_button.config(state='normal'))
                    
                # Update audio meter
                if audio_data is not None:
                    self.root.after(0, lambda: self.audio_meter.update_levels(audio_data))
                    
                time.sleep(0.1)  # Small delay to prevent CPU overload
                    
            except Exception as e:
                print(f"Error in record_and_analyze: {e}")
                self.root.after(0, lambda: self.show_error(str(e)))
                break
    
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

    def upload_audio(self):
        """Handle audio file upload"""
        filetypes = (
            ('Audio files', '*.wav *.mp3 *.m4a *.aac *.flac'),
            ('All files', '*.*')
        )
        
        filename = filedialog.askopenfilename(
            title='Open audio file',
            filetypes=filetypes
        )
        
        if filename:
            self.process_audio_file(filename)

    def upload_video(self):
        """Handle video file upload"""
        filetypes = (
            ('Video files', '*.mp4 *.mov *.avi *.mkv'),
            ('All files', '*.*')
        )
        
        filename = filedialog.askopenfilename(
            title='Open video file',
            filetypes=filetypes
        )
        
        if filename:
            self.process_video_file(filename)

    def process_audio_file(self, filepath):
        """Process uploaded audio file"""
        try:
            # Update status
            self.status_label.config(text=f"Processing audio file: {os.path.basename(filepath)}")
            self.file_info_label.config(text=f"File: {os.path.basename(filepath)}")
            
            # Transcribe audio
            transcription = self.transcriber.transcribe_file(filepath)
            self.current_transcription = transcription
            
            # Update transcription display
            self.transcription_text.delete(1.0, tk.END)
            self.transcription_text.insert(tk.END, transcription)
            
            # Analyze emotions hierarchically
            self.update_hierarchical_display(transcription)
            
        except Exception as e:
            self.show_error(f"Error processing audio: {str(e)}")

    def update_hierarchical_display(self, text):
        """Update display with hierarchical emotion analysis"""
        if not text.strip():
            return
        
        # Get hierarchical analysis
        results = analyze_hierarchical_text(text, self.analyzer)
        
        # Clear previous displays
        self.emotion_text.delete(1.0, tk.END)
        
        # Display overall summary
        self.emotion_text.insert(tk.END, "Overall Sentiment:\n", "heading")
        self.emotion_text.insert(tk.END, "-" * 40 + "\n")
        for emotion in results["overall"]["emotions"][:3]:
            self.emotion_text.insert(tk.END, 
                f"{emotion['label'].capitalize()}: {emotion['score']:.1%}\n",
                emotion['label'].lower())
        
        # Display timeline analysis
        self.emotion_text.insert(tk.END, "\nEmotion Timeline:\n", "heading")
        self.emotion_text.insert(tk.END, "-" * 40 + "\n")
        
        # Process each sentence with timestamp estimation
        total_words = len(text.split())
        words_per_second = 2.5  # Average speaking rate
        estimated_duration = total_words / words_per_second
        
        for i, sent in enumerate(results["sentences"]):
            # Estimate timestamp
            progress = i / len(results["sentences"])
            timestamp = estimated_duration * progress
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            
            # Format display
            self.emotion_text.insert(tk.END, 
                f"\n[{minutes:02d}:{seconds:02d}] ", "timestamp")
            
            # Show sentence with dominant emotion
            self.emotion_text.insert(tk.END, f"{sent['text']}\n")
            if sent["dominant_emotion"]:
                emotion = sent["dominant_emotion"]
                self.emotion_text.insert(tk.END,
                    f"→ {emotion['label'].capitalize()} ({emotion['score']:.1%})\n",
                    emotion['label'].lower())
        
        # Add text tags for formatting
        self.emotion_text.tag_configure("heading", font=("Arial", 12, "bold"))
        self.emotion_text.tag_configure("timestamp", font=("Courier", 10))
        for emotion in ["joy", "sadness", "anger", "fear", "surprise", "disgust"]:
            self.emotion_text.tag_configure(emotion, foreground=self.get_emotion_color(emotion))

    def process_video_file(self, filepath):
        """Process uploaded video file"""
        try:
            # Update status
            self.status_label.config(text=f"Processing video file: {os.path.basename(filepath)}")
            self.file_info_label.config(text=f"File: {os.path.basename(filepath)}")
            
            # Extract audio from video using pydub
            audio = AudioSegment.from_file(filepath)
            
            # Create temporary audio file
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            audio.export(temp_audio.name, format='wav')
            
            # Process the extracted audio
            self.process_audio_file(temp_audio.name)
            
            # Clean up
            os.unlink(temp_audio.name)
            
        except Exception as e:
            self.show_error(f"Error processing video: {str(e)}")

    def play_recording(self):
        """Play the current recording"""
        if self.recorded_audio is not None:
            try:
                sd.play(self.recorded_audio, self.transcriber.sample_rate)
                sd.wait()  # Wait until audio is finished playing
            except Exception as e:
                self.show_error(f"Error playing audio: {str(e)}")

    def save_recording(self):
        """Save the current recording with timestamp"""
        if self.recorded_audio is not None:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recording_{timestamp}.wav"
                filepath = os.path.join(self.recordings_dir, filename)
                
                # Save audio
                sf.write(filepath, self.recorded_audio, self.transcriber.sample_rate)
                
                # Save transcription
                transcript_path = os.path.join(self.recordings_dir, f"transcript_{timestamp}.txt")
                with open(transcript_path, 'w') as f:
                    f.write(self.current_transcription)
                
                self.status_label.config(text=f"Recording saved: {filename}")
                
            except Exception as e:
                self.show_error(f"Error saving recording: {str(e)}")

    def get_emotion_color(self, emotion):
        """Get the color for a given emotion"""
        # Implement your logic to determine the color based on the emotion
        # For example, you can use a dictionary to map emotions to colors
        emotion_colors = {
            "joy": "#2ecc71",
            "sadness": "#e74c3c",
            "anger": "#e74c3c",
            "fear": "#9b59b6",
            "surprise": "#f1c40f",
            "disgust": "#e74c3c"
        }
        return emotion_colors.get(emotion, "#000000")

    def setup_audio_meter(self):
        """Setup audio meter components"""
        # Create audio meter frame
        meter_frame = ttk.Frame(self.root)
        meter_frame.pack(pady=5)
        
        # Create audio meter
        self.audio_meter = AudioMeter(meter_frame)
        self.audio_meter.pack(side='left', padx=5)
        
        # Create emotion display
        self.emotion_display = EmotionDisplay(self.root)
        self.emotion_display.pack(pady=10, padx=20)
        
        # Create analysis mode selector
        self.analysis_mode = tk.StringVar(value="full")
        mode_frame = ttk.Frame(self.root)
        mode_frame.pack(pady=5)
        
        ttk.Radiobutton(
            mode_frame,
            text="Full Analysis",
            variable=self.analysis_mode,
            value="full"
        ).pack(side='left', padx=5)
        
        ttk.Radiobutton(
            mode_frame,
            text="Segment Analysis",
            variable=self.analysis_mode,
            value="segment"
        ).pack(side='left', padx=5)
        
        # Create playback controls
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=5)
        
        self.play_button = ttk.Button(
            control_frame,
            text="Play",
            command=self.play_recording,
            state='disabled'
        )
        self.play_button.pack(side='left', padx=5)
        
        self.save_button = ttk.Button(
            control_frame,
            text="Save",
            command=self.save_recording,
            state='disabled'
        )
        self.save_button.pack(side='left', padx=5)

    def update_displays(self, text):
        """Update both transcription and emotion displays"""
        # Update transcription
        self.transcription_text.delete(1.0, tk.END)
        self.transcription_text.insert(tk.END, text)
        
        # Update emotion analysis
        self.update_hierarchical_display(text)
        
        # Scroll both displays to show latest content
        self.transcription_text.see(tk.END)
        self.emotion_text.see(tk.END)

def main():
    root = tk.Tk()
    app = SentimentAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 