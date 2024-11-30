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
import librosa

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
    def __init__(self, parent, width=40, height=200):  # Increased width
        super().__init__(parent, width=width, height=height, bg='white')
        self.configure(highlightthickness=1, highlightbackground='gray')
        self.width = width
        self.height = height
        self.level = 0
        self.smoothing = 0.3  # Add smoothing factor
        self.draw_meter()
        
    def update_levels(self, audio_data):
        if audio_data is None or len(audio_data) == 0:
            target_level = 0
        else:
            # Calculate RMS value with adjusted sensitivity
            rms = np.sqrt(np.mean(np.square(audio_data)))
            # Scale to 0-1 range with better calibration
            target_level = min(1.0, rms * 50)  # Reduced from 100 to 50
        
        # Apply smoothing
        self.level = (self.level * (1 - self.smoothing) + 
                     target_level * self.smoothing)
        
        self.draw_meter()
        
    def draw_meter(self):
        self.delete("all")
        
        # Draw background
        self.create_rectangle(0, 0, self.width, self.height, fill='white')
        
        # Draw level markers with labels
        for i in range(10):
            y = self.height * (i / 10)
            self.create_line(0, y, 5, y, fill='gray')
            if i % 2 == 0:  # Only show every other label
                self.create_text(
                    self.width - 12, 
                    y, 
                    text=f"{100 - (i * 10)}", 
                    font=('Arial', 7),
                    fill='gray'
                )
        
        # Draw level bar with smoother gradient
        bar_height = int(self.level * self.height)
        if bar_height > 0:
            # More segments for smoother gradient
            segments = 40  # Increased from 30
            segment_height = bar_height / segments
            
            for i in range(segments):
                y_top = self.height - (i + 1) * segment_height
                y_bottom = self.height - i * segment_height
                
                # Calculate color based on position with smoother transitions
                position = i / segments
                if position < 0.6:  # Expanded green range
                    color = self.interpolate_color('#2ecc71', '#f1c40f', position/0.6)
                elif position < 0.8:  # Shorter yellow range
                    color = self.interpolate_color('#f1c40f', '#e74c3c', (position-0.6)/0.2)
                else:  # Red range
                    color = '#e74c3c'
                
                self.create_rectangle(
                    4,              # Increased left padding
                    y_top,
                    self.width - 16,  # Space for labels
                    y_bottom,
                    fill=color,
                    outline='',
                    width=0  # Remove borders between segments
                )
    
    def interpolate_color(self, color1, color2, factor):
        """Interpolate between two colors"""
        # Convert hex to RGB
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
        r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
        
        # Interpolate
        r = int(r1 + (r2 - r1) * factor)
        g = int(g1 + (g2 - g1) * factor)
        b = int(b1 + (b2 - b1) * factor)
        
        # Convert back to hex
        return f'#{r:02x}{g:02x}{b:02x}'

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
        """Setup GUI components"""
        # Create main controls frame
        controls_frame = ttk.Frame(self.root)
        controls_frame.pack(pady=10)
        
        # Create record button
        self.record_button = ttk.Button(
            controls_frame,
            text="Start Recording",
            command=self.toggle_recording
        )
        self.record_button.pack(side='left', padx=5)
        
        # Create upload buttons
        ttk.Button(
            controls_frame,
            text="Upload Audio",
            command=self.upload_audio
        ).pack(side='left', padx=5)
        
        if VIDEO_SUPPORT:
            ttk.Button(
                controls_frame,
                text="Upload Video",
                command=self.upload_video
            ).pack(side='left', padx=5)
        
        # Create playback controls
        self.play_button = ttk.Button(
            controls_frame,
            text="Play Recording",
            command=self.play_recording,
            state='disabled'
        )
        self.play_button.pack(side='left', padx=5)
        
        self.save_button = ttk.Button(
            controls_frame,
            text="Save Recording",
            command=self.save_recording,
            state='disabled'
        )
        self.save_button.pack(side='left', padx=5)
        
        # Create status labels
        self.status_label = ttk.Label(self.root, text="Ready")
        self.status_label.pack(pady=5)
        
        self.file_info_label = ttk.Label(self.root, text="")
        self.file_info_label.pack(pady=5)
        
        # Create audio meter
        meter_frame = ttk.LabelFrame(self.root, text="Audio Level")
        meter_frame.pack(pady=5, padx=20)
        
        self.audio_meter = AudioMeter(meter_frame)
        self.audio_meter.pack(pady=5, padx=10)
        
        # Create transcription and emotion displays
        display_frame = ttk.Frame(self.root)
        display_frame.pack(pady=5, padx=20, fill='both', expand=True)
        
        # Left side: Transcription
        left_frame = ttk.LabelFrame(display_frame, text="Transcription")
        left_frame.pack(side='left', padx=5, fill='both', expand=True)
        
        self.transcription_text = tk.Text(
            left_frame,
            height=15,
            width=40,
            wrap=tk.WORD,
            font=('Arial', 10)
        )
        self.transcription_text.pack(pady=5, padx=5, fill='both', expand=True)
        
        # Right side: Emotion Analysis
        right_frame = ttk.LabelFrame(display_frame, text="Emotion Analysis")
        right_frame.pack(side='right', padx=5, fill='both', expand=True)
        
        self.emotion_text = tk.Text(
            right_frame,
            height=15,
            width=40,
            wrap=tk.WORD,
            font=('Arial', 10)
        )
        self.emotion_text.pack(pady=5, padx=5, fill='both', expand=True)
        
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
        """Record audio and analyze in real-time with optimized handling"""
        try:
            # Initialize audio input stream with optimized parameters
            chunk_size = 16384  # Increased buffer size
            buffer = []  # Audio buffer
            buffer_duration = 0  # Track buffer duration
            last_transcription = ""  # Track last transcription
            
            # Use a queue for audio processing
            import queue  # Added import here
            audio_queue = queue.Queue(maxsize=100)
            
            def audio_callback(indata, frames, time, status):
                """Callback for audio input"""
                try:
                    if status:
                        print(f"Status: {status}")
                    audio_queue.put(indata.copy())
                except queue.Full:
                    print("Queue is full, dropping audio chunk")
            
            stream = sd.InputStream(
                channels=1,
                samplerate=self.transcriber.sample_rate,
                blocksize=chunk_size,
                dtype=np.float32
            )
            
            with stream:
                while self.is_recording:
                    # Get audio data
                    audio_data, overflowed = stream.read(chunk_size)
                    audio_data = audio_data.flatten()
                    
                    # Update audio meter
                    self.root.after(0, lambda d=audio_data: self.audio_meter.update_levels(d))
                    
                    # Add to buffer
                    buffer.append(audio_data)
                    buffer_duration += len(audio_data) / self.transcriber.sample_rate
                    
                    # Process when buffer reaches ~2 seconds
                    if buffer_duration >= 2.0:
                        # Combine buffer
                        audio_segment = np.concatenate(buffer)
                        
                        # Store for playback/saving
                        if self.recorded_audio is None:
                            self.recorded_audio = audio_segment
                        else:
                            self.recorded_audio = np.concatenate((self.recorded_audio, audio_segment))
                        
                        # Transcribe
                        transcription, _ = self.transcriber.transcribe_realtime(audio_segment)
                        if transcription and transcription != last_transcription:
                            self.current_transcription += f"{transcription} "
                            # Update displays in GUI thread
                            self.root.after(0, lambda t=self.current_transcription: self.update_displays(t))
                            last_transcription = transcription
                            
                            # Enable controls
                            self.root.after(0, lambda: self.play_button.config(state='normal'))
                            self.root.after(0, lambda: self.save_button.config(state='normal'))
                        
                        # Reset buffer with overlap
                        overlap_samples = int(0.5 * self.transcriber.sample_rate)
                        if len(buffer) > 1:
                            buffer = [buffer[-1][-overlap_samples:]]
                            buffer_duration = len(buffer[0]) / self.transcriber.sample_rate
                        else:
                            buffer = []
                            buffer_duration = 0
                    
                    time.sleep(0.01)  # Small delay
                
        except Exception as e:
            print(f"Error in record_and_analyze: {e}")
            self.root.after(0, lambda err=str(e): self.show_error(err))
    
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
        print("Play button clicked")  # Debug output
        
        if self.recorded_audio is not None:
            try:
                print(f"Playing audio of length: {len(self.recorded_audio)}")  # Debug output
                
                # Update status
                self.status_label.config(text="Playing recording...")
                
                # Disable play button while playing
                self.play_button.config(state='disabled')
                
                # Play audio
                sd.play(self.recorded_audio, self.transcriber.sample_rate)
                sd.wait()  # Wait until audio is finished playing
                
                # Re-enable play button and update status
                self.play_button.config(state='normal')
                self.status_label.config(text="Ready")
                print("Finished playing audio")  # Debug output
                
            except Exception as e:
                error_msg = f"Error playing audio: {str(e)}"
                print(error_msg)  # Debug output
                self.show_error(error_msg)
                self.play_button.config(state='normal')
        else:
            print("No audio recording available to play")  # Debug output
            self.show_error("No recording available to play")

    def save_recording(self):
        """Save the current recording and its transcript"""
        print("Save button clicked")  # Debug output
        
        if self.recorded_audio is not None:
            try:
                print(f"Saving audio of length: {len(self.recorded_audio)}")  # Debug output
                
                # Create timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Create recordings directory if it doesn't exist
                if not os.path.exists(self.recordings_dir):
                    os.makedirs(self.recordings_dir)
                    print(f"Created recordings directory: {self.recordings_dir}")  # Debug output
                
                # Save audio file
                audio_filename = f"recording_{timestamp}.wav"
                audio_filepath = os.path.join(self.recordings_dir, audio_filename)
                sf.write(audio_filepath, self.recorded_audio, self.transcriber.sample_rate)
                print(f"Saved audio file: {audio_filepath}")  # Debug output
                
                # Save transcript and sentiment
                text_filename = f"transcript_{timestamp}.txt"
                text_filepath = os.path.join(self.recordings_dir, text_filename)
                
                with open(text_filepath, 'w') as f:
                    f.write("Transcription:\n")
                    f.write("-" * 40 + "\n")
                    f.write(self.current_transcription)
                    f.write("\n\nSentiment Analysis:\n")
                    f.write("-" * 40 + "\n")
                    
                    # Get current emotions from analyzer
                    emotions = self.analyzer.analyze_text(self.current_transcription)
                    for emotion in emotions:
                        f.write(f"{emotion['label'].capitalize()}: {emotion['score']:.1%}\n")
                
                print(f"Saved transcript file: {text_filepath}")  # Debug output
                
                # Update status
                self.status_label.config(
                    text=f"Saved recording as {audio_filename} and {text_filename}"
                )
                print("Save completed successfully")  # Debug output
                
            except Exception as e:
                error_msg = f"Error saving recording: {str(e)}"
                print(error_msg)  # Debug output
                self.show_error(error_msg)
        else:
            print("No audio recording available to save")  # Debug output
            self.show_error("No recording available to save")

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