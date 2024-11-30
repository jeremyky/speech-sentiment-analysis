from src.sentiment_analysis.bert_model import SentimentAnalyzer
from src.speech_recognition.whisper_client import WhisperTranscriber
import numpy as np
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score
import soundfile as sf
import glob
import kagglehub
import librosa
import tempfile
import traceback
import json

def download_ravdess():
    """Download RAVDESS dataset using kagglehub"""
    print("Downloading RAVDESS dataset...")
    try:
        dataset_path = kagglehub.dataset_download(
            "uwrfkaggler/ravdess-emotional-speech-audio"
        )
        print(f"Dataset downloaded to: {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def evaluate_speech_recognition():
    """Evaluate speech-to-text accuracy using RAVDESS"""
    print("\nSpeech Recognition Evaluation:")
    print("-" * 50)
    
    try:
        dataset_path = download_ravdess()
        if not dataset_path:
            return
        
        transcriber = WhisperTranscriber()
        total_wer = 0
        successful_tests = 0
        
        # Get all audio files
        audio_files = glob.glob(os.path.join(dataset_path, "**/*.wav"), recursive=True)
        
        print("\nTranscription Results:")
        print("-" * 50)
        
        # Test first 10 samples
        for i, audio_file in enumerate(audio_files[:10]):
            # Get ground truth text from filename
            filename = os.path.basename(audio_file)
            parts = filename.split("-")
            text = "Kids are talking by the door" if parts[4] == "01" else "Dogs are sitting by the door"
            
            print(f"\nTest Case {i+1}:")
            print(f"Audio File: {filename}")
            print(f"Ground Truth: {text}")
            
            # Test transcription
            transcription = transcriber.transcribe_file(audio_file)
            wer = transcriber.calculate_wer(text, transcription)
            total_wer += wer
            successful_tests += 1
            
            print(f"Transcribed: {transcription}")
            print(f"Word Error Rate: {wer:.2%}")
        
        if successful_tests > 0:
            print("\nOverall Speech Recognition Results:")
            print("-" * 40)
            print(f"Average Word Error Rate: {total_wer/successful_tests:.2%}")
            
    except Exception as e:
        print(f"Error in speech recognition evaluation: {e}")
        import traceback
        traceback.print_exc()

def evaluate_sentiment_analysis():
    """Evaluate text-to-emotion accuracy using predefined test cases"""
    print("\nSentiment Analysis Evaluation:")
    print("-" * 50)
    
    try:
        analyzer = SentimentAnalyzer()
        
        # Test cases with known emotions
        test_cases = [
            {
                "text": "I am really happy about this amazing result!",
                "ground_truth": ["joy"]
            },
            {
                "text": "This makes me so angry and frustrated.",
                "ground_truth": ["anger", "disgust"]
            },
            {
                "text": "I'm feeling quite sad and disappointed today.",
                "ground_truth": ["sadness"]
            },
            {
                "text": "That was really scary and frightening.",
                "ground_truth": ["fear"]
            },
            {
                "text": "Wow, what a surprising turn of events!",
                "ground_truth": ["surprise"]
            }
        ]
        
        all_predictions = []
        all_ground_truth = []
        
        print("\nEmotion Detection Results:")
        print("-" * 50)
        
        for i, test in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            print(f"Text: {test['text']}")
            print(f"Ground Truth Emotions: {test['ground_truth']}")
            
            # Get emotion predictions
            emotions = analyzer.analyze_text(test['text'])
            predictions = [emotion["label"].lower() for emotion in emotions if emotion["score"] > 0.2]
            
            # Store for metrics
            all_predictions.append(predictions)
            all_ground_truth.append(test["ground_truth"])
            
            print(f"Predicted Emotions: {predictions}")
            print("\nDetailed Scores:")
            print("-" * 40)
            for emotion in emotions:
                print(f"{emotion['label'].capitalize()}: {emotion['score']:.1%}")
        
        # Calculate metrics
        mlb = MultiLabelBinarizer()
        y_true = mlb.fit_transform(all_ground_truth)
        y_pred = mlb.transform(all_predictions)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        print("\nOverall Sentiment Analysis Results:")
        print("-" * 40)
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"F1 Score: {f1:.2%}")
        print("\nEmotion Labels:", mlb.classes_)
            
    except Exception as e:
        print(f"Error in sentiment analysis evaluation: {e}")
        import traceback
        traceback.print_exc()

def evaluate_speech_recognition_robustness():
    """Evaluate speech recognition under different conditions"""
    print("\nSpeech Recognition Robustness Evaluation:")
    print("-" * 50)
    
    try:
        dataset_path = download_ravdess()
        if not dataset_path:
            return
        
        transcriber = WhisperTranscriber()
        
        # Enhanced conditions with audio preprocessing
        conditions = {
            "normal": lambda x, sr: x,  # No modification
            "noise": lambda x, sr: x + np.random.normal(0, 0.01, len(x)),
            "low_quality_raw": lambda x, sr: librosa.resample(x, orig_sr=sr, target_sr=8000),
            "low_quality_enhanced": lambda x, sr: enhance_audio(librosa.resample(x, orig_sr=sr, target_sr=8000), sr),
            "low_quality_denoise": lambda x, sr: denoise_audio(librosa.resample(x, orig_sr=sr, target_sr=8000)),
            "low_quality_normalize": lambda x, sr: normalize_audio(librosa.resample(x, orig_sr=sr, target_sr=8000))
        }
        
        for condition, modifier in conditions.items():
            print(f"\nTesting condition: {condition}")
            total_wer = 0
            successful_tests = 0
            
            # Test first 5 samples for each condition
            audio_files = glob.glob(os.path.join(dataset_path, "**/*.wav"), recursive=True)[:5]
            
            for audio_file in audio_files:
                # Load and modify audio
                audio, sr = sf.read(audio_file)
                modified_audio = modifier(audio, sr)
                
                # Get ground truth
                filename = os.path.basename(audio_file)
                parts = filename.split("-")
                text = "Kids are talking by the door" if parts[4] == "01" else "Dogs are sitting by the door"
                
                # Test transcription
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    sf.write(temp_audio.name, modified_audio, sr)
                    transcription = transcriber.transcribe_file(temp_audio.name)
                    wer = transcriber.calculate_wer(text, transcription)
                    total_wer += wer
                    successful_tests += 1
                    print(f"File: {filename}")
                    print(f"Ground Truth: {text}")
                    print(f"Transcribed: {transcription}")
                    print(f"WER: {wer:.2%}")
                    os.unlink(temp_audio.name)
            
            print(f"Average WER for {condition}: {total_wer/successful_tests:.2%}")
            
    except Exception as e:
        print(f"Error in robustness evaluation: {e}")
        traceback.print_exc()

def enhance_audio(audio, sr):
    """Enhance low quality audio"""
    # Apply a high-shelf filter to boost high frequencies
    y_filter = librosa.effects.preemphasis(audio)
    
    # Apply dynamic range compression using librosa's decompose
    S = librosa.stft(y_filter)
    S_harmonic, S_percussive = librosa.decompose.hpss(S)
    y_compressed = librosa.istft(S_harmonic)
    
    # Normalize
    y_compressed = librosa.util.normalize(y_compressed)
    
    return y_compressed

def denoise_audio(audio):
    """Remove noise from audio using spectral gating"""
    # Convert to spectrogram
    D = librosa.stft(audio)
    
    # Estimate noise profile
    noise_profile = np.mean(np.abs(D[:, :10]), axis=1, keepdims=True)
    
    # Apply spectral subtraction
    D_cleaned = D * (np.abs(D) > 2*noise_profile)
    
    # Convert back to time domain
    y_denoised = librosa.istft(D_cleaned)
    
    return y_denoised

def normalize_audio(audio):
    """Normalize audio volume"""
    # Peak normalization
    y_peak = librosa.util.normalize(audio, norm=np.inf)
    
    # RMS normalization
    y_rms = librosa.util.normalize(audio, norm=2)
    
    # Return the average of both normalizations
    return (y_peak + y_rms) / 2

def evaluate_sentiment_analysis_edge_cases():
    """Evaluate sentiment analysis on edge cases"""
    print("\nSentiment Analysis Edge Cases Evaluation:")
    print("-" * 50)
    
    try:
        analyzer = SentimentAnalyzer()
        
        edge_cases = [
            {
                "text": "I'm happy but also a bit nervous about the presentation.",
                "ground_truth": ["joy", "fear"],
                "category": "Mixed emotions"
            },
            {
                "text": "This is just a normal day, nothing special.",
                "ground_truth": ["neutral"],
                "category": "Neutral"
            },
            {
                "text": "😊 This makes me so happy! 🎉",
                "ground_truth": ["joy"],
                "category": "With emojis"
            },
            {
                "text": "ABSOLUTELY FURIOUS!!!",
                "ground_truth": ["anger"],
                "category": "All caps"
            },
            {
                "text": "i guess im kinda sad... idk...",
                "ground_truth": ["sadness"],
                "category": "Informal text"
            }
        ]
        
        all_predictions = []
        all_ground_truth = []
        
        print("\nEdge Case Results:")
        print("-" * 50)
        
        for i, test in enumerate(edge_cases, 1):
            print(f"\nTest Case {i} ({test['category']}):")
            print(f"Text: {test['text']}")
            print(f"Ground Truth Emotions: {test['ground_truth']}")
            
            # Get emotion predictions
            emotions = analyzer.analyze_text(test['text'])
            predictions = [emotion["label"].lower() for emotion in emotions if emotion["score"] > 0.2]
            
            # Store for metrics
            all_predictions.append(predictions)
            all_ground_truth.append(test["ground_truth"])
            
            print(f"Predicted Emotions: {predictions}")
            print("\nDetailed Scores:")
            print("-" * 40)
            for emotion in emotions:
                print(f"{emotion['label'].capitalize()}: {emotion['score']:.1%}")
        
        # Calculate metrics
        mlb = MultiLabelBinarizer()
        y_true = mlb.fit_transform(all_ground_truth)
        y_pred = mlb.transform(all_predictions)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        print("\nEdge Case Metrics:")
        print("-" * 40)
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"F1 Score: {f1:.2%}")
        print("\nEmotion Labels:", mlb.classes_)
            
    except Exception as e:
        print(f"Error in edge case evaluation: {e}")
        import traceback
        traceback.print_exc()

def evaluate_end_to_end():
    """Evaluate complete pipeline: audio -> text -> emotion"""
    print("\nEnd-to-End Pipeline Evaluation:")
    print("-" * 50)
    
    try:
        dataset_path = download_ravdess()
        if not dataset_path:
            return
        
        transcriber = WhisperTranscriber()
        analyzer = SentimentAnalyzer()
        
        # RAVDESS emotion mapping
        emotion_map = {
            "01": "neutral",
            "02": "calm",
            "03": "joy",
            "04": "sadness",
            "05": "anger",
            "06": "fear",
            "07": "disgust",
            "08": "surprise"
        }
        
        total_matches = 0
        total_tests = 0
        
        # Test first 10 samples
        audio_files = glob.glob(os.path.join(dataset_path, "**/*.wav"), recursive=True)[:10]
        
        for audio_file in audio_files:
            # Get ground truth emotion from filename
            filename = os.path.basename(audio_file)
            parts = filename.split("-")
            ground_truth_emotion = emotion_map[parts[2]]
            
            # Run complete pipeline
            transcription = transcriber.transcribe_file(audio_file)
            emotions = analyzer.analyze_text(transcription)
            predicted_emotion = emotions[0]["label"].lower() if emotions else "unknown"
            
            print(f"\nTest Case {total_tests + 1}:")
            print(f"Audio: {filename}")
            print(f"Transcription: {transcription}")
            print(f"Ground Truth Emotion: {ground_truth_emotion}")
            print(f"Predicted Emotion: {predicted_emotion}")
            
            if predicted_emotion == ground_truth_emotion:
                total_matches += 1
            total_tests += 1
        
        print("\nEnd-to-End Results:")
        print("-" * 40)
        print(f"Accuracy: {total_matches/total_tests:.2%}")
        
    except Exception as e:
        print(f"Error in end-to-end evaluation: {e}")
        import traceback
        traceback.print_exc()

def analyze_long_text(text, analyzer):
    """Analyze emotions in longer text with segmentation"""
    # Split text into words
    words = text.split()
    
    # Configuration
    segment_size = 50
    overlap = 10
    
    # Initialize results
    segments = []
    overall_emotions = {}
    
    # Process text in segments
    for i in range(0, len(words), segment_size - overlap):
        # Get segment
        segment = " ".join(words[i:i + segment_size])
        
        # Get emotions for segment
        emotions = analyzer.analyze_text(segment)
        
        # Store segment results
        segment_result = {
            "text": segment,
            "start_word": i,
            "end_word": min(i + segment_size, len(words)),
            "emotions": emotions
        }
        segments.append(segment_result)
        
        # Aggregate emotions for overall analysis
        for emotion in emotions:
            if emotion["label"] not in overall_emotions:
                overall_emotions[emotion["label"]] = []
            overall_emotions[emotion["label"]].append(emotion["score"])
    
    # Calculate overall emotion scores
    overall_scores = {
        label: np.mean(scores) for label, scores in overall_emotions.items()
    }
    
    return {
        "segments": segments,
        "overall_emotions": [
            {"label": label, "score": score} 
            for label, score in sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        ]
    }

def evaluate_long_text_analysis():
    """Evaluate emotion analysis on longer texts"""
    print("\nLong Text Emotion Analysis Evaluation:")
    print("-" * 50)
    
    analyzer = SentimentAnalyzer()
    
    test_cases = [
        {
            "text": """I started the day feeling really excited about the presentation. 
                      As I prepared, I got increasingly nervous, but tried to stay focused. 
                      During the actual presentation, I was initially terrified, but as people 
                      responded positively, I grew more confident. By the end, I felt incredibly 
                      proud of what I'd accomplished, though slightly exhausted.""",
            "expected_emotions": {
                "overall": ["pride", "anxiety", "relief"],
                "segments": {
                    "start": ["excitement", "nervousness"],
                    "middle": ["fear", "growing_confidence"],
                    "end": ["pride", "exhaustion"]
                }
            }
        }
        # Add more test cases...
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Text: {test['text']}\n")
        
        results = analyze_long_text(test['text'], analyzer)
        
        print("Overall Emotions:")
        print("-" * 40)
        for emotion in results["overall_emotions"][:3]:
            print(f"{emotion['label'].capitalize()}: {emotion['score']:.1%}")
        
        print("\nSegment Analysis:")
        print("-" * 40)
        for i, segment in enumerate(results["segments"], 1):
            print(f"\nSegment {i} (Words {segment['start_word']}-{segment['end_word']}):")
            print(f"Text: {segment['text'][:100]}...")
            print("Top Emotions:")
            for emotion in segment["emotions"][:2]:
                print(f"{emotion['label'].capitalize()}: {emotion['score']:.1%}")

def analyze_hierarchical_text(text, analyzer):
    """Analyze emotions at multiple text levels: overall, paragraph, sentence"""
    
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    
    # Initialize results structure
    results = {
        "overall": {},
        "paragraphs": [],
        "sentences": []
    }
    
    # Overall analysis
    overall_emotions = analyzer.analyze_text(text)
    results["overall"] = {
        "text": text,
        "emotions": overall_emotions,
        "dominant_emotion": overall_emotions[0] if overall_emotions else None
    }
    
    # Paragraph analysis
    for i, para in enumerate(paragraphs):
        if not para.strip():
            continue
            
        para_emotions = analyzer.analyze_text(para)
        results["paragraphs"].append({
            "index": i,
            "text": para,
            "emotions": para_emotions,
            "dominant_emotion": para_emotions[0] if para_emotions else None
        })
        
    # Sentence analysis
    import nltk
    nltk.download('punkt', quiet=True)
    sentences = nltk.sent_tokenize(text)
    
    for i, sentence in enumerate(sentences):
        sent_emotions = analyzer.analyze_text(sentence)
        results["sentences"].append({
            "index": i,
            "text": sentence,
            "emotions": sent_emotions,
            "dominant_emotion": sent_emotions[0] if sent_emotions else None
        })
    
    return results

def main():
    print("Starting Comprehensive Evaluation...")
    try:
        # Basic evaluations
        evaluate_speech_recognition()
        evaluate_sentiment_analysis()
        
        # Advanced evaluations
        evaluate_speech_recognition_robustness()
        evaluate_sentiment_analysis_edge_cases()
        evaluate_end_to_end()
        evaluate_long_text_analysis()
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 