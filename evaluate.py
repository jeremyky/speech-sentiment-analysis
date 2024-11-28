from src.sentiment_analysis.bert_model import SentimentAnalyzer
from src.speech_recognition.whisper_client import WhisperTranscriber
import numpy as np
import os
from sklearn.preprocessing import MultiLabelBinarizer
import soundfile as sf
import glob
import kagglehub

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

def main():
    print("Starting Evaluation...")
    try:
        # First evaluate speech recognition
        evaluate_speech_recognition()
        
        # Then evaluate sentiment analysis
        evaluate_sentiment_analysis()
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 