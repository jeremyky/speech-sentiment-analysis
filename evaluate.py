from src.sentiment_analysis.bert_model import SentimentAnalyzer
from src.speech_recognition.whisper_client import WhisperTranscriber
import numpy as np
import os
import tempfile
import soundfile as sf
import pandas as pd

def evaluate_with_test_data():
    """Evaluate using a small test dataset"""
    print("Starting evaluation with test data...")
    
    transcriber = WhisperTranscriber()
    analyzer = SentimentAnalyzer()
    
    # Test cases with known ground truth
    test_cases = [
        {
            "text": "I am really happy about this amazing result!",
            "ground_truth_emotions": ["happiness"],
            "audio_file": None  # Optional: Add path to test audio file
        },
        {
            "text": "This makes me so angry and frustrated.",
            "ground_truth_emotions": ["anger", "disgust"],
            "audio_file": None
        },
        {
            "text": "I'm feeling quite sad and disappointed today.",
            "ground_truth_emotions": ["sadness"],
            "audio_file": None
        },
        {
            "text": "Wow, this is such a surprising and exciting development!",
            "ground_truth_emotions": ["surprise", "happiness"],
            "audio_file": None
        }
    ]
    
    print("\nEvaluation Results:")
    print("-" * 50)
    
    all_predictions = []
    all_ground_truth = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Text: {test['text']}")
        
        # Get emotion predictions
        emotions = analyzer.analyze_text(test['text'])
        predictions = [emotion["label"] for emotion in emotions]
        
        # Store for metrics calculation
        all_predictions.extend(predictions)
        all_ground_truth.extend(test["ground_truth_emotions"])
        
        print(f"Ground Truth Emotions: {test['ground_truth_emotions']}")
        print(f"Predicted Emotions: {predictions}")
        
        # If audio file is provided, test transcription
        if test["audio_file"] and os.path.exists(test["audio_file"]):
            transcription = transcriber.transcribe_file(test["audio_file"])
            wer = transcriber.calculate_wer(test["text"], transcription)
            print(f"Transcribed Text: {transcription}")
            print(f"Word Error Rate: {wer:.2%}" if wer else "WER: N/A")
    
    # Calculate overall metrics
    print("\nOverall Metrics:")
    print("-" * 50)
    
    metrics = analyzer.evaluate_metrics(all_predictions, all_ground_truth)
    if metrics:
        print(f"Emotion Detection Precision: {metrics['precision']:.2%}")
        print(f"Emotion Detection Recall: {metrics['recall']:.2%}")
        print(f"Emotion Detection F1: {metrics['f1']:.2%}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])

def main():
    print("Starting Evaluation...")
    try:
        evaluate_with_test_data()
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 