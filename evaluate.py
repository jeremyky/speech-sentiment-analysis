from src.sentiment_analysis.bert_model import SentimentAnalyzer
from src.speech_recognition.whisper_client import WhisperTranscriber
import numpy as np
import os
import json
from sklearn.preprocessing import MultiLabelBinarizer

def evaluate_with_test_cases():
    """Evaluate using comprehensive test cases"""
    print("Starting evaluation with test cases...")
    
    transcriber = WhisperTranscriber()
    analyzer = SentimentAnalyzer()
    
    # Comprehensive test cases
    test_cases = [
        {
            "text": "I am really happy about this amazing result!",
            "ground_truth_emotions": ["joy"],
            "category": "Single Emotion - Positive"
        },
        {
            "text": "This makes me so angry and frustrated.",
            "ground_truth_emotions": ["anger", "disgust"],
            "category": "Multiple Emotions - Negative"
        },
        {
            "text": "I'm feeling quite sad and disappointed today.",
            "ground_truth_emotions": ["sadness"],
            "category": "Single Emotion - Negative"
        },
        {
            "text": "Wow, this is such a surprising and exciting development!",
            "ground_truth_emotions": ["surprise", "joy"],
            "category": "Multiple Emotions - Positive"
        },
        {
            "text": "I'm not sure how to feel about this situation.",
            "ground_truth_emotions": ["neutral"],
            "category": "Neutral"
        },
        {
            "text": "The presentation was both impressive and nerve-wracking.",
            "ground_truth_emotions": ["joy", "fear"],
            "category": "Mixed Emotions"
        }
    ]
    
    print("\nEvaluation Results:")
    print("-" * 50)
    
    # Store predictions and ground truth per test case
    test_predictions = []
    test_ground_truth = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i} ({test['category']}):")
        print(f"Text: {test['text']}")
        
        # Get emotion predictions
        emotions = analyzer.analyze_text(test['text'])
        predictions = [emotion["label"].lower() for emotion in emotions if emotion["score"] > 0.2]  # Only keep significant predictions
        
        # Store complete set for this test case
        test_predictions.append(predictions)
        test_ground_truth.append(test["ground_truth_emotions"])
        
        print(f"Ground Truth Emotions: {test['ground_truth_emotions']}")
        print(f"Predicted Emotions: {predictions}")
        
        # Print individual prediction scores
        print("\nDetailed Emotion Scores:")
        print("-" * 40)
        for emotion in emotions:
            print(f"{emotion['label'].capitalize()}: {emotion['score']:.1%}")
        print("-" * 40)
    
    # Convert to multi-label format
    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(test_ground_truth)
    y_pred = mlb.transform(test_predictions)
    
    # Calculate overall metrics
    print("\nOverall Metrics:")
    print("-" * 50)
    
    # Calculate metrics using binary format
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print("\nEmotion Detection Metrics:")
    print("-" * 40)
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    print("-" * 40)
    
    print("\nEmotion Labels:", mlb.classes_)

def main():
    print("Starting Evaluation...")
    try:
        evaluate_with_test_cases()
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 