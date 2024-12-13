import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
from transformers import pipeline
import whisper
import torch

def load_test_samples():
    # Simulated test data
    return [
        {
            'text': "I'm really happy about this result!",
            'ground_truth_emotion': 'joy',
            'audio_path': None  # We'll simulate audio for now
        },
        {
            'text': "This makes me so angry and frustrated.",
            'ground_truth_emotion': 'anger',
            'audio_path': None
        },
        {
            'text': "I feel quite sad about what happened.",
            'ground_truth_emotion': 'sadness',
            'audio_path': None
        },
        {
            'text': "That's really impressive work!",
            'ground_truth_emotion': 'joy',
            'audio_path': None
        },
        {
            'text': "I'm worried about the upcoming deadline.",
            'ground_truth_emotion': 'fear',
            'audio_path': None
        }
    ]

def evaluate_sentiment_model():
    # Initialize sentiment model
    model_name = "bhadresh-savani/bert-base-go-emotion"
    sentiment_pipeline = pipeline("text-classification", model=model_name, top_k=None)
    
    # Get test samples
    test_samples = load_test_samples()
    
    results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'latency': []
    }
    
    predictions = []
    ground_truth = []
    
    print("Evaluating sentiment analysis model...")
    
    for sample in test_samples:
        start_time = time.time()
        
        # Get prediction
        prediction = sentiment_pipeline(sample['text'])
        latency = time.time() - start_time
        
        # Get highest scoring emotion
        pred_emotion = max(prediction[0], key=lambda x: x['score'])['label']
        
        predictions.append(pred_emotion)
        ground_truth.append(sample['ground_truth_emotion'])
        results['latency'].append(latency)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(ground_truth, predictions),
        'precision': precision_score(ground_truth, predictions, average='weighted', zero_division=0),
        'recall': recall_score(ground_truth, predictions, average='weighted', zero_division=0),
        'f1': f1_score(ground_truth, predictions, average='weighted', zero_division=0),
        'avg_latency': np.mean(results['latency'])
    }
    
    return metrics

def generate_latex_tables(metrics):
    sentiment_table = f"""
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{|l|c|}}
\\hline
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\hline
Accuracy & {metrics['accuracy']:.2%} \\\\
Precision & {metrics['precision']:.2%} \\\\
Recall & {metrics['recall']:.2%} \\\\
F1 Score & {metrics['f1']:.2%} \\\\
Average Latency & {metrics['avg_latency']:.3f}s \\\\
\\hline
\\end{{tabular}}
\\caption{{Sentiment Analysis Performance}}
\\label{{tab:sentiment_analysis}}
\\end{{table}}
"""
    return sentiment_table

def main():
    print("Starting evaluation...")
    
    # Evaluate sentiment analysis
    sentiment_metrics = evaluate_sentiment_model()
    
    # Generate LaTeX tables
    latex_table = generate_latex_tables(sentiment_metrics)
    
    # Save tables to file
    with open('evaluation_tables.tex', 'w') as f:
        f.write(latex_table)
    
    # Print results to console
    print("\nResults:")
    print("-" * 40)
    print(f"Accuracy: {sentiment_metrics['accuracy']:.2%}")
    print(f"Precision: {sentiment_metrics['precision']:.2%}")
    print(f"Recall: {sentiment_metrics['recall']:.2%}")
    print(f"F1 Score: {sentiment_metrics['f1']:.2%}")
    print(f"Average Latency: {sentiment_metrics['avg_latency']:.3f}s")
    print("\nLaTeX tables have been saved to evaluation_tables.tex")

if __name__ == "__main__":
    main()