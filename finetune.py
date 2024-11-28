from src.sentiment_analysis.bert_model import SentimentAnalyzer

def main():
    print("Starting fine-tuning process...")
    analyzer = SentimentAnalyzer()
    
    # Fine-tune the model
    analyzer.fine_tune(output_dir="models/fine_tuned_emotion")
    
    print("Fine-tuning complete!")
    
    # Test the fine-tuned model
    test_texts = [
        "I'm so happy today!",
        "This makes me really angry",
        "I'm feeling quite sad",
        "What a wonderful surprise!",
        "I'm really proud of what we achieved"
    ]
    
    print("\nTesting fine-tuned model:")
    print("-" * 40)
    for text in test_texts:
        print(f"\nText: {text}")
        emotions = analyzer.analyze_text(text)
        for emotion in emotions:
            print(f"{emotion['label'].capitalize()}: {emotion['score']:.1%}")
    
if __name__ == "__main__":
    main() 