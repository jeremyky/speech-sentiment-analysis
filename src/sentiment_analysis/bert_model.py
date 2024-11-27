from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

class SentimentAnalyzer:
    def __init__(self):
        # Use a simpler sentiment analysis model that's publicly available
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
        except OSError:
            print("First, you need to login to Hugging Face:")
            print("1. Run: pip install --upgrade huggingface_hub")
            print("2. Run: huggingface-cli login")
            print("3. Enter your access token from https://huggingface.co/settings/tokens")
            raise
        
    def analyze_text(self, text):
        """Analyze sentiment of given text"""
        try:
            result = self.sentiment_pipeline(text)
            return result[0]  # Returns {'label': 'POSITIVE/NEGATIVE', 'score': 0.xxx}
        except Exception as e:
            print(f"Error analyzing text: {e}")
            return {'label': 'ERROR', 'score': 0.0}

    def fine_tune(self, train_texts, train_labels):
        """Fine-tune the model on GoEmotions dataset"""
        # Implementation for fine-tuning will go here
        pass