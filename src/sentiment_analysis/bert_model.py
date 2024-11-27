from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

class SentimentAnalyzer:
    def __init__(self):
        # Use a model trained for emotion detection
        try:
            self.sentiment_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=3  # Return top 3 emotions
            )
        except OSError:
            print("First, you need to login to Hugging Face:")
            print("1. Run: pip install --upgrade huggingface_hub")
            print("2. Run: huggingface-cli login")
            print("3. Enter your access token from https://huggingface.co/settings/tokens")
            raise
        
        # Emotion color mapping
        self.emotion_colors = {
            'joy': '#2ecc71',      # Green
            'surprise': '#f1c40f',  # Yellow
            'neutral': '#95a5a6',   # Gray
            'sadness': '#3498db',   # Blue
            'anger': '#e74c3c',     # Red
            'fear': '#9b59b6',      # Purple
            'disgust': '#1abc9c'    # Turquoise
        }
        
    def analyze_text(self, text):
        """Analyze emotions in text"""
        try:
            if not text or text.isspace():
                return [{'label': 'Waiting...', 'score': 0.0, 'color': '#95a5a6'}]
            
            print(f"Analyzing text: {text}")  # Debug print
            results = self.sentiment_pipeline(text)
            print(f"Raw results: {results}")  # Debug print
            
            # Format results for display
            emotions = []
            for result in results[0]:  # Note: results[0] because pipeline returns a list of dictionaries
                emotion = {
                    'label': result['label'],
                    'score': result['score'],
                    'color': self.emotion_colors.get(result['label'], '#95a5a6')
                }
                emotions.append(emotion)
                print(f"Processed emotion: {emotion}")  # Debug print
            
            return emotions
        except Exception as e:
            print(f"Error in analyze_text: {e}")  # Debug print
            import traceback
            traceback.print_exc()  # Print full error traceback
            return [{'label': 'ERROR', 'score': 0.0, 'color': '#95a5a6'}]

    def fine_tune(self, train_texts, train_labels):
        """Fine-tune the model on GoEmotions dataset"""
        # Implementation for fine-tuning will go here
        pass