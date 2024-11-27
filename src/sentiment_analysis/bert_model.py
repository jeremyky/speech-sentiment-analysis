from transformers import pipeline
import torch
from src.utils.config_manager import ConfigManager

class SentimentAnalyzer:
    def __init__(self):
        self.config = ConfigManager()
        self.emotion_config = self.config.get_emotion_config()
        
        # Determine device (GPU if available, otherwise CPU)
        self.device = 0 if torch.cuda.is_available() else -1
        
        try:
            self.sentiment_pipeline = pipeline(
                "text-classification",
                model=self.config.get_model_path(),
                top_k=self.emotion_config['model_config']['top_k'],
                device=self.device  # Use GPU if available
            )
            print(f"Using {'GPU' if self.device == 0 else 'CPU'} for sentiment analysis")
        except OSError:
            print("First, you need to login to Hugging Face:")
            print("1. Run: pip install --upgrade huggingface_hub")
            print("2. Run: huggingface-cli login")
            print("3. Enter your access token from https://huggingface.co/settings/tokens")
            raise

        # Create emotion color mapping from config
        self.emotion_colors = {
            emotion['label']: emotion['color'] 
            for emotion in self.emotion_config['emotions']
        }

    def analyze_text(self, text):
        """Analyze emotions in text"""
        try:
            if not text or text.isspace():
                return [{'label': 'Waiting...', 'score': 0.0, 'color': '#95a5a6'}]

            results = self.sentiment_pipeline(text)
            
            emotions = []
            for result in results[0]:
                emotion = {
                    'label': result['label'],
                    'score': result['score'],
                    'color': self.emotion_colors.get(result['label'], '#95a5a6')
                }
                emotions.append(emotion)

            return emotions
            
        except Exception as e:
            print(f"Error in analyze_text: {e}")
            import traceback
            traceback.print_exc()
            return [{'label': 'ERROR', 'score': 0.0, 'color': '#95a5a6'}]

    def fine_tune(self, train_texts, train_labels):
        """Fine-tune the model on GoEmotions dataset"""
        # Implementation for fine-tuning will go here
        pass