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

    def chunk_text(self, text, max_length=500):
        """Split text into chunks that won't exceed model's max length"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def analyze_text(self, text):
        """Analyze emotions in text"""
        try:
            if not text or text.isspace():
                return [{'label': 'Waiting...', 'score': 0.0, 'color': '#95a5a6'}]

            # Split text into manageable chunks
            chunks = self.chunk_text(text)
            
            # Analyze each chunk
            all_results = []
            for chunk in chunks:
                chunk_results = self.sentiment_pipeline(chunk)
                all_results.extend(chunk_results)
            
            # Aggregate results
            emotion_scores = {}
            for result in all_results:
                for emotion in result:
                    label = emotion['label']
                    score = emotion['score']
                    if label in emotion_scores:
                        emotion_scores[label] = max(emotion_scores[label], score)
                    else:
                        emotion_scores[label] = score
            
            # Convert to list and sort by score
            emotions = [
                {
                    'label': label,
                    'score': score,
                    'color': self.emotion_colors.get(label, '#95a5a6')
                }
                for label, score in emotion_scores.items()
            ]
            emotions.sort(key=lambda x: x['score'], reverse=True)
            
            # Return top k emotions based on config
            top_k = self.emotion_config['model_config']['top_k']
            return emotions[:top_k]
            
        except Exception as e:
            print(f"Error in analyze_text: {e}")
            import traceback
            traceback.print_exc()
            return [{'label': 'ERROR', 'score': 0.0, 'color': '#95a5a6'}]

    def fine_tune(self, train_texts, train_labels):
        """Fine-tune the model on GoEmotions dataset"""
        # Implementation for fine-tuning will go here
        pass