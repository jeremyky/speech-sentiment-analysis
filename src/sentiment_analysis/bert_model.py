from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    pipeline
)
import torch
from src.utils.config_manager import ConfigManager
import datasets
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class SentimentAnalyzer:
    def __init__(self):
        self.config = ConfigManager()
        self.emotion_config = self.config.get_emotion_config()
        
        # Add debug prints
        print(f"Using emotion mode: {self.config.main_config['model']['emotion_mode']}")
        print(f"Available emotions: {[e['label'] for e in self.emotion_config['emotions']]}")
        print(f"Top k emotions to show: {self.emotion_config['model_config']['top_k']}")
        
        # Determine device (GPU if available, otherwise CPU)
        self.device = 0 if torch.cuda.is_available() else -1
        
        try:
            model_path = self.config.get_model_path()
            print(f"Loading model from: {model_path}")  # Debug print
            
            self.sentiment_pipeline = pipeline(
                "text-classification",
                model=model_path,
                top_k=self.emotion_config['model_config']['top_k'],
                device=self.device
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
            
            # Get top k emotions based on config
            top_k = self.emotion_config['model_config']['top_k']
            top_emotions = emotions[:top_k]
            
            # Print top emotions to command line
            print("\nTop emotions detected:")
            print("-" * 40)
            for emotion in top_emotions:
                print(f"{emotion['label'].capitalize()}: {emotion['score']:.1%}")
            print("-" * 40)
            
            return top_emotions
            
        except Exception as e:
            print(f"Error in analyze_text: {e}")
            import traceback
            traceback.print_exc()
            return [{'label': 'ERROR', 'score': 0.0, 'color': '#95a5a6'}]

    def fine_tune(self, output_dir="fine_tuned_model"):
        """Fine-tune the model on GoEmotions dataset"""
        try:
            # Load GoEmotions dataset
            dataset = datasets.load_dataset("go_emotions", "raw")
            
            # Get list of emotions from config
            emotions = [e['label'] for e in self.emotion_config['emotions']]
            
            def preprocess_data(examples):
                # Convert multi-label format to our emotion categories
                labels = [0] * len(emotions)
                for emotion_idx in examples['labels']:
                    if emotion_idx < len(emotions):
                        labels[emotion_idx] = 1
                return {
                    'text': examples['text'],
                    'labels': labels
                }
            
            # Preprocess dataset
            tokenized_dataset = dataset.map(
                preprocess_data,
                remove_columns=dataset['train'].column_names
            )
            
            # Define training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=3,
                weight_decay=0.01,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
            )
            
            def compute_metrics(pred):
                labels = pred.label_ids
                preds = pred.predictions > 0.5
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels, preds, average='weighted'
                )
                acc = accuracy_score(labels, preds)
                return {
                    'accuracy': acc,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall
                }
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset['train'],
                eval_dataset=tokenized_dataset['validation'],
                compute_metrics=compute_metrics,
            )
            
            # Fine-tune the model
            trainer.train()
            
            # Save the fine-tuned model
            trainer.save_model(output_dir)
            
            # Update the pipeline with fine-tuned model
            self.sentiment_pipeline = pipeline(
                "text-classification",
                model=output_dir,
                top_k=self.emotion_config['model_config']['top_k'],
                device=self.device
            )
            
            print("Fine-tuning complete! Model saved to:", output_dir)
            
        except Exception as e:
            print(f"Error during fine-tuning: {e}")
            import traceback
            traceback.print_exc()