from speech_recognition.whisper_client import WhisperTranscriber
from sentiment_analysis.bert_model import SentimentAnalyzer

def main():
    # Initialize components
    transcriber = WhisperTranscriber()
    sentiment_analyzer = SentimentAnalyzer()

    while True:
        try:
            # Record and transcribe audio
            print("Listening for speech...")
            transcription = transcriber.transcribe_realtime()
            print(f"Transcribed: {transcription}")

            # Analyze sentiment
            sentiment = sentiment_analyzer.analyze_text(transcription)
            print(f"Sentiment: {sentiment['label']} (confidence: {sentiment['score']:.2f})")

        except KeyboardInterrupt:
            print("\nStopping the application...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main() 