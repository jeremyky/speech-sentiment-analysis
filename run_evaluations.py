from evaluate import (
    evaluate_speech_recognition,
    evaluate_sentiment_analysis,
    evaluate_sentiment_edge_cases,
    evaluate_end_to_end
)

def run_all_evaluations():
    print("=" * 80)
    print("RUNNING COMPREHENSIVE EVALUATION SUITE")
    print("=" * 80)
    
    # 1. Speech Recognition Tests
    print("\n\n" + "=" * 30)
    print("SPEECH RECOGNITION EVALUATION")
    print("=" * 30)
    evaluate_speech_recognition()
    
    # 2. Sentiment Analysis Tests
    print("\n\n" + "=" * 30)
    print("SENTIMENT ANALYSIS EVALUATION")
    print("=" * 30)
    evaluate_sentiment_analysis()
    
    # 3. Edge Cases
    print("\n\n" + "=" * 30)
    print("EDGE CASES EVALUATION")
    print("=" * 30)
    evaluate_sentiment_edge_cases()
    
    # 4. End-to-End Pipeline
    print("\n\n" + "=" * 30)
    print("END-TO-END PIPELINE EVALUATION")
    print("=" * 30)
    evaluate_end_to_end()

if __name__ == "__main__":
    run_all_evaluations() 