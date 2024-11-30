import nltk
from src.sentiment_analysis.bert_model import SentimentAnalyzer

def analyze_hierarchical_text(text, analyzer):
    """Analyze emotions at multiple text levels: overall, paragraph, sentence"""
    
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    
    # Initialize results structure
    results = {
        "overall": {},
        "paragraphs": [],
        "sentences": []
    }
    
    # Overall analysis
    overall_emotions = analyzer.analyze_text(text)
    results["overall"] = {
        "text": text,
        "emotions": overall_emotions,
        "dominant_emotion": overall_emotions[0] if overall_emotions else None
    }
    
    # Paragraph analysis
    for i, para in enumerate(paragraphs):
        if not para.strip():
            continue
            
        para_emotions = analyzer.analyze_text(para)
        results["paragraphs"].append({
            "index": i,
            "text": para,
            "emotions": para_emotions,
            "dominant_emotion": para_emotions[0] if para_emotions else None
        })
        
    # Sentence analysis
    nltk.download('punkt', quiet=True)
    sentences = nltk.sent_tokenize(text)
    
    for i, sentence in enumerate(sentences):
        sent_emotions = analyzer.analyze_text(sentence)
        results["sentences"].append({
            "index": i,
            "text": sentence,
            "emotions": sent_emotions,
            "dominant_emotion": sent_emotions[0] if sent_emotions else None
        })
    
    return results 