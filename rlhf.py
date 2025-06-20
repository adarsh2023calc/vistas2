from textblob import TextBlob

from collections import defaultdict

# Analytics storage
feedback_analytics = {
    'sentiment_scores': [],
    'common_issues': defaultdict(int),
    'success_patterns': defaultdict(int)
}

def analyze_feedback(feedback_text: str, code: str, error: str) -> dict:
    # Sentiment analysis
    blob = TextBlob(feedback_text)
    sentiment_score = blob.sentiment.polarity
    print(sentiment_score)
    # Extract common patterns/issues
    analysis = {
        'sentiment': sentiment_score,
        'code_length': len(code),
        'error_type': error.split(':')[0] if ':' in error else error,
        'key_terms': [word for word in blob.noun_phrases]
    }
    return analysis

def update_model_weights(feedback_data: dict):
    # Placeholder for model weight updates based on feedback
    if feedback_data['sentiment'] > 0.5:
        # Store successful patterns for future reference
        feedback_analytics['success_patterns'][feedback_data['error_type']] += 1
    else:
        # Store issues for improvement
        feedback_analytics['common_issues'][feedback_data['error_type']] += 1
    
    feedback_analytics['sentiment_scores'].append(feedback_data['sentiment'])


