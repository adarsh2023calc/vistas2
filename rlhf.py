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
    if feedback_data.get('is_pairwise', False):
        # Process pairwise preference data for DPO training
        preferred = feedback_data.get('preferred_response')
        non_preferred = feedback_data.get('non_preferred_response')
        if preferred and non_preferred:
            # Store successful patterns for DPO training
            feedback_analytics['dpo_pairs'].append({
                'preferred': preferred,
                'non_preferred': non_preferred,
                'context': feedback_data.get('context_info', {}),
                'metrics': feedback_data.get('comparison_metrics', {})
            })
    
    # Update general feedback metrics
    if feedback_data['sentiment'] > 0.5:
        feedback_analytics['success_patterns'][feedback_data['error_type']] += 1
    else:
        feedback_analytics['common_issues'][feedback_data['error_type']] += 1
    
    feedback_analytics['sentiment_scores'].append(feedback_data['sentiment'])



