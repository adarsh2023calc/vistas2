# db.py
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime
from textblob import TextBlob



def store_feedback(feedback_type: str, feedback_text: str, code: str, error: str, output: str) -> bool:
        """Store user feedback with analysis in MongoDB.
        
        Args:
            feedback_type: Either 'positive' or 'negative'
            feedback_text: Optional detailed feedback from user
            code: The code that was debugged
            error: The error message
            output: The AI's debugging suggestion
            
        Returns:
            bool: True if feedback was stored successfully
        """
        try:
            
            
            # Perform sentiment analysis
            blob = TextBlob(feedback_text)
            sentiment_score = blob.sentiment.polarity
            
            # Extract error type
            error_type = error.split(':')[0] if ':' in error else error
            
            # Create feedback document with analysis
            feedback_doc = {
                'feedback_type': feedback_type,
                'feedback_text': feedback_text,
                'code': code,
                'error': error,
                'output': output,
                'timestamp': datetime.utcnow(),
                'analysis': {
                    'sentiment_score': sentiment_score,
                    'code_length': len(code),
                    'error_type': error_type,
                    'key_terms': [word for word in blob.noun_phrases],
                    'complexity_metrics': {
                        'lines_of_code': len(code.splitlines()),
                        'has_error_handling': 'try' in code.lower() and 'except' in code.lower(),
                        'has_comments': '#' in code or '"""' in code
                    }
                }
            }
            
            # Store in MongoDB
            feedback_collection.insert_one(feedback_doc)
            # Create indexes for analysis fields if they don't exist
            feedback_collection.create_index([('analysis.sentiment_score', 1)])
            feedback_collection.create_index([('analysis.error_type', 1)])
            
            return True
        except Exception as e:
            print(f"Failed to store feedback: {e}")
            return False
            

load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

client = MongoClient(mongo_uri)

# Use these names ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
db = client["vistas_login"]  
users_collection = db["users"]  
print("Mongo URI:", mongo_uri)
  

try:
    client = MongoClient(mongo_uri)
    # Test the connection
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
    
    db = client["vistas_login"]  
    users_collection = db["users"]
    feedback_collection = db["feedback"]
    feedback_collection.create_index([('timestamp', -1)])
    feedback_collection.create_index([('feedback_type', 1)])
    
    
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    raise


