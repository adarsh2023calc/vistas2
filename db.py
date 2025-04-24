# db.py
from pymongo import MongoClient
from dotenv import load_dotenv
import os


load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

client = MongoClient(mongo_uri)

# Use these names ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
db = client["vistas_login"]  
users_collection = db["users"]  
print("Mongo URI:", mongo_uri)
  



load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

try:
    client = MongoClient(mongo_uri)
    # Test the connection
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
    
    db = client["vistas_login"]  
    users_collection = db["users"]
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    raise
