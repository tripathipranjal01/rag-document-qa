# Database service for data persistence

import pickle
import os
from typing import Dict, Any

class DatabaseService:
    def __init__(self, filename: str = "data_simple.pkl"):
        self.filename = filename
        self.data = {
            "documents": {},
            "chunks": {},
            "sessions": {},
            "analytics": {},
            "chat_history": {},
            "document_progress": {}
        }
        self.load_data()
    
    def save_data(self):
        with open(self.filename, "wb") as f:
            pickle.dump(self.data, f)
    
    def load_data(self):
        try:
            if os.path.exists(self.filename):
                with open(self.filename, "rb") as f:
                    loaded_data = pickle.load(f)
                    self.data.update(loaded_data)
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def get(self, key: str) -> Any:
        return self.data.get(key, {})
    
    def set(self, key: str, value: Any):
        self.data[key] = value
        self.save_data()
    
    def update(self, key: str, value: Any):
        if key not in self.data:
            self.data[key] = {}
        self.data[key].update(value)
        self.save_data()
