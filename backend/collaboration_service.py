# Collaboration service for multi-user features

from typing import Dict, List, Any
from datetime import datetime
import uuid

class CollaborationService:
    def __init__(self):
        self.shared_documents = {}
        self.user_permissions = {}
    
    def share_document(self, owner_session: str, doc_id: str, user_sessions: List[str], permissions: str = "read"):
        if doc_id not in self.shared_documents:
            self.shared_documents[doc_id] = {
                "owner": owner_session,
                "shared_with": [],
                "permissions": {}
            }
        
        for user_session in user_sessions:
            if user_session not in self.shared_documents[doc_id]["shared_with"]:
                self.shared_documents[doc_id]["shared_with"].append(user_session)
                self.shared_documents[doc_id]["permissions"][user_session] = permissions
    
    def get_shared_documents(self, session_id: str) -> List[Dict[str, Any]]:
        shared_docs = []
        for doc_id, doc_info in self.shared_documents.items():
            if session_id in doc_info["shared_with"]:
                shared_docs.append({
                    "doc_id": doc_id,
                    "owner": doc_info["owner"],
                    "permissions": doc_info["permissions"].get(session_id, "read")
                })
        return shared_docs
