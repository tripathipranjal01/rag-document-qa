import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class CollaborationService:
    """Advanced collaboration and sharing service"""
    
    def __init__(self):
        self.shared_documents = {}  # {doc_id: {shared_by: session_id, shared_with: [session_ids], permissions: "read"|"write"}}
        self.document_comments = {}  # {doc_id: [{session_id, comment, timestamp, comment_id}]}
        self.document_versions = {}  # {doc_id: [{version_id, session_id, timestamp, changes}]}
        self.collaboration_sessions = {}  # {session_id: {team_id, role, permissions}}
        self.document_activity = {}  # {doc_id: [{session_id, action, timestamp}]}
    
    def share_document(self, doc_id: str, owner_session: str, target_session: str, 
                      permissions: str = "read") -> Dict[str, Any]:
        """Share document with another session"""
        if permissions not in ["read", "write", "admin"]:
            raise ValueError("Invalid permissions. Must be 'read', 'write', or 'admin'")
        
        # Initialize sharing record if not exists
        if doc_id not in self.shared_documents:
            self.shared_documents[doc_id] = {
                "shared_by": owner_session,
                "shared_with": [],
                "permissions": {},
                "shared_at": datetime.now().isoformat()
            }
        
        # Add target session to shared list
        if target_session not in self.shared_documents[doc_id]["shared_with"]:
            self.shared_documents[doc_id]["shared_with"].append(target_session)
        
        # Set permissions
        self.shared_documents[doc_id]["permissions"][target_session] = permissions
        
        # Track activity
        self._track_activity(doc_id, owner_session, "shared_document", {
            "target_session": target_session,
            "permissions": permissions
        })
        
        logger.info(f"Document {doc_id} shared with session {target_session} ({permissions} permissions)")
        
        return {
            "success": True,
            "message": f"Document shared with session {target_session}",
            "permissions": permissions
        }
    
    def unshare_document(self, doc_id: str, owner_session: str, target_session: str) -> Dict[str, Any]:
        """Remove sharing for a document"""
        if doc_id not in self.shared_documents:
            raise ValueError("Document is not shared")
        
        if self.shared_documents[doc_id]["shared_by"] != owner_session:
            raise ValueError("Only the document owner can unshare")
        
        if target_session in self.shared_documents[doc_id]["shared_with"]:
            self.shared_documents[doc_id]["shared_with"].remove(target_session)
            if target_session in self.shared_documents[doc_id]["permissions"]:
                del self.shared_documents[doc_id]["permissions"][target_session]
        
        # Track activity
        self._track_activity(doc_id, owner_session, "unshared_document", {
            "target_session": target_session
        })
        
        logger.info(f"Document {doc_id} unshared from session {target_session}")
        
        return {
            "success": True,
            "message": f"Document unshared from session {target_session}"
        }
    
    def add_comment(self, doc_id: str, session_id: str, comment_text: str, 
                   parent_comment_id: Optional[str] = None) -> Dict[str, Any]:
        """Add comment to document"""
        if not comment_text.strip():
            raise ValueError("Comment text cannot be empty")
        
        # Initialize comments if not exists
        if doc_id not in self.document_comments:
            self.document_comments[doc_id] = []
        
        comment_id = str(uuid.uuid4())
        comment = {
            "comment_id": comment_id,
            "session_id": session_id,
            "comment": comment_text,
            "timestamp": datetime.now().isoformat(),
            "parent_comment_id": parent_comment_id,
            "replies": [],
            "likes": 0,
            "edited": False
        }
        
        # Add to parent comment if it's a reply
        if parent_comment_id:
            for existing_comment in self.document_comments[doc_id]:
                if existing_comment["comment_id"] == parent_comment_id:
                    existing_comment["replies"].append(comment_id)
                    break
        
        self.document_comments[doc_id].append(comment)
        
        # Track activity
        self._track_activity(doc_id, session_id, "added_comment", {
            "comment_id": comment_id,
            "parent_comment_id": parent_comment_id
        })
        
        logger.info(f"Comment added to document {doc_id} by session {session_id}")
        
        return {
            "success": True,
            "comment_id": comment_id,
            "message": "Comment added successfully"
        }
    
    def edit_comment(self, doc_id: str, comment_id: str, session_id: str, 
                    new_text: str) -> Dict[str, Any]:
        """Edit an existing comment"""
        if doc_id not in self.document_comments:
            raise ValueError("Document has no comments")
        
        for comment in self.document_comments[doc_id]:
            if comment["comment_id"] == comment_id and comment["session_id"] == session_id:
                comment["comment"] = new_text
                comment["edited"] = True
                comment["edited_at"] = datetime.now().isoformat()
                
                # Track activity
                self._track_activity(doc_id, session_id, "edited_comment", {
                    "comment_id": comment_id
                })
                
                logger.info(f"Comment {comment_id} edited by session {session_id}")
                
                return {
                    "success": True,
                    "message": "Comment edited successfully"
                }
        
        raise ValueError("Comment not found or you don't have permission to edit it")
    
    def delete_comment(self, doc_id: str, comment_id: str, session_id: str) -> Dict[str, Any]:
        """Delete a comment"""
        if doc_id not in self.document_comments:
            raise ValueError("Document has no comments")
        
        for i, comment in enumerate(self.document_comments[doc_id]):
            if comment["comment_id"] == comment_id and comment["session_id"] == session_id:
                # Remove comment
                deleted_comment = self.document_comments[doc_id].pop(i)
                
                # Remove from parent comment replies
                if deleted_comment.get("parent_comment_id"):
                    for parent_comment in self.document_comments[doc_id]:
                        if parent_comment["comment_id"] == deleted_comment["parent_comment_id"]:
                            if comment_id in parent_comment["replies"]:
                                parent_comment["replies"].remove(comment_id)
                            break
                
                # Track activity
                self._track_activity(doc_id, session_id, "deleted_comment", {
                    "comment_id": comment_id
                })
                
                logger.info(f"Comment {comment_id} deleted by session {session_id}")
                
                return {
                    "success": True,
                    "message": "Comment deleted successfully"
                }
        
        raise ValueError("Comment not found or you don't have permission to delete it")
    
    def get_comments(self, doc_id: str, session_id: str) -> List[Dict[str, Any]]:
        """Get all comments for a document"""
        if doc_id not in self.document_comments:
            return []
        
        # Check if user has access to this document
        has_access = self._check_document_access(doc_id, session_id)
        if not has_access:
            raise ValueError("Access denied to document")
        
        return self.document_comments[doc_id]
    
    def get_shared_documents(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all documents shared with a session"""
        shared_docs = []
        
        for doc_id, share_info in self.shared_documents.items():
            if session_id in share_info["shared_with"]:
                shared_docs.append({
                    "doc_id": doc_id,
                    "shared_by": share_info["shared_by"],
                    "permissions": share_info["permissions"].get(session_id, "read"),
                    "shared_at": share_info["shared_at"]
                })
        
        return shared_docs
    
    def get_document_activity(self, doc_id: str, session_id: str) -> List[Dict[str, Any]]:
        """Get activity history for a document"""
        if doc_id not in self.document_activity:
            return []
        
        # Check if user has access to this document
        has_access = self._check_document_access(doc_id, session_id)
        if not has_access:
            raise ValueError("Access denied to document")
        
        return self.document_activity[doc_id]
    
    def create_collaboration_session(self, session_id: str, team_name: str, 
                                   role: str = "member") -> Dict[str, Any]:
        """Create a collaboration session/team"""
        team_id = str(uuid.uuid4())
        
        self.collaboration_sessions[session_id] = {
            "team_id": team_id,
            "team_name": team_name,
            "role": role,
            "created_at": datetime.now().isoformat(),
            "members": [session_id]
        }
        
        logger.info(f"Collaboration session created: {team_name} (ID: {team_id})")
        
        return {
            "success": True,
            "team_id": team_id,
            "team_name": team_name,
            "message": "Collaboration session created successfully"
        }
    
    def _check_document_access(self, doc_id: str, session_id: str) -> bool:
        """Check if session has access to document"""
        # Check if user owns the document (this would be checked in main service)
        # For now, just check if document is shared with this session
        if doc_id in self.shared_documents:
            return session_id in self.shared_documents[doc_id]["shared_with"]
        return False
    
    def _track_activity(self, doc_id: str, session_id: str, action: str, 
                       details: Dict[str, Any] = None):
        """Track document activity"""
        if doc_id not in self.document_activity:
            self.document_activity[doc_id] = []
        
        activity = {
            "session_id": session_id,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        self.document_activity[doc_id].append(activity)
        
        # Keep only last 100 activities per document
        if len(self.document_activity[doc_id]) > 100:
            self.document_activity[doc_id] = self.document_activity[doc_id][-100:]
    
    def get_collaboration_summary(self, session_id: str) -> Dict[str, Any]:
        """Get collaboration summary for a session"""
        shared_docs = self.get_shared_documents(session_id)
        team_info = self.collaboration_sessions.get(session_id, {})
        
        return {
            "shared_documents_count": len(shared_docs),
            "team_info": team_info,
            "recent_activity": self._get_recent_activity(session_id)
        }
    
    def _get_recent_activity(self, session_id: str) -> List[Dict[str, Any]]:
        """Get recent activity for a session"""
        recent_activity = []
        
        for doc_id, activities in self.document_activity.items():
            for activity in activities:
                if activity["session_id"] == session_id:
                    recent_activity.append({
                        "doc_id": doc_id,
                        "action": activity["action"],
                        "timestamp": activity["timestamp"]
                    })
        
        # Sort by timestamp and return last 10
        recent_activity.sort(key=lambda x: x["timestamp"], reverse=True)
        return recent_activity[:10]
