import openai
import tiktoken
import numpy as np
from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime
import uuid

class RAGService:
    def __init__(self):
        self.openai_client = openai
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.chunks = []
        self.documents = {}
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """Split text into chunks with overlap"""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI"""
        try:
            response = self.openai_client.Embedding.create(
                model="text-embedding-3-small",
                input=text
            )
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def process_document(self, doc_id: str, doc_name: str, content: str) -> Dict[str, Any]:
        """Process document: chunk, embed, and store with metadata"""
        try:
            # Create document record
            doc_record = {
                "id": doc_id,
                "name": doc_name,
                "content": content,
                "created_at": datetime.now().isoformat(),
                "status": "processing"
            }
            
            # Chunk the content
            chunks = self.chunk_text(content)
            
            # Process each chunk
            chunk_records = []
            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                
                # Get embedding
                embedding = self.get_embedding(chunk)
                
                # Create chunk record with metadata
                chunk_record = {
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "content": chunk,
                    "embedding": embedding,
                    "chunk_index": i,
                    "page": 1,  # For now, assume single page
                    "created_at": datetime.now().isoformat()
                }
                
                chunk_records.append(chunk_record)
                self.chunks.append(chunk_record)
            
            # Update document status
            doc_record["status"] = "indexed"
            doc_record["chunk_count"] = len(chunks)
            self.documents[doc_id] = doc_record
            
            return {
                "success": True,
                "doc_id": doc_id,
                "chunks_created": len(chunks),
                "status": "indexed"
            }
            
        except Exception as e:
            if doc_id in self.documents:
                self.documents[doc_id]["status"] = "failed"
                self.documents[doc_id]["error"] = str(e)
            
            return {
                "success": False,
                "error": str(e),
                "status": "failed"
            }
    
    def search(self, query: str, doc_id: Optional[str] = None, top_k: int = 8) -> List[Dict[str, Any]]:
        """Search for relevant chunks based on query"""
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []
            
            # Filter chunks by document if specified
            candidate_chunks = self.chunks
            if doc_id:
                candidate_chunks = [chunk for chunk in self.chunks if chunk["doc_id"] == doc_id]
            
            # Calculate similarities
            similarities = []
            for chunk in candidate_chunks:
                if chunk.get("embedding"):
                    similarity = self.cosine_similarity(query_embedding, chunk["embedding"])
                    similarities.append((similarity, chunk))
            
            # Sort by similarity and return top-k
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_results = similarities[:top_k]
            
            # Format results
            results = []
            for similarity, chunk in top_results:
                results.append({
                    "chunk_id": chunk["id"],
                    "doc_id": chunk["doc_id"],
                    "doc_name": chunk["doc_name"],
                    "content": chunk["content"],
                    "page": chunk["page"],
                    "similarity": similarity,
                    "chunk_index": chunk["chunk_index"]
                })
            
            return results
            
        except Exception as e:
            print(f"Error in search: {e}")
            return []
    
    def get_document_status(self, doc_id: str) -> Dict[str, Any]:
        """Get document processing status"""
        if doc_id in self.documents:
            return self.documents[doc_id]
        return {"status": "not_found"}
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents with their status"""
        return list(self.documents.values())
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document and all its chunks"""
        try:
            # Remove chunks
            self.chunks = [chunk for chunk in self.chunks if chunk["doc_id"] != doc_id]
            
            # Remove document
            if doc_id in self.documents:
                del self.documents[doc_id]
            
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
