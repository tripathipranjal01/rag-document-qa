# RAG (Retrieval-Augmented Generation) service

import numpy as np
from typing import List, Dict, Any
from openai import OpenAI
import tiktoken

class RAGService:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            raise Exception(f"Error getting embeddings: {str(e)}")
    
    def search_similar_chunks(self, query: str, chunks: List[Dict], top_k: int = 8) -> List[Dict]:
        # Get query embedding
        query_embedding = self.get_embeddings([query])[0]
        
        if not chunks:
            return []
        
        # Calculate similarities
        similarities = []
        for chunk in chunks:
            similarity = np.dot(query_embedding, chunk["embedding"])
            similarities.append((similarity, chunk))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in similarities[:top_k]]
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        if not context_chunks:
            return "I don't have enough information to answer your question. Please upload some documents first."
        
        context = "\n\n".join([chunk["content"] for chunk in context_chunks])
        
        prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context. 
        If the answer cannot be found in the context, say so. Be concise and accurate.

        Context:
        {context}

        Question: {query}

        Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating answer: {str(e)}")
