import hashlib
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import OrderedDict
import threading
import json

logger = logging.getLogger(__name__)

class PerformanceService:
    """Advanced performance optimization service with caching and cost control"""
    
    def __init__(self, max_cache_size: int = 10000, max_query_cache_size: int = 1000):
        self.max_cache_size = max_cache_size
        self.max_query_cache_size = max_query_cache_size
        
        # Embedding cache with LRU eviction
        self.embedding_cache = OrderedDict()
        self.embedding_cache_hits = 0
        self.embedding_cache_misses = 0
        
        # Query result cache
        self.query_cache = OrderedDict()
        self.query_cache_hits = 0
        self.query_cache_misses = 0
        
        # Batch processing queue
        self.batch_queue = []
        self.batch_size = 10
        self.batch_timeout = 5.0  # seconds
        self.last_batch_time = time.time()
        
        # Cost tracking
        self.cost_metrics = {
            "total_tokens": 0,
            "total_embeddings": 0,
            "total_queries": 0,
            "estimated_cost": 0.0,
            "cost_per_query": 0.0
        }
        
        # Performance metrics
        self.performance_metrics = {
            "avg_embedding_time": 0.0,
            "avg_query_time": 0.0,
            "cache_hit_rate": 0.0,
            "throughput": 0.0
        }
        
        # Thread safety
        self.cache_lock = threading.Lock()
        self.batch_lock = threading.Lock()
        
        # Start background batch processor
        self.batch_processor_thread = threading.Thread(target=self._batch_processor, daemon=True)
        self.batch_processor_thread.start()
    
    def get_embedding_with_cache(self, text: str, embedding_func) -> Optional[List[float]]:
        """Get embedding with intelligent caching"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        with self.cache_lock:
            # Check cache first
            if text_hash in self.embedding_cache:
                self.embedding_cache_hits += 1
                # Move to end (LRU)
                embedding = self.embedding_cache.pop(text_hash)
                self.embedding_cache[text_hash] = embedding
                logger.debug(f"Embedding cache hit for text: {text[:50]}...")
                return embedding
            
            self.embedding_cache_misses += 1
        
        # Generate new embedding
        start_time = time.time()
        try:
            embedding = embedding_func(text)
            generation_time = time.time() - start_time
            
            # Update performance metrics
            self._update_embedding_metrics(generation_time)
            
            # Cache the embedding
            with self.cache_lock:
                self._add_to_cache(text_hash, embedding, self.embedding_cache, self.max_cache_size)
            
            logger.debug(f"Embedding generated and cached for text: {text[:50]}... (time: {generation_time:.3f}s)")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def get_query_result_with_cache(self, query: str, doc_id: Optional[str], 
                                   scope: str, query_func) -> Optional[Dict[str, Any]]:
        """Get query result with intelligent caching"""
        # Create cache key
        cache_key = hashlib.md5(f"{query}_{doc_id}_{scope}".encode()).hexdigest()
        
        with self.cache_lock:
            # Check cache first
            if cache_key in self.query_cache:
                self.query_cache_hits += 1
                # Move to end (LRU)
                result = self.query_cache.pop(cache_key)
                self.query_cache[cache_key] = result
                result["cached"] = True
                logger.debug(f"Query cache hit for: {query[:50]}...")
                return result
            
            self.query_cache_misses += 1
        
        # Execute query
        start_time = time.time()
        try:
            result = query_func(query, doc_id, scope)
            query_time = time.time() - start_time
            
            # Update performance metrics
            self._update_query_metrics(query_time)
            
            # Cache the result
            with self.cache_lock:
                result["cached"] = False
                self._add_to_cache(cache_key, result, self.query_cache, self.max_query_cache_size)
            
            logger.debug(f"Query executed and cached: {query[:50]}... (time: {query_time:.3f}s)")
            return result
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return None
    
    def add_to_batch_queue(self, text: str, callback_func) -> str:
        """Add text to batch processing queue"""
        batch_id = str(hashlib.md5(f"{text}_{time.time()}".encode()).hexdigest())
        
        with self.batch_lock:
            self.batch_queue.append({
                "id": batch_id,
                "text": text,
                "callback": callback_func,
                "timestamp": time.time()
            })
        
        logger.debug(f"Added to batch queue: {text[:50]}... (ID: {batch_id})")
        return batch_id
    
    def _batch_processor(self):
        """Background batch processor"""
        while True:
            try:
                current_time = time.time()
                
                with self.batch_lock:
                    # Check if we should process batch
                    should_process = (
                        len(self.batch_queue) >= self.batch_size or
                        (len(self.batch_queue) > 0 and 
                         current_time - self.last_batch_time >= self.batch_timeout)
                    )
                    
                    if should_process:
                        batch_items = self.batch_queue[:self.batch_size]
                        self.batch_queue = self.batch_queue[self.batch_size:]
                        self.last_batch_time = current_time
                    else:
                        batch_items = []
                
                if batch_items:
                    self._process_batch(batch_items)
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                time.sleep(1)
    
    def _process_batch(self, batch_items: List[Dict[str, Any]]):
        """Process a batch of items"""
        if not batch_items:
            return
        
        logger.info(f"Processing batch of {len(batch_items)} items")
        
        # Extract texts for batch processing
        texts = [item["text"] for item in batch_items]
        
        try:
            # Here you would implement batch embedding generation
            # For now, we'll process them individually
            for item in batch_items:
                try:
                    # Call the callback function
                    result = item["callback"](item["text"])
                    logger.debug(f"Batch item processed: {item['id']}")
                except Exception as e:
                    logger.error(f"Error processing batch item {item['id']}: {e}")
        
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
    
    def track_cost(self, tokens: int, embeddings: int = 0, queries: int = 0):
        """Track API costs"""
        # OpenAI pricing (approximate)
        embedding_cost_per_1k = 0.00002  # $0.00002 per 1K tokens
        gpt_cost_per_1k = 0.00015       # $0.00015 per 1K tokens
        
        self.cost_metrics["total_tokens"] += tokens
        self.cost_metrics["total_embeddings"] += embeddings
        self.cost_metrics["total_queries"] += queries
        
        # Calculate estimated cost
        embedding_cost = (self.cost_metrics["total_tokens"] / 1000) * embedding_cost_per_1k
        query_cost = (tokens / 1000) * gpt_cost_per_1k
        
        self.cost_metrics["estimated_cost"] = embedding_cost + query_cost
        
        if self.cost_metrics["total_queries"] > 0:
            self.cost_metrics["cost_per_query"] = self.cost_metrics["estimated_cost"] / self.cost_metrics["total_queries"]
    
    def _update_embedding_metrics(self, generation_time: float):
        """Update embedding performance metrics"""
        # Simple moving average
        current_avg = self.performance_metrics["avg_embedding_time"]
        if current_avg == 0:
            self.performance_metrics["avg_embedding_time"] = generation_time
        else:
            self.performance_metrics["avg_embedding_time"] = (current_avg * 0.9) + (generation_time * 0.1)
    
    def _update_query_metrics(self, query_time: float):
        """Update query performance metrics"""
        # Simple moving average
        current_avg = self.performance_metrics["avg_query_time"]
        if current_avg == 0:
            self.performance_metrics["avg_query_time"] = query_time
        else:
            self.performance_metrics["avg_query_time"] = (current_avg * 0.9) + (query_time * 0.1)
    
    def _add_to_cache(self, key: str, value: Any, cache: OrderedDict, max_size: int):
        """Add item to cache with LRU eviction"""
        if key in cache:
            # Move to end (LRU)
            cache.pop(key)
        
        cache[key] = value
        
        # Evict oldest items if cache is full
        while len(cache) > max_size:
            cache.popitem(last=False)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        total_embedding_requests = self.embedding_cache_hits + self.embedding_cache_misses
        total_query_requests = self.query_cache_hits + self.query_cache_misses
        
        embedding_hit_rate = (self.embedding_cache_hits / total_embedding_requests 
                             if total_embedding_requests > 0 else 0)
        query_hit_rate = (self.query_cache_hits / total_query_requests 
                         if total_query_requests > 0 else 0)
        
        return {
            "cache_performance": {
                "embedding_cache_size": len(self.embedding_cache),
                "query_cache_size": len(self.query_cache),
                "embedding_hit_rate": round(embedding_hit_rate * 100, 2),
                "query_hit_rate": round(query_hit_rate * 100, 2),
                "total_embedding_requests": total_embedding_requests,
                "total_query_requests": total_query_requests
            },
            "performance_metrics": {
                "avg_embedding_time": round(self.performance_metrics["avg_embedding_time"], 3),
                "avg_query_time": round(self.performance_metrics["avg_query_time"], 3),
                "batch_queue_size": len(self.batch_queue)
            },
            "cost_metrics": {
                "total_tokens": self.cost_metrics["total_tokens"],
                "total_embeddings": self.cost_metrics["total_embeddings"],
                "total_queries": self.cost_metrics["total_queries"],
                "estimated_cost": round(self.cost_metrics["estimated_cost"], 4),
                "cost_per_query": round(self.cost_metrics["cost_per_query"], 4)
            },
            "optimization_suggestions": self._get_optimization_suggestions()
        }
    
    def _get_optimization_suggestions(self) -> List[str]:
        """Get optimization suggestions based on current metrics"""
        suggestions = []
        
        # Cache hit rate suggestions
        total_embedding_requests = self.embedding_cache_hits + self.embedding_cache_misses
        if total_embedding_requests > 0:
            embedding_hit_rate = self.embedding_cache_hits / total_embedding_requests
            if embedding_hit_rate < 0.3:
                suggestions.append("Consider increasing embedding cache size for better hit rate")
            elif embedding_hit_rate > 0.8:
                suggestions.append("High cache hit rate - consider reducing cache size to save memory")
        
        # Cost optimization suggestions
        if self.cost_metrics["cost_per_query"] > 0.01:
            suggestions.append("High cost per query - consider optimizing chunk sizes or using cheaper models")
        
        # Performance suggestions
        if self.performance_metrics["avg_query_time"] > 5.0:
            suggestions.append("Slow query performance - consider implementing query result caching")
        
        if len(self.batch_queue) > 50:
            suggestions.append("Large batch queue - consider increasing batch processing frequency")
        
        return suggestions
    
    def clear_cache(self, cache_type: str = "all"):
        """Clear specified cache"""
        with self.cache_lock:
            if cache_type in ["all", "embedding"]:
                self.embedding_cache.clear()
                self.embedding_cache_hits = 0
                self.embedding_cache_misses = 0
            
            if cache_type in ["all", "query"]:
                self.query_cache.clear()
                self.query_cache_hits = 0
                self.query_cache_misses = 0
        
        logger.info(f"Cache cleared: {cache_type}")
    
    def export_metrics(self, filepath: str):
        """Export performance metrics to JSON file"""
        metrics_data = {
            "timestamp": time.time(),
            "performance_summary": self.get_performance_summary(),
            "cache_state": {
                "embedding_cache_keys": list(self.embedding_cache.keys())[:100],  # First 100 keys
                "query_cache_keys": list(self.query_cache.keys())[:100]
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        logger.info(f"Performance metrics exported to {filepath}")
    
    def reset_metrics(self):
        """Reset all performance metrics (for testing)"""
        self.embedding_cache_hits = 0
        self.embedding_cache_misses = 0
        self.query_cache_hits = 0
        self.query_cache_misses = 0
        self.cost_metrics = {
            "total_tokens": 0,
            "total_embeddings": 0,
            "total_queries": 0,
            "estimated_cost": 0.0,
            "cost_per_query": 0.0
        }
        self.performance_metrics = {
            "avg_embedding_time": 0.0,
            "avg_query_time": 0.0,
            "cache_hit_rate": 0.0,
            "throughput": 0.0
        }
        
        logger.info("Performance metrics reset")
