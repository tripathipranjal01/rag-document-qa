import time
import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

logger = logging.getLogger(__name__)

class AnalyticsService:
    """Advanced analytics and monitoring service"""
    
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "total_documents": 0,
            "total_chunks": 0,
            "query_times": [],
            "popular_queries": Counter(),
            "document_types": Counter(),
            "error_counts": Counter(),
            "session_activity": defaultdict(int),
            "api_usage": {
                "embeddings_generated": 0,
                "tokens_processed": 0,
                "files_processed": 0,
                "cache_hits": 0,
                "cache_misses": 0
            },
            "performance": {
                "avg_response_time": 0,
                "p95_response_time": 0,
                "p99_response_time": 0,
                "throughput": 0
            },
            "user_engagement": {
                "active_sessions": 0,
                "documents_per_session": defaultdict(int),
                "queries_per_session": defaultdict(int)
            }
        }
        
        self.start_time = time.time()
        self.request_times = []
    
    def track_query(self, query: str, processing_time: float, session_id: str):
        """Track a query with performance metrics"""
        self.metrics["total_queries"] += 1
        self.metrics["query_times"].append(processing_time)
        self.metrics["popular_queries"][query.lower()[:50]] += 1
        self.metrics["session_activity"][session_id] += 1
        self.metrics["user_engagement"]["queries_per_session"][session_id] += 1
        
        # Keep only last 1000 query times for performance calculation
        if len(self.metrics["query_times"]) > 1000:
            self.metrics["query_times"] = self.metrics["query_times"][-1000:]
        
        # Update performance metrics
        self._update_performance_metrics()
        
        logger.info(f"Query tracked: {query[:50]}... (time: {processing_time:.2f}s)")
    
    def track_document_upload(self, file_type: str, chunk_count: int, session_id: str):
        """Track document upload"""
        self.metrics["total_documents"] += 1
        self.metrics["total_chunks"] += chunk_count
        self.metrics["document_types"][file_type] += 1
        self.metrics["api_usage"]["files_processed"] += 1
        self.metrics["user_engagement"]["documents_per_session"][session_id] += 1
        
        logger.info(f"Document uploaded: {file_type} with {chunk_count} chunks")
    
    def track_embedding_generation(self, token_count: int, cache_hit: bool = False):
        """Track embedding generation"""
        self.metrics["api_usage"]["embeddings_generated"] += 1
        self.metrics["api_usage"]["tokens_processed"] += token_count
        
        if cache_hit:
            self.metrics["api_usage"]["cache_hits"] += 1
        else:
            self.metrics["api_usage"]["cache_misses"] += 1
    
    def track_error(self, error_type: str, error_message: str):
        """Track errors"""
        self.metrics["error_counts"][error_type] += 1
        logger.error(f"Error tracked: {error_type} - {error_message}")
    
    def track_session_activity(self, session_id: str):
        """Track session activity"""
        self.metrics["session_activity"][session_id] += 1
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        if self.metrics["query_times"]:
            times = sorted(self.metrics["query_times"])
            self.metrics["performance"]["avg_response_time"] = sum(times) / len(times)
            self.metrics["performance"]["p95_response_time"] = times[int(len(times) * 0.95)]
            self.metrics["performance"]["p99_response_time"] = times[int(len(times) * 0.99)]
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        # Calculate cache hit rate
        total_cache_requests = (self.metrics["api_usage"]["cache_hits"] + 
                              self.metrics["api_usage"]["cache_misses"])
        cache_hit_rate = (self.metrics["api_usage"]["cache_hits"] / total_cache_requests 
                         if total_cache_requests > 0 else 0)
        
        # Calculate API efficiency
        api_efficiency = (self.metrics["api_usage"]["tokens_processed"] / 
                         max(self.metrics["api_usage"]["embeddings_generated"], 1))
        
        # Calculate active sessions (sessions with activity in last hour)
        active_sessions = len([s for s, count in self.metrics["session_activity"].items() 
                             if count > 0])
        
        return {
            "system_overview": {
                "total_queries": self.metrics["total_queries"],
                "total_documents": self.metrics["total_documents"],
                "total_chunks": self.metrics["total_chunks"],
                "active_sessions": active_sessions,
                "uptime_hours": (time.time() - self.start_time) / 3600
            },
            "performance_metrics": {
                "avg_response_time": round(self.metrics["performance"]["avg_response_time"], 2),
                "p95_response_time": round(self.metrics["performance"]["p95_response_time"], 2),
                "p99_response_time": round(self.metrics["performance"]["p99_response_time"], 2),
                "cache_hit_rate": round(cache_hit_rate * 100, 1),
                "api_efficiency": round(api_efficiency, 2)
            },
            "usage_patterns": {
                "popular_queries": dict(self.metrics["popular_queries"].most_common(5)),
                "document_types": dict(self.metrics["document_types"]),
                "top_active_sessions": dict(Counter(self.metrics["session_activity"]).most_common(5))
            },
            "api_usage": {
                "embeddings_generated": self.metrics["api_usage"]["embeddings_generated"],
                "tokens_processed": self.metrics["api_usage"]["tokens_processed"],
                "files_processed": self.metrics["api_usage"]["files_processed"],
                "cache_hits": self.metrics["api_usage"]["cache_hits"],
                "cache_misses": self.metrics["api_usage"]["cache_misses"]
            },
            "error_summary": dict(self.metrics["error_counts"]),
            "user_engagement": {
                "avg_documents_per_session": sum(self.metrics["user_engagement"]["documents_per_session"].values()) / max(len(self.metrics["user_engagement"]["documents_per_session"]), 1),
                "avg_queries_per_session": sum(self.metrics["user_engagement"]["queries_per_session"].values()) / max(len(self.metrics["user_engagement"]["queries_per_session"]), 1)
            }
        }
    
    def get_user_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get analytics for a specific user session"""
        return {
            "session_stats": {
                "documents_uploaded": self.metrics["user_engagement"]["documents_per_session"].get(session_id, 0),
                "queries_made": self.metrics["user_engagement"]["queries_per_session"].get(session_id, 0),
                "total_activity": self.metrics["session_activity"].get(session_id, 0)
            },
            "performance": {
                "avg_response_time": round(self.metrics["performance"]["avg_response_time"], 2),
                "cache_hit_rate": round((self.metrics["api_usage"]["cache_hits"] / 
                                       max(self.metrics["api_usage"]["cache_hits"] + self.metrics["api_usage"]["cache_misses"], 1)) * 100, 1)
            }
        }
    
    def export_analytics(self, filepath: str):
        """Export analytics to JSON file"""
        analytics_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "summary": self.get_analytics_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(analytics_data, f, indent=2, default=str)
        
        logger.info(f"Analytics exported to {filepath}")
    
    def reset_analytics(self):
        """Reset all analytics (for testing)"""
        self.__init__()
        logger.info("Analytics reset")
