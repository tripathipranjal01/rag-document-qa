# Analytics service for tracking performance metrics

import time
from typing import Dict, List, Any
from datetime import datetime

class AnalyticsService:
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "avg_query_time": 0,
            "processing_times": [],
            "P50": 0,
            "P95": 0,
            "P99": 0
        }
    
    def record_query(self, processing_time: float):
        self.metrics["total_queries"] += 1
        self.metrics["processing_times"].append(processing_time)
        self.metrics["avg_query_time"] = sum(self.metrics["processing_times"]) / len(self.metrics["processing_times"])
        
        # Calculate percentiles
        if len(self.metrics["processing_times"]) > 0:
            sorted_times = sorted(self.metrics["processing_times"])
            self.metrics["P50"] = sorted_times[len(sorted_times) // 2]
            self.metrics["P95"] = sorted_times[int(len(sorted_times) * 0.95)]
            self.metrics["P99"] = sorted_times[int(len(sorted_times) * 0.99)]
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.copy()
