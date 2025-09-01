# Performance monitoring service

import time
from typing import Dict, Any
from datetime import datetime

class PerformanceService:
    def __init__(self):
        self.metrics = {
            "response_times": [],
            "error_count": 0,
            "success_count": 0,
            "start_time": datetime.now()
        }
    
    def record_request(self, response_time: float, success: bool = True):
        self.metrics["response_times"].append(response_time)
        if success:
            self.metrics["success_count"] += 1
        else:
            self.metrics["error_count"] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        response_times = self.metrics["response_times"]
        if not response_times:
            return {
                "avg_response_time": 0,
                "min_response_time": 0,
                "max_response_time": 0,
                "p95_response_time": 0,
                "total_requests": 0,
                "success_rate": 0
            }
        
        sorted_times = sorted(response_times)
        total_requests = len(response_times)
        success_rate = self.metrics["success_count"] / total_requests if total_requests > 0 else 0
        
        return {
            "avg_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "p95_response_time": sorted_times[int(len(sorted_times) * 0.95)],
            "total_requests": total_requests,
            "success_rate": success_rate
        }
