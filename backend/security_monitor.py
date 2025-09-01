#!/usr/bin/env python3
"""
Security and Cost Monitoring Script for RAG Document Q&A System

This script provides comprehensive monitoring of:
- Security status and vulnerabilities
- Cost tracking and optimization
- Performance metrics
- Compliance checks
"""

import os
import json
import time
import hashlib
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityMonitor:
    def __init__(self, api_base_url: str = "http://localhost:8001"):
        self.api_base_url = api_base_url
        self.security_checks = []
        self.cost_alerts = []
        self.performance_alerts = []
    
    def run_security_audit(self) -> Dict[str, Any]:
        """Run comprehensive security audit"""
        logger.info("Starting security audit...")
        
        audit_results = {
            "timestamp": datetime.now().isoformat(),
            "security_score": 0,
            "issues_found": [],
            "recommendations": [],
            "checks_passed": 0,
            "total_checks": 0
        }
        
        # Check API endpoints
        self._check_api_security(audit_results)
        
        # Check file upload security
        self._check_file_upload_security(audit_results)
        
        # Check session security
        self._check_session_security(audit_results)
        
        # Check environment security
        self._check_environment_security(audit_results)
        
        # Calculate security score
        audit_results["security_score"] = (audit_results["checks_passed"] / audit_results["total_checks"]) * 100
        
        return audit_results
    
    def _check_api_security(self, audit_results: Dict[str, Any]):
        """Check API security endpoints"""
        try:
            response = requests.get(f"{self.api_base_url}/api/security/status")
            if response.status_code == 200:
                security_status = response.json()
                audit_results["total_checks"] += len(security_status["security_features"])
                
                for feature, enabled in security_status["security_features"].items():
                    if enabled:
                        audit_results["checks_passed"] += 1
                    else:
                        audit_results["issues_found"].append(f"Security feature disabled: {feature}")
                        audit_results["recommendations"].append(f"Enable {feature}")
                
                logger.info(f"API security check completed: {audit_results['checks_passed']}/{audit_results['total_checks']} passed")
            else:
                audit_results["issues_found"].append("Security status endpoint not accessible")
                audit_results["recommendations"].append("Ensure security endpoint is properly configured")
        except Exception as e:
            audit_results["issues_found"].append(f"API security check failed: {str(e)}")
            audit_results["recommendations"].append("Check API connectivity and configuration")
    
    def _check_file_upload_security(self, audit_results: Dict[str, Any]):
        """Check file upload security measures"""
        audit_results["total_checks"] += 4
        
        # Check file size limits
        try:
            response = requests.get(f"{self.api_base_url}/api/security/status")
            if response.status_code == 200:
                security_status = response.json()
                max_size = security_status.get("max_file_size_mb", 0)
                if max_size <= 30:
                    audit_results["checks_passed"] += 1
                else:
                    audit_results["issues_found"].append(f"File size limit too high: {max_size}MB")
                    audit_results["recommendations"].append("Reduce file size limit to 30MB or less")
                
                # Check allowed file types
                allowed_types = security_status.get("allowed_file_types", [])
                dangerous_types = ['.exe', '.bat', '.sh', '.php', '.js', '.html']
                if not any(dangerous_type in allowed_types for dangerous_type in dangerous_types):
                    audit_results["checks_passed"] += 1
                else:
                    audit_results["issues_found"].append("Dangerous file types allowed")
                    audit_results["recommendations"].append("Remove dangerous file types from allowlist")
                
                # Check content validation
                if security_status["security_features"].get("content_validation", False):
                    audit_results["checks_passed"] += 1
                else:
                    audit_results["issues_found"].append("Content validation not enabled")
                    audit_results["recommendations"].append("Enable content validation for uploaded files")
                
                # Check filename sanitization
                if security_status["security_features"].get("filename_sanitization", False):
                    audit_results["checks_passed"] += 1
                else:
                    audit_results["issues_found"].append("Filename sanitization not enabled")
                    audit_results["recommendations"].append("Enable filename sanitization")
        except Exception as e:
            audit_results["issues_found"].append(f"File upload security check failed: {str(e)}")
    
    def _check_session_security(self, audit_results: Dict[str, Any]):
        """Check session security measures"""
        audit_results["total_checks"] += 2
        
        try:
            response = requests.get(f"{self.api_base_url}/api/security/status")
            if response.status_code == 200:
                security_status = response.json()
                
                # Check session timeout
                timeout_hours = security_status.get("session_timeout_hours", 0)
                if timeout_hours <= 24:
                    audit_results["checks_passed"] += 1
                else:
                    audit_results["issues_found"].append(f"Session timeout too long: {timeout_hours} hours")
                    audit_results["recommendations"].append("Reduce session timeout to 24 hours or less")
                
                # Check session isolation
                if security_status["security_features"].get("session_isolation", False):
                    audit_results["checks_passed"] += 1
                else:
                    audit_results["issues_found"].append("Session isolation not enabled")
                    audit_results["recommendations"].append("Enable session isolation")
        except Exception as e:
            audit_results["issues_found"].append(f"Session security check failed: {str(e)}")
    
    def _check_environment_security(self, audit_results: Dict[str, Any]):
        """Check environment security"""
        audit_results["total_checks"] += 3
        
        # Check for hardcoded API keys
        backend_files = ["main_simple.py", "main_advanced.py", "main_enhanced.py"]
        for file in backend_files:
            if os.path.exists(file):
                with open(file, 'r') as f:
                    content = f.read()
                    if "sk-" in content and "your_openai_api_key_here" not in content:
                        audit_results["issues_found"].append(f"Hardcoded API key found in {file}")
                        audit_results["recommendations"].append("Move API key to environment variables")
                    else:
                        audit_results["checks_passed"] += 1
        
        # Check .env file permissions
        if os.path.exists(".env"):
            stat = os.stat(".env")
            if stat.st_mode & 0o777 == 0o600:
                audit_results["checks_passed"] += 1
            else:
                audit_results["issues_found"].append(".env file has incorrect permissions")
                audit_results["recommendations"].append("Set .env file permissions to 600")
        else:
            audit_results["issues_found"].append(".env file not found")
            audit_results["recommendations"].append("Create .env file for environment variables")
    
    def run_cost_analysis(self) -> Dict[str, Any]:
        """Run comprehensive cost analysis"""
        logger.info("Starting cost analysis...")
        
        try:
            response = requests.get(f"{self.api_base_url}/api/cost/estimate")
            if response.status_code == 200:
                cost_data = response.json()
                
                analysis = {
                    "timestamp": datetime.now().isoformat(),
                    "current_costs": cost_data["cost_breakdown"],
                    "monthly_estimate": cost_data["estimates"]["monthly_estimate"],
                    "cost_per_query": cost_data["estimates"]["cost_per_query"],
                    "optimization_score": 0,
                    "alerts": [],
                    "recommendations": []
                }
                
                # Check for cost optimization opportunities
                cache_hit_rate = float(cost_data["cost_optimization"]["cache_hit_rate"].rstrip('%'))
                if cache_hit_rate < 50:
                    analysis["alerts"].append(f"Low cache hit rate: {cache_hit_rate}%")
                    analysis["recommendations"].append("Increase cache size and optimize cache strategy")
                
                tokens_per_query = cost_data["cost_optimization"]["tokens_per_query"]
                if tokens_per_query > 2000:
                    analysis["alerts"].append(f"High token usage per query: {tokens_per_query}")
                    analysis["recommendations"].append("Optimize chunk sizes and reduce context length")
                
                monthly_estimate = cost_data["estimates"]["monthly_estimate"]
                if monthly_estimate > 100:
                    analysis["alerts"].append(f"High monthly cost estimate: ${monthly_estimate}")
                    analysis["recommendations"].append("Implement rate limiting and cost controls")
                
                # Calculate optimization score
                optimization_factors = [
                    100 if cache_hit_rate > 70 else cache_hit_rate,
                    100 if tokens_per_query < 1500 else max(0, 100 - (tokens_per_query - 1500) / 10),
                    100 if monthly_estimate < 50 else max(0, 100 - (monthly_estimate - 50) / 2)
                ]
                analysis["optimization_score"] = sum(optimization_factors) / len(optimization_factors)
                
                return analysis
            else:
                return {"error": "Failed to fetch cost data"}
        except Exception as e:
            return {"error": f"Cost analysis failed: {str(e)}"}
    
    def run_performance_audit(self) -> Dict[str, Any]:
        """Run performance audit"""
        logger.info("Starting performance audit...")
        
        try:
            response = requests.get(f"{self.api_base_url}/api/health")
            if response.status_code == 200:
                health_data = response.json()
                
                audit = {
                    "timestamp": datetime.now().isoformat(),
                    "performance_score": 0,
                    "alerts": [],
                    "recommendations": []
                }
                
                # Check response times
                response_times = health_data.get("performance", {}).get("response_times", {})
                p95_time = response_times.get("p95", 0)
                p99_time = response_times.get("p99", 0)
                
                if p95_time > 8:
                    audit["alerts"].append(f"P95 response time too high: {p95_time}s")
                    audit["recommendations"].append("Optimize query processing and increase caching")
                
                if p99_time > 10:
                    audit["alerts"].append(f"P99 response time too high: {p99_time}s")
                    audit["recommendations"].append("Implement performance monitoring and optimization")
                
                # Check cache efficiency
                cache_size = health_data.get("performance", {}).get("embedding_cache_size", 0)
                if cache_size < 100:
                    audit["alerts"].append(f"Low cache size: {cache_size}")
                    audit["recommendations"].append("Increase cache size for better performance")
                
                # Calculate performance score
                performance_factors = [
                    100 if p95_time <= 8 else max(0, 100 - (p95_time - 8) * 10),
                    100 if p99_time <= 10 else max(0, 100 - (p99_time - 10) * 5),
                    100 if cache_size >= 100 else cache_size
                ]
                audit["performance_score"] = sum(performance_factors) / len(performance_factors)
                
                return audit
            else:
                return {"error": "Failed to fetch health data"}
        except Exception as e:
            return {"error": f"Performance audit failed: {str(e)}"}
    
    def generate_report(self) -> str:
        """Generate comprehensive monitoring report"""
        logger.info("Generating comprehensive report...")
        
        security_audit = self.run_security_audit()
        cost_analysis = self.run_cost_analysis()
        performance_audit = self.run_performance_audit()
        
        report = f"""
# RAG Document Q&A System - Security & Cost Monitoring Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Security Audit
- **Security Score**: {security_audit.get('security_score', 0):.1f}%
- **Checks Passed**: {security_audit.get('checks_passed', 0)}/{security_audit.get('total_checks', 0)}
- **Issues Found**: {len(security_audit.get('issues_found', []))}

### Security Issues:
{chr(10).join(f"- {issue}" for issue in security_audit.get('issues_found', []))}

### Security Recommendations:
{chr(10).join(f"- {rec}" for rec in security_audit.get('recommendations', []))}

## Cost Analysis
- **Monthly Estimate**: ${cost_analysis.get('monthly_estimate', 0):.2f}
- **Cost per Query**: ${cost_analysis.get('cost_per_query', 0):.4f}
- **Optimization Score**: {cost_analysis.get('optimization_score', 0):.1f}%

### Cost Alerts:
{chr(10).join(f"- {alert}" for alert in cost_analysis.get('alerts', []))}

### Cost Recommendations:
{chr(10).join(f"- {rec}" for rec in cost_analysis.get('recommendations', []))}

## Performance Audit
- **Performance Score**: {performance_audit.get('performance_score', 0):.1f}%

### Performance Alerts:
{chr(10).join(f"- {alert}" for alert in performance_audit.get('alerts', []))}

### Performance Recommendations:
{chr(10).join(f"- {rec}" for rec in performance_audit.get('recommendations', []))}

## Overall Assessment
- **Security**: {'✅ Good' if security_audit.get('security_score', 0) >= 80 else '⚠️ Needs Attention' if security_audit.get('security_score', 0) >= 60 else '❌ Critical Issues'}
- **Cost**: {'✅ Optimized' if cost_analysis.get('optimization_score', 0) >= 80 else '⚠️ High Costs' if cost_analysis.get('optimization_score', 0) >= 60 else '❌ Cost Issues'}
- **Performance**: {'✅ Good' if performance_audit.get('performance_score', 0) >= 80 else '⚠️ Needs Optimization' if performance_audit.get('performance_score', 0) >= 60 else '❌ Performance Issues'}
"""
        
        return report

def main():
    """Main function to run monitoring"""
    monitor = SecurityMonitor()
    
    print("🔍 RAG Document Q&A System - Security & Cost Monitor")
    print("=" * 60)
    
    # Generate and display report
    report = monitor.generate_report()
    print(report)
    
    # Save report to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"monitoring_report_{timestamp}.txt"
    
    with open(report_filename, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Report saved to: {report_filename}")

if __name__ == "__main__":
    main()
