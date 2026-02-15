import logging
import json
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# Global stats
request_counts = {"total": 0}
latency_stats = {"total_seconds": 0.0, "count": 0}

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        
        # Update stats
        request_counts["total"] += 1
        status_key = str(response.status_code)
        request_counts[status_key] = request_counts.get(status_key, 0) + 1
        
        latency_stats["total_seconds"] += process_time
        latency_stats["count"] += 1
        
        log_dict = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency": f"{process_time:.4f}s"
        }
        
        # Avoid logging massive base64 strings or binary data
        logging.info(json.dumps(log_dict))
        
        return response

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("api.log"),
            logging.StreamHandler()
        ]
    )
