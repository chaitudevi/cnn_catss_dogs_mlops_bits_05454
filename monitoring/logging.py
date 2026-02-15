import logging
import json
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        
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
