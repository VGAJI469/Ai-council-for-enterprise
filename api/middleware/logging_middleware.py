"""FastAPI middleware for structured request/response logging."""
import time
import logging
from fastapi import Request
from fastapi.responses import Response
from utils.logger import get_logger

logger = get_logger("api.middleware")


async def logging_middleware(request: Request, call_next):
    start = time.time()
    try:
        logger.info("request_start", extra={"extra": {"method": request.method, "path": request.url.path}})
        response: Response = await call_next(request)
        duration = time.time() - start
        logger.info("request_end", extra={"extra": {"method": request.method, "path": request.url.path, "status_code": response.status_code, "duration": round(duration, 4)}})
        return response
    except Exception as e:
        duration = time.time() - start
        logger.exception("request_error", extra={"extra": {"method": request.method, "path": request.url.path, "duration": round(duration, 4)}})
        raise
