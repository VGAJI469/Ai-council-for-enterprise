"""FastAPI application entry point."""
from fastapi import FastAPI
from api.routes.council_routes import router
from utils.logger import setup_logging, get_logger
from api.middleware.logging_middleware import logging_middleware

# Initialize structured logging
setup_logging()

logger = get_logger(__name__)

app = FastAPI(
    title="Adaptive AI Enterprise Council API",
    description="Multi-agent financial risk governance system",
    version="1.0.0"
)
app.middleware("http")(logging_middleware)
app.include_router(router)

@app.get("/health")
async def health(): return {"status": "healthy", "system": "Adaptive AI Enterprise Council v1.0"}
