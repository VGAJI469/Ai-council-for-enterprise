"""FastAPI application entry point."""
from fastapi import FastAPI
from api.routes.council_routes import router

app = FastAPI(
    title="Adaptive AI Enterprise Council API",
    description="Multi-agent financial risk governance system",
    version="1.0.0"
)
app.include_router(router)

@app.get("/health")
async def health(): return {"status": "healthy", "system": "Adaptive AI Enterprise Council v1.0"}
