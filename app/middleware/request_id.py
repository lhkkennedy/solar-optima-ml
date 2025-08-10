import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        # Attach to state for downstream usage
        request.state.request_id = request_id
        try:
            response: Response = await call_next(request)
        except Exception as exc:
            logger.exception(f"Unhandled error. request_id={request_id} path={request.url.path}")
            raise
        response.headers["X-Request-Id"] = request_id
        return response


def add_request_id_middleware(app):
    """Helper to register the RequestIdMiddleware on a FastAPI app"""
    app.add_middleware(RequestIdMiddleware)