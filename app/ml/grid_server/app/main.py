''' main.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: FastAPI entrypoint for the
GRID lip reading application.
@date: 10 May 2026
@returns: FastAPI Application

'''


# Imports
from contextlib import (
    asynccontextmanager,
)

from fastapi import (
    FastAPI,
)

from app.grpc.server import (
    start_grpc_server,
)

from app.api.routes.health import (
    router as health_router,
)

from app.api.routes.predict import (
    router as predict_router,
)

from app.api.routes.root import (
    router as root_router,
)

from app.utils.model_loader import (
    load_model,
)



# Application Lifespan
@asynccontextmanager
async def lifespan(
    app: FastAPI
):

    '''
    @description: Handles application
    startup and shutdown events.

    @args:
        app:
            FastAPI application instance.

    '''

    global grpc_server

    # Warmup TensorFlow Model
    load_model()

    # Start gRPC Server
    grpc_server = (
        start_grpc_server()
    )

    yield

    # Shutdown gRPC Server
    if grpc_server:
        grpc_server.stop(
            0,
        )



# FastAPI Application
app: FastAPI = FastAPI(
    title="GRID Lip Reading API",
    description=(
        "Lip reading inference API "
        "using TensorFlow and MediaPipe."
    ),
    version="1.0.0",
    lifespan=lifespan
)



# Register Routes
app.include_router(
    health_router
)

app.include_router(
    predict_router
)


app.include_router(
    root_router
)
