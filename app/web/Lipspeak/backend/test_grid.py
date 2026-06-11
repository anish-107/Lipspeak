from app.grpc.clients.grid_client import GridClient

print(
    GridClient.predict(
        b"hello"
    )
)