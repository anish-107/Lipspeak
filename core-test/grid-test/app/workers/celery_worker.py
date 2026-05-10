''' celery_worker.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Celery worker configuration
for asynchronous GRID lip reading tasks.
@date: 10 May 2026
@returns: Celery Application

'''


# Imports
from celery import (
    Celery
)



# Celery Application
celery_app: Celery = Celery(
    "lipspeak_worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)



# Celery Configuration
celery_app.conf.update(

    task_serializer="json",

    accept_content=[
        "json"
    ],

    result_serializer="json",

    timezone="UTC",

    enable_utc=True
)