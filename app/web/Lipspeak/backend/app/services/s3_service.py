"""s3_service.py

@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: AWS S3 operations.
@date: 11 June 2026
@returns: S3 upload operations.

"""


# Imports
import uuid

import boto3

from app.core.config import (
    settings,
)


# S3 Client
s3_client = boto3.client(
    "s3",
    region_name=settings.AWS_REGION,
    aws_access_key_id=
    settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=
    settings.AWS_SECRET_ACCESS_KEY,
)


# S3 Service
class S3Service:
    """S3 service."""

    @staticmethod
    def upload_video(
        file_path: str,
        filename: str,
    ) -> str:
        """Upload video to S3."""

        unique_name = (
            f"{uuid.uuid4()}-"
            f"{filename}"
        )

        s3_client.upload_file(
            file_path,
            settings.AWS_S3_BUCKET,
            unique_name,
        )

        return (
            f"https://"
            f"{settings.AWS_S3_BUCKET}"
            f".s3."
            f"{settings.AWS_REGION}"
            f".amazonaws.com/"
            f"{unique_name}"
        )