"""realtime_transcription_service.py

@authors: Anish Kumar, Bidipta Barua,
Dibyasmita Hati, Arpan Haldar
@description: Realtime transcription orchestration.
@date: 11 June 2026
@returns: Realtime transcript generation.

"""

from app.grpc.clients.avsr_client import (
    AVSRClient,
)
import tempfile
import subprocess


class RealtimeTranscriptionService:
    """Realtime transcription service."""

    CHUNKS_PER_WINDOW = 5

    def __init__(self):
        """Initialize service."""

        self.chunks: list[bytes] = []

        self.transcript_history: list[str] = []
    

    def merge_chunks(
        self,
    ) -> bytes:
    
        with tempfile.TemporaryDirectory() as temp_dir:
    
            chunk_paths = []
    
            for index, chunk in enumerate(
                self.chunks
            ):
    
                chunk_path = (
                    f"{temp_dir}/chunk_{index}.webm"
                )
    
                with open(
                    chunk_path,
                    "wb",
                ) as file:
    
                    file.write(
                        chunk,
                    )
    
                chunk_paths.append(
                    chunk_path,
                )
    
            concat_file = (
                f"{temp_dir}/list.txt"
            )
    
            with open(
                concat_file,
                "w",
            ) as file:
    
                for path in chunk_paths:
    
                    file.write(
                        f"file '{path}'\n"
                    )
    
            merged_path = (
                f"{temp_dir}/merged.webm"
            )
    
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    concat_file,
                    "-c",
                    "copy",
                    merged_path,
                ],
                check=True,
            )
    
            with open(
                merged_path,
                "rb",
            ) as file:
    
                return file.read()
        

    def process_chunk(
        self,
        chunk: bytes,
    ) -> str:
        """Process incoming chunk."""

        self.chunks.append(
            chunk,
        )

        print(
            f"Realtime chunk received: "
            f"{len(chunk)} bytes "
            f"({len(self.chunks)}/"
            f"{self.CHUNKS_PER_WINDOW})"
        )

        if (
            len(self.chunks)
            < self.CHUNKS_PER_WINDOW
        ):
            return ""

        video_bytes = (
            self.merge_chunks()
        )

        print(
            f"Sending "
            f"{len(video_bytes)} bytes "
            f"to AVSR..."
        )

        try:

            transcript = (
                AVSRClient.predict(
                    video_bytes,
                )
            )

            print(
                "\nBACKEND RECEIVED:",
                repr(transcript),
                "\n",
            )

            if transcript:

                self.transcript_history.append(
                    transcript,
                )

        except Exception as exc:

            print(
                f"AVSR Error: {exc}"
            )

            transcript = ""

        self.chunks.clear()

        return " ".join(
            self.transcript_history,
        )

    def reset(
        self,
    ):
        """Reset session."""

        self.chunks.clear()

        self.transcript_history.clear()