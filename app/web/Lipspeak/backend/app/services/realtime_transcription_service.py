"""realtime_transcription_service.py

@authors: Anish Kumar, Bidipta Barua,
          Dibyasmita Hati, Arpan Haldar
@description: Realtime transcription orchestration.
@date: 11 June 2026
@returns: Realtime transcript generation.
"""

import asyncio
from app.grpc.clients.avsr_client import AVSRClient

class RealtimeTranscriptionService:
    """Realtime transcription service."""

    def __init__(self):
        """Initialize service."""
        self.transcript_history: list[str] = []
        self.last_transcript = ""

    async def process_chunk(self, video_bytes: bytes) -> str:
        """Process incoming 5-second complete video file and return accumulated transcript."""
        if not video_bytes:
            return ""

        print(f"Realtime video window received: {len(video_bytes)} bytes")

        try:
            print(f"Sending {len(video_bytes)} bytes to AVSR...")

            # Run the synchronous gRPC call in a background thread 
            # so it DOES NOT block the FastAPI websocket event loop!
            transcript = await asyncio.to_thread(AVSRClient.predict, video_bytes)

            print(f"\nBACKEND RECEIVED: {repr(transcript)}\n")

            if transcript and transcript.strip():
                clean_transcript = transcript.strip()
                self.last_transcript = clean_transcript
                self.transcript_history.append(clean_transcript)

        except Exception as exc:
            print(f"AVSR Error: {exc}")

        return " ".join(self.transcript_history)

    def get_latest_transcript(self) -> str:
        """Return latest transcript."""
        return self.last_transcript

    def reset(self):
        """Reset session."""
        self.transcript_history.clear()
        self.last_transcript = ""