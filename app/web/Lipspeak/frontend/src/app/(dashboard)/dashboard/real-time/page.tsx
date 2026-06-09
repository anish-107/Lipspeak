/** page.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Real-time speech recognition page.
 * @date: 10 June 2026
 * @returns: Real-time analysis page.
 *
 */


// Client Component
"use client";


// Imports
import { useRealtime } from "@/hooks/useRealtime";

import { WebcamFeed } from "@/components/realtime/WebcamFeed";
import { RealtimeControls } from "@/components/realtime/RealtimeControls";
import { LiveTranscript } from "@/components/realtime/LiveTranscript";


// Real Time Page
export default function RealTimePage() {
  /* ---------------------------------------------------------------------- */
  /*                                 Hook                                   */
  /* ---------------------------------------------------------------------- */

  const {
    videoRef,

    isRecording,

    transcript,

    connectionStatus,

    startRecording,

    stopRecording,
  } = useRealtime();

  /* ---------------------------------------------------------------------- */
  /*                                Render                                  */
  /* ---------------------------------------------------------------------- */
  
  return (
    <section className="space-y-8">
      {/* Header */}
      <div>
        <h1
          className="
            text-4xl
            font-black
            tracking-tight
          "
        >
          Real-Time Analysis
        </h1>
  
        <p
          className="
            mt-3
            text-muted-foreground
          "
        >
          Perform live lip-reading and
          receive transcripts in real time.
        </p>
      </div>
  
      {/* Top Section */}
      <div
        className="
          grid
          gap-6
          xl:grid-cols-[2fr_1fr]
        "
      >
        <WebcamFeed
          isRecording={isRecording}
          videoRef={videoRef}
        />
  
        <RealtimeControls
          isRecording={isRecording}
          connectionStatus={
            connectionStatus
          }
          onStart={startRecording}
          onStop={stopRecording}
        />
      </div>
  
      {/* Transcript */}
      <LiveTranscript
        transcript={transcript}
        isRecording={isRecording}
      />
    </section>
  );
}