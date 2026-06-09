/** WebcamFeed.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Webcam feed display component for real-time speech recognition.
 * @date: 10 June 2026
 * @returns: Webcam feed component.
 *
 */


// Client Component
"use client";


// Imports
// import { Camera } from "lucide-react";


// Props Interface
interface WebcamFeedProps {
  isRecording: boolean;

  videoRef: React.RefObject<HTMLVideoElement | null>;
}


// Webcam Feed Component
export function WebcamFeed({
  isRecording,
  videoRef,
}: Readonly<WebcamFeedProps>) {
  return (
    <div
      className="
        glass-card
        ai-border
        rounded-3xl
        p-6
      "
    >
      {/* Header */}
      <div
        className="
          flex
          items-center
          justify-between
        "
      >
        <h2
          className="
            text-2xl
            font-bold
          "
        >
          Webcam Feed
        </h2>

        {isRecording && (
          <div
            className="
              flex
              items-center
              gap-2
              rounded-full
              bg-red-500/10
              px-3
              py-1
              text-sm
              font-medium
              text-red-500
            "
          >
            <span
              className="
                h-2
                w-2
                animate-pulse
                rounded-full
                bg-red-500
              "
            />

            Recording
          </div>
        )}
      </div>

      {/* Camera Feed */}
      <div
        className="
          mt-6
          overflow-hidden
          rounded-2xl
          border
          border-border
          bg-black
        "
      >
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="
            h-60
            w-full
            object-cover
            md:h-100
          "
        />
      </div>

      {/* Helper Text */}
      <p
        className="
          mt-4
          text-center
          text-sm
          text-muted-foreground
        "
      >
        {isRecording
          ? "Live camera feed is active."
          : "Camera ready for real-time analysis."}
      </p>
    </div>
  );
}