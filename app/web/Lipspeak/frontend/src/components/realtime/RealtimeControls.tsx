/** RealtimeControls.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Recording controls for real-time speech recognition.
 * @date: 10 June 2026
 * @returns: Real-time controls component.
 *
 */


// Imports
import {
  Play,
  Square,
  Wifi,
  WifiOff,
  Loader2,
} from "lucide-react";

import { Button } from "@/components/ui/shadcn/button";


// Props Interface
interface RealtimeControlsProps {
  isRecording: boolean;

  connectionStatus:
    | "disconnected"
    | "connecting"
    | "connected";

  onStart: () => void;

  onStop: () => void;
}


// Realtime Controls Component
export function RealtimeControls({
  isRecording,
  connectionStatus,
  onStart,
  onStop,
}: Readonly<RealtimeControlsProps>) {
  /* ---------------------------------------------------------------------- */
  /*                                Render                                  */
  /* ---------------------------------------------------------------------- */

  return (
    <div
      className="
        glass-card
        ai-border
        flex
        h-full
        flex-col
        rounded-3xl
        p-6
      "
    >
      {/* Header */}
      <h2
        className="
          text-2xl
          font-bold
        "
      >
        Controls
      </h2>

      <p
        className="
          mt-2
          text-sm
          text-muted-foreground
        "
      >
        Manage your real-time session.
      </p>

      {/* Status */}
      <div
        className="
          mt-6
          flex
          items-center
          gap-3
          rounded-2xl
          border
          border-border
          p-4
        "
      >
        {connectionStatus ===
          "connected" && (
          <>
            <Wifi
              className="
                h-5
                w-5
                text-green-500
              "
            />

            <span className="font-medium">
              Connected
            </span>
          </>
        )}

        {connectionStatus ===
          "connecting" && (
          <>
            <Loader2
              className="
                h-5
                w-5
                animate-spin
                text-yellow-500
              "
            />

            <span className="font-medium">
              Connecting...
            </span>
          </>
        )}

        {connectionStatus ===
          "disconnected" && (
          <>
            <WifiOff
              className="
                h-5
                w-5
                text-red-500
              "
            />

            <span className="font-medium">
              Disconnected
            </span>
          </>
        )}
      </div>

      {/* Buttons */}
      <div
        className="
          mt-6
          flex
          flex-col
          gap-3
          sm:flex-row
        "
      >
        <Button
          onClick={onStart}
          disabled={
            isRecording ||
            connectionStatus ===
              "connecting"
          }
          className="
            flex-1
            rounded-xl
            bg-green-600
            hover:bg-green-500
          "
        >
          <Play
            className="
              mr-2
              h-4
              w-4
            "
          />

          Start Recording
        </Button>

        <Button
          variant="destructive"
          onClick={onStop}
          disabled={!isRecording}
          className="
            flex-1
            rounded-xl
          "
        >
          <Square
            className="
              mr-2
              h-4
              w-4
            "
          />

          Stop Recording
        </Button>
      </div>
    </div>
  );
}