/** ResultsPanel.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Transcript results panel for pre-recorded video analysis.
 * @date: 10 June 2026
 * @returns: Results panel component.
 *
 */


// Imports
import { Loader2, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/shadcn/button";


// Props Interface
interface ResultsPanelProps {
  transcript: string;

  loading: boolean;

  hasVideo: boolean;

  onAnalyze: () => void;
}


// Results Panel Component
export function ResultsPanel({
  transcript,
  loading,
  hasVideo,
  onAnalyze,
}: Readonly<ResultsPanelProps>) {
  /* ---------------------------------------------------------------------- */
  /*                                Render                                  */
  /* ---------------------------------------------------------------------- */

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
          flex-col
          gap-4
          sm:flex-row
          sm:items-center
          sm:justify-between
        "
      >
        <div>
          <h2
            className="
              text-2xl
              font-bold
            "
          >
            Transcript Results
          </h2>

          <p
            className="
              mt-2
              text-sm
              text-muted-foreground
            "
          >
            AI generated speech transcript.
          </p>
        </div>

        <Button
          onClick={onAnalyze}
          disabled={!hasVideo || loading}
          className="
            rounded-xl
            bg-linear-to-r
            from-indigo-600
            via-purple-600
            to-cyan-600
          "
        >
          {loading ? (
            <>
              <Loader2
                className="
                  mr-2
                  h-4
                  w-4
                  animate-spin
                "
              />

              Processing...
            </>
          ) : (
            <>
              <Sparkles
                className="
                  mr-2
                  h-4
                  w-4
                "
              />

              Analyze Video
            </>
          )}
        </Button>
      </div>

      {/* Empty State */}
      {!loading && !transcript && (
        <div
          className="
            mt-6
            flex
            min-h-55
            items-center
            justify-center
            rounded-2xl
            border
            border-dashed
            border-border
            p-6
            text-center
          "
        >
          <p className="text-muted-foreground">
            Upload a video and click
            &quot;Analyze Video&quot; to generate
            a transcript.
          </p>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div
          className="
            mt-6
            flex
            min-h-55
            flex-col
            items-center
            justify-center
            rounded-2xl
            border
            border-border
          "
        >
          <Loader2
            className="
              h-10
              w-10
              animate-spin
              text-primary
            "
          />

          <p
            className="
              mt-4
              text-muted-foreground
            "
          >
            Processing video...
          </p>
        </div>
      )}

      {/* Transcript */}
      {!loading && transcript && (
        <div
          className="
            mt-6
            rounded-2xl
            border
            border-border
            p-5
          "
        >
          <p
            className="
              whitespace-pre-wrap
              leading-7
            "
          >
            {transcript}
          </p>
        </div>
      )}
    </div>
  );
}