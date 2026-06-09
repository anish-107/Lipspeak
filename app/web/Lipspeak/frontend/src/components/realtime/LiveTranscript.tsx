/** LiveTranscript.tsx
* @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
* @description: Live transcript display panel for real-time speech recognition.
* @date: 10 June 2026
* @returns: Live transcript component.
*
*/

// Client Component
"use client";

// Imports
import { useState } from "react";
import {
  Check,
  Copy,
  Download,
  MessageSquareText,
} from "lucide-react";
import { Button } from "@/components/ui/shadcn/button";

// Props Interface
interface LiveTranscriptProps {
  transcript: string;
  isRecording: boolean;
}

// Live Transcript Component
export function LiveTranscript({
  transcript,
  isRecording,
}: Readonly<LiveTranscriptProps>) {
  /* ---------------------------------------------------------------------- /
  / State /
  / ---------------------------------------------------------------------- */
  const [copied, setCopied] = useState(false);

  /* ---------------------------------------------------------------------- /
  / Handlers /
  / ---------------------------------------------------------------------- */
  const handleCopy = async () => {
    if (!transcript) return;
    
    await navigator.clipboard.writeText(transcript);
    setCopied(true);

    setTimeout(() => {
      setCopied(false);
    }, 2000);
  };

  const handleDownload = () => {
    if (!transcript) return;

    const blob = new Blob([transcript], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    
    link.href = url;
    link.download = `transcript-${Date.now()}.txt`;
    link.click();
    
    URL.revokeObjectURL(url);
  };

  /* ---------------------------------------------------------------------- /
  / Empty State /
  / ---------------------------------------------------------------------- */
  if (!transcript) {
    return (
      <div className="rounded-2xl border border-border bg-background p-6 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <h2 className="text-lg font-semibold text-foreground">Live Transcript</h2>
          
          <div className="flex gap-2">
            <Button variant="outline" size="sm" disabled>
              <Copy className="mr-2 h-4 w-4" />
              Copy
            </Button>

            <Button variant="outline" size="sm" disabled>
              <Download className="mr-2 h-4 w-4" />
              Download
            </Button>
          </div>
        </div>

        <div className="mt-6 flex min-h-70 flex-col items-center justify-center rounded-2xl border border-dashed border-border text-center p-6">
          <MessageSquareText className="h-14 w-14 text-muted-foreground" />
          <p className="mt-4 max-w-md text-muted-foreground text-sm">
            {isRecording
              ? "Waiting for speech..."
              : "Start a session to begin transcription."}
          </p>
        </div>
      </div>
    );
  }

  /* ---------------------------------------------------------------------- /
  / Active Render State /
  / ---------------------------------------------------------------------- */
  return (
    <div className="rounded-2xl border border-border bg-background p-6 shadow-sm">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold text-foreground">Live Transcript</h2>
          {isRecording && (
            <div className="flex items-center gap-2 rounded-full bg-green-500/10 px-3 py-1 text-xs font-medium text-green-600 dark:text-green-400">
              <span className="h-2 w-2 animate-pulse rounded-full bg-green-500" />
              Live
            </div>
          )}
        </div>

        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={handleCopy}>
            {copied ? (
              <>
                <Check className="mr-2 h-4 w-4 text-green-500" />
                Copied
              </>
            ) : (
              <>
                <Copy className="mr-2 h-4 w-4" />
                Copy
              </>
            )}
          </Button>

          <Button variant="outline" size="sm" onClick={handleDownload}>
            <Download className="mr-2 h-4 w-4" />
            Download
          </Button>
        </div>
      </div>

      <div className="mt-6 max-h-112.5 overflow-y-auto rounded-2xl border border-border bg-muted/30 p-5 backdrop-blur-sm">
        <p className="whitespace-pre-wrap text-base leading-7 text-foreground/90 font-normal">
          {transcript}
        </p>
      </div>
    </div>
  );
}