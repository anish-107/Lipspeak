/** page.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Pre-recorded video analysis page.
 * @date: 10 June 2026
 * @returns: Pre-recorded analysis page.
 *
 */


// Client Component
"use client";


// Imports
import { usePreRecorded } from "@/hooks/usePreRecorded";

import { VideoUploader } from "@/components/prerecorded/VideoUploader";
import { VideoPlayer } from "@/components/prerecorded/VideoPlayer";
import { ResultsPanel } from "@/components/prerecorded/ResultsPanel";


// Pre Recorded Page
export default function PreRecordedPage() {
  /* ---------------------------------------------------------------------- */
  /*                                 Hook                                   */
  /* ---------------------------------------------------------------------- */

  const {
    videoFile,
    videoUrl,
    transcript,
    loading,
    handleVideoSelect,
    processVideo,
  } = usePreRecorded();

  /* ---------------------------------------------------------------------- */
  /*                                Render                                  */
  /* ---------------------------------------------------------------------- */

  return (
    <section className="space-y-8">
      {/* Page Header */}
      <div>
        <h1
          className="
            text-4xl
            font-black
            tracking-tight
          "
        >
          Pre-Recorded Analysis
        </h1>

        <p
          className="
            mt-3
            text-muted-foreground
          "
        >
          Upload a video and generate
          AI-powered speech transcripts.
        </p>
      </div>

      {/* Upload */}
      <VideoUploader
        selectedFile={videoFile}
        onVideoSelect={
          handleVideoSelect
        }
      />

      {/* Preview */}
      <VideoPlayer
        videoUrl={videoUrl}
      />

      {/* Results */}
      <ResultsPanel
        transcript={transcript}
        loading={loading}
        hasVideo={!!videoFile}
        onAnalyze={processVideo}
      />
    </section>
  );
}