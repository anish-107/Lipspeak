// src/app/(dashboard)/dashboard/pre-recorded/page.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Pre-recorded video analysis dashboard page.
 * @date: 10 May 2026
 * @returns: Video upload analysis page component.
 */

import { VideoUploader } from "@/components/prerecorded/VideoUploader";
import { VideoPlayer } from "@/components/prerecorded/VideoPlayer";
import { ResultsPanel } from "@/components/prerecorded/ResultsPanel";

export default function PreRecordedPage() {
  return (
    <section className="space-y-10">
      <div>
        <h1 className="text-4xl font-black tracking-tight">
          Pre-Recorded Analysis
        </h1>

        <p className="mt-3 text-zinc-400">
          Upload videos and generate AI-powered speech predictions.
        </p>
      </div>

      <div className="grid gap-8 xl:grid-cols-[1.4fr_0.8fr]">
        <div className="space-y-8">
          <VideoUploader />

          <VideoPlayer />
        </div>

        <ResultsPanel />
      </div>
    </section>
  );
}