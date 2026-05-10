// src/components/prerecorded/VideoPlayer.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Video preview player for uploaded recordings.
 * @date: 10 May 2026
 * @returns: Video player component.
 */

import { PlayCircle } from "lucide-react";

export function VideoPlayer() {
  return (
    <div className="rounded-[32px] border border-white/10 bg-zinc-900/60 p-6 backdrop-blur-2xl">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">
            Video Preview
          </h2>

          <p className="mt-2 text-zinc-400">
            Uploaded media playback and frame analysis.
          </p>
        </div>
      </div>

      <div className="mt-8 flex aspect-video items-center justify-center rounded-[28px] border border-white/10 bg-black/60">
        <div className="text-center">
          <PlayCircle className="mx-auto h-20 w-20 text-indigo-400" />

          <h3 className="mt-6 text-2xl font-bold">
            Uploaded Video Preview
          </h3>

          <p className="mt-3 text-zinc-500">
            Selected video will appear here.
          </p>
        </div>
      </div>
    </div>
  );
}