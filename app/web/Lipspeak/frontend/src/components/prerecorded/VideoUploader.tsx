// src/components/prerecorded/VideoUploader.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Video upload component for pre-recorded AI analysis.
 * @date: 10 May 2026
 * @returns: Video uploader component.
 */

"use client";

import { Upload } from "lucide-react";

export function VideoUploader() {
  return (
    <div className="rounded-[32px] border border-dashed border-indigo-500/30 bg-zinc-900/60 p-10 backdrop-blur-2xl">
      <div className="flex flex-col items-center justify-center text-center">
        <div className="flex h-24 w-24 items-center justify-center rounded-full bg-indigo-500/10">
          <Upload className="h-10 w-10 text-indigo-400" />
        </div>

        <h2 className="mt-8 text-3xl font-black">
          Upload Video
        </h2>

        <p className="mt-4 max-w-xl text-zinc-400">
          Drag and drop your pre-recorded video or browse files
          for AI speech recognition analysis.
        </p>

        <label className="mt-8 cursor-pointer rounded-2xl bg-indigo-600 px-6 py-4 font-medium transition-all hover:bg-indigo-500">
          Choose Video

          <input
            type="file"
            accept="video/*"
            className="hidden"
          />
        </label>
      </div>
    </div>
  );
}