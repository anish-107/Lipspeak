// src/components/realtime/WebcamFeed.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Webcam feed preview component for real-time prediction.
 * @date: 10 May 2026
 * @returns: Webcam feed component.
 */

"use client";

import { Camera } from "lucide-react";

export function WebcamFeed() {
  return (
    <div className="rounded-[32px] border border-white/10 bg-zinc-900/60 p-6 backdrop-blur-2xl">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">
            Live Webcam Feed
          </h2>

          <p className="mt-2 text-zinc-400">
            Real-time neural speech recognition stream.
          </p>
        </div>

        <div className="flex items-center gap-2 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-4 py-2">
          <div className="h-2.5 w-2.5 rounded-full bg-emerald-400 animate-pulse" />

          <span className="text-sm text-emerald-300">
            Live
          </span>
        </div>
      </div>

      <div className="mt-8 flex aspect-video items-center justify-center rounded-[28px] border border-white/10 bg-black/60">
        <div className="text-center">
          <div className="mx-auto flex h-24 w-24 items-center justify-center rounded-full bg-indigo-500/10">
            <Camera className="h-10 w-10 text-indigo-400" />
          </div>

          <h3 className="mt-6 text-2xl font-bold">
            Webcam Preview
          </h3>

          <p className="mt-3 text-zinc-500">
            Real webcam stream integration will appear here.
          </p>
        </div>
      </div>
    </div>
  );
}