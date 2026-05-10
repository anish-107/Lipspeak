// src/components/realtime/Controls.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Real-time AI control actions panel.
 * @date: 10 May 2026
 * @returns: Real-time controls component.
 */

"use client";

import { Play, Square, RotateCcw } from "lucide-react";

export function Controls() {
  return (
    <div className="rounded-[32px] border border-white/10 bg-zinc-900/60 p-8 backdrop-blur-2xl">
      <h2 className="text-2xl font-bold">
        Controls
      </h2>

      <div className="mt-8 space-y-4">
        <button className="flex w-full items-center justify-center gap-3 rounded-2xl bg-indigo-600 px-5 py-4 font-medium transition-all hover:bg-indigo-500">
          <Play className="h-5 w-5" />

          Start Detection
        </button>

        <button className="flex w-full items-center justify-center gap-3 rounded-2xl border border-red-500/20 bg-red-500/10 px-5 py-4 font-medium text-red-300 transition-all hover:bg-red-500/20">
          <Square className="h-5 w-5" />

          Stop Detection
        </button>

        <button className="flex w-full items-center justify-center gap-3 rounded-2xl border border-white/10 bg-black/40 px-5 py-4 font-medium transition-all hover:bg-zinc-900">
          <RotateCcw className="h-5 w-5" />

          Reset Session
        </button>
      </div>
    </div>
  );
}