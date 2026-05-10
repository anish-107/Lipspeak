// src/app/(dashboard)/dashboard/real-time/page.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Real-time lip reading dashboard page.
 * @date: 10 May 2026
 * @returns: Real-time prediction page component.
 */

import { WebcamFeed } from "@/components/realtime/WebcamFeed";
import { PredictionBox } from "@/components/realtime/PredictionBox";
import { Controls } from "@/components/realtime/Controls";

export default function RealTimePage() {
  return (
    <section className="space-y-10">
      <div>
        <h1 className="text-4xl font-black tracking-tight">
          Real-Time Detection
        </h1>

        <p className="mt-3 text-zinc-400">
          Live webcam-based AI lip reading and speech prediction.
        </p>
      </div>

      <div className="grid gap-8 xl:grid-cols-[1.5fr_0.8fr]">
        <WebcamFeed />

        <div className="space-y-8">
          <PredictionBox />

          <Controls />
        </div>
      </div>
    </section>
  );
}