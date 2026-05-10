// src/components/prerecorded/ResultsPanel.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: AI prediction results panel for uploaded videos.
 * @date: 10 May 2026
 * @returns: Results analysis component.
 */

export function ResultsPanel() {
  return (
    <div className="rounded-[32px] border border-white/10 bg-zinc-900/60 p-8 backdrop-blur-2xl">
      <div>
        <h2 className="text-2xl font-bold">
          Prediction Results
        </h2>

        <p className="mt-2 text-zinc-400">
          Neural speech recognition output.
        </p>
      </div>

      <div className="mt-10 rounded-[28px] border border-white/10 bg-black/40 p-8">
        <p className="text-sm uppercase tracking-[0.3em] text-zinc-500">
          Detected Speech
        </p>

        <h3 className="mt-6 text-4xl font-black gradient-text">
          WELCOME TO LIPSPEAK AI
        </h3>

        <p className="mt-6 text-zinc-400">
          Confidence score: 96.8%
        </p>
      </div>

      <div className="mt-8 space-y-4">
        <div className="rounded-2xl border border-white/10 bg-black/40 p-5">
          <p className="text-sm text-zinc-500">
            Video Length
          </p>

          <h4 className="mt-2 text-2xl font-black">
            02:14
          </h4>
        </div>

        <div className="rounded-2xl border border-white/10 bg-black/40 p-5">
          <p className="text-sm text-zinc-500">
            Frames Analyzed
          </p>

          <h4 className="mt-2 text-2xl font-black">
            4.8K
          </h4>
        </div>

        <div className="rounded-2xl border border-white/10 bg-black/40 p-5">
          <p className="text-sm text-zinc-500">
            AI Accuracy
          </p>

          <h4 className="mt-2 text-2xl font-black">
            96.8%
          </h4>
        </div>
      </div>
    </div>
  );
}