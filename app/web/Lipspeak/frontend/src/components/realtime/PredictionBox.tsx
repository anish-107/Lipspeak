// src/components/realtime/PredictionBox.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: AI prediction output panel for real-time speech recognition.
 * @date: 10 May 2026
 * @returns: Prediction output component.
 */

export function PredictionBox() {
  return (
    <div className="rounded-[32px] border border-white/10 bg-zinc-900/60 p-8 backdrop-blur-2xl">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">
            AI Predictions
          </h2>

          <p className="mt-2 text-zinc-400">
            Live speech inference results.
          </p>
        </div>

        <div className="rounded-full border border-cyan-500/20 bg-cyan-500/10 px-4 py-2 text-sm text-cyan-300">
          Processing
        </div>
      </div>

      <div className="mt-10 rounded-[28px] border border-white/10 bg-black/40 p-8">
        <p className="text-sm uppercase tracking-[0.3em] text-zinc-500">
          Current Prediction
        </p>

        <h3 className="mt-6 text-5xl font-black gradient-text">
          HELLO
        </h3>

        <p className="mt-6 text-zinc-400">
          Confidence score: 97.4%
        </p>
      </div>

      <div className="mt-8 grid grid-cols-2 gap-4">
        <div className="rounded-2xl border border-white/10 bg-black/40 p-5">
          <p className="text-sm text-zinc-500">
            Frames Processed
          </p>

          <h4 className="mt-3 text-3xl font-black">
            1.2K
          </h4>
        </div>

        <div className="rounded-2xl border border-white/10 bg-black/40 p-5">
          <p className="text-sm text-zinc-500">
            Prediction Delay
          </p>

          <h4 className="mt-3 text-3xl font-black">
            42ms
          </h4>
        </div>
      </div>
    </div>
  );
}