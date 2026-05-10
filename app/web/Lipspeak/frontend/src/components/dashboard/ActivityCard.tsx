// src/components/dashboard/ActivityCard.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Recent AI activity statistics card.
 * @date: 10 May 2026
 * @returns: Activity card component.
 */

export function ActivityCard() {
  return (
    <div className="rounded-[32px] border border-white/10 bg-zinc-900/60 p-8 backdrop-blur-2xl">
      <h2 className="text-2xl font-bold">
        Activity
      </h2>

      <div className="mt-8 space-y-5">
        <div className="rounded-2xl border border-white/10 bg-black/40 p-5">
          <p className="text-sm text-zinc-500">
            Real-Time Sessions
          </p>

          <h3 className="mt-2 text-3xl font-black">
            124
          </h3>
        </div>

        <div className="rounded-2xl border border-white/10 bg-black/40 p-5">
          <p className="text-sm text-zinc-500">
            Uploaded Videos
          </p>

          <h3 className="mt-2 text-3xl font-black">
            32
          </h3>
        </div>
      </div>
    </div>
  );
}