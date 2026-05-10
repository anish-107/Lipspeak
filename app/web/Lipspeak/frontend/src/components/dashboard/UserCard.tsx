// src/components/dashboard/UserCard.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Dashboard user profile card.
 * @date: 10 May 2026
 * @returns: User card component.
 */

export function UserCard() {
  return (
    <div className="rounded-[32px] border border-white/10 bg-zinc-900/60 p-8 backdrop-blur-2xl">
      <div className="flex items-center gap-5">
        <div className="flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-r from-indigo-500 to-cyan-500 text-3xl font-black">
          A
        </div>

        <div>
          <h2 className="text-2xl font-bold">
            Anish Kumar
          </h2>

          <p className="mt-1 text-zinc-400">
            AI Platform Administrator
          </p>
        </div>
      </div>

      <div className="mt-8 grid grid-cols-2 gap-4">
        <div className="rounded-2xl border border-white/10 bg-black/40 p-5">
          <p className="text-sm text-zinc-500">
            Projects
          </p>

          <h3 className="mt-3 text-3xl font-black">
            12
          </h3>
        </div>

        <div className="rounded-2xl border border-white/10 bg-black/40 p-5">
          <p className="text-sm text-zinc-500">
            Predictions
          </p>

          <h3 className="mt-3 text-3xl font-black">
            8.4K
          </h3>
        </div>
      </div>
    </div>
  );
}