// src/components/dashboard/QuickActions.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Dashboard quick action shortcuts.
 * @date: 10 May 2026
 * @returns: Quick actions component.
 */

import Link from "next/link";

export function QuickActions() {
  return (
    <div className="rounded-[32px] border border-white/10 bg-zinc-900/60 p-8 backdrop-blur-2xl">
      <h2 className="text-2xl font-bold">
        Quick Actions
      </h2>

      <div className="mt-8 space-y-4">
        <Link
          href="/dashboard/real-time"
          className="block rounded-2xl bg-indigo-600 px-5 py-4 text-center font-medium transition-all hover:bg-indigo-500"
        >
          Start Real-Time Detection
        </Link>

        <Link
          href="/dashboard/pre-recorded"
          className="block rounded-2xl border border-white/10 bg-black/40 px-5 py-4 text-center font-medium transition-all hover:bg-zinc-900"
        >
          Upload Video
        </Link>
      </div>
    </div>
  );
}