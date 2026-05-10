// src/app/(dashboard)/dashboard/page.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Main dashboard overview page.
 * @date: 10 May 2026
 * @returns: Dashboard home page component.
 */

import { UserCard } from "@/components/dashboard/UserCard";
import { ActivityCard } from "@/components/dashboard/ActivityCard";
import { QuickActions } from "@/components/dashboard/QuickActions";

export default function DashboardPage() {
  return (
    <section className="space-y-10">
      <div>
        <h1 className="text-4xl font-black tracking-tight">
          AI Dashboard
        </h1>

        <p className="mt-3 text-zinc-400">
          Monitor AI speech recognition systems and predictions.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <UserCard />

        <ActivityCard />

        <QuickActions />
      </div>

      <div className="rounded-[32px] border border-white/10 bg-zinc-900/60 p-8 backdrop-blur-2xl">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold">
              Neural Processing Status
            </h2>

            <p className="mt-2 text-zinc-400">
              Real-time AI inference monitoring system.
            </p>
          </div>

          <div className="flex items-center gap-3 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-4 py-2">
            <div className="h-2.5 w-2.5 rounded-full bg-emerald-400 animate-pulse" />

            <span className="text-sm text-emerald-300">
              Active
            </span>
          </div>
        </div>

        <div className="mt-10 grid gap-6 md:grid-cols-3">
          <div className="rounded-2xl border border-white/10 bg-black/40 p-6">
            <p className="text-sm text-zinc-500">
              Predictions Today
            </p>

            <h3 className="mt-4 text-4xl font-black">
              12.4K
            </h3>
          </div>

          <div className="rounded-2xl border border-white/10 bg-black/40 p-6">
            <p className="text-sm text-zinc-500">
              Average Accuracy
            </p>

            <h3 className="mt-4 text-4xl font-black">
              97%
            </h3>
          </div>

          <div className="rounded-2xl border border-white/10 bg-black/40 p-6">
            <p className="text-sm text-zinc-500">
              Active AI Models
            </p>

            <h3 className="mt-4 text-4xl font-black">
              12
            </h3>
          </div>
        </div>
      </div>
    </section>
  );
}