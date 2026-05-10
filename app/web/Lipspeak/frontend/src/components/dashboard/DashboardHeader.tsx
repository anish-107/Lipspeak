// src/components/dashboard/DashboardHeader.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Dashboard top navigation header.
 * @date: 10 May 2026
 * @returns: Dashboard header component.
 */

export function DashboardHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-white/10 bg-black/70 backdrop-blur-2xl">
      <div className="flex h-20 items-center justify-between px-6 md:px-10">
        <div>
          <h1 className="text-xl font-bold">
            AI Speech Analytics
          </h1>

          <p className="text-sm text-zinc-500">
            Neural lip-reading control center
          </p>
        </div>

        <div className="flex items-center gap-4">
          <div className="hidden rounded-full border border-white/10 bg-zinc-900/60 px-5 py-2 text-sm text-zinc-400 md:flex">
            Connected
          </div>

          <div className="flex h-11 w-11 items-center justify-center rounded-full bg-gradient-to-r from-indigo-500 to-cyan-500 font-bold">
            A
          </div>
        </div>
      </div>
    </header>
  );
}