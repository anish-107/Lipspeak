// src/components/common/Footer.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Footer component for public pages.
 * @date: 10 May 2026
 * @returns: Footer component.
 */

import Link from "next/link";

export function Footer() {
  return (
    <footer className="border-t border-white/10 bg-black">
      <div className="mx-auto flex max-w-7xl flex-col gap-8 px-6 py-12 md:flex-row md:items-center md:justify-between">
        <div>
          <h3 className="text-2xl font-black">LipSpeak AI</h3>

          <p className="mt-3 text-sm text-zinc-500 max-w-md">
            AI-powered visual speech recognition platform for futuristic human
            communication systems.
          </p>
        </div>

        <div className="flex items-center gap-6 text-sm text-zinc-400">
          <Link href="/privacy">Privacy</Link>
          <Link href="/terms">Terms</Link>
          <Link href="/login">Login</Link>
        </div>
      </div>
    </footer>
  );
}