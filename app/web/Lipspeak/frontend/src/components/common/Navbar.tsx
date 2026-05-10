// src/components/common/Navbar.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Public navigation bar component.
 * @date: 10 May 2026
 * @returns: Navbar component.
 */

"use client";

import Link from "next/link";
import { Button } from "@/components/ui/shadcn/button";

export function Navbar() {
  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-black/70 backdrop-blur-2xl">
      <div className="mx-auto flex h-20 max-w-7xl items-center justify-between px-6">
        <Link href="/" className="text-2xl font-black tracking-tight">
          LipSpeak AI
        </Link>

        <nav className="hidden md:flex items-center gap-8 text-sm text-zinc-400">
          <Link href="/">Home</Link>
          <Link href="/privacy">Privacy</Link>
          <Link href="/terms">Terms</Link>
        </nav>

        <div className="flex items-center gap-4">
          <Link href="/login">
            <Button
              variant="ghost"
              className="text-zinc-300 hover:text-white"
            >
              Login
            </Button>
          </Link>

          <Link href="/signup">
            <Button className="bg-indigo-600 hover:bg-indigo-500 rounded-xl">
              Get Started
            </Button>
          </Link>
        </div>
      </div>
    </header>
  );
}