// src/components/common/CTASection.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: CTA section encouraging user signup.
 * @date: 10 May 2026
 * @returns: CTA section component.
 */

import Link from "next/link";
import { Button } from "@/components/ui/shadcn/button";

export function CTASection() {
  return (
    <section className="px-6 py-28">
      <div className="mx-auto max-w-5xl rounded-[40px] border border-indigo-500/20 bg-gradient-to-r from-indigo-600/10 to-cyan-500/10 p-12 md:p-20 text-center backdrop-blur-xl">
        <h2 className="text-4xl md:text-6xl font-black leading-tight">
          Experience Next-Generation
          <span className="block bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
            AI Lip Reading
          </span>
        </h2>

        <p className="mt-6 text-lg text-zinc-400 max-w-2xl mx-auto">
          Start analyzing speech visually with powerful AI tools designed for
          accessibility, research, and intelligent automation.
        </p>

        <div className="mt-10">
          <Link href="/signup">
            <Button className="h-12 px-10 rounded-xl bg-indigo-600 hover:bg-indigo-500">
              Create Free Account
            </Button>
          </Link>
        </div>
      </div>
    </section>
  );
}