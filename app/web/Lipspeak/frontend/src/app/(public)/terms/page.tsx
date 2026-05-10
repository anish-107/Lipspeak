// src/app/(public)/terms/page.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Terms and conditions page for LipSpeak AI.
 * @date: 10 May 2026
 * @returns: Terms page component.
 */

import { Navbar } from "@/components/common/Navbar";
import { Footer } from "@/components/common/Footer";

export default function TermsPage() {
  return (
    <main className="min-h-screen bg-black text-white">
      <Navbar />

      <section className="relative px-6 py-24">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(99,102,241,0.15),transparent_45%)]" />

        <div className="relative mx-auto max-w-4xl">
          <div className="mb-16">
            <h1 className="text-5xl font-black tracking-tight">
              Terms & Conditions
            </h1>

            <p className="mt-6 text-lg text-zinc-400">
              Please read these terms carefully before using LipSpeak AI.
            </p>
          </div>

          <div className="space-y-10 rounded-[32px] border border-white/10 bg-zinc-900/60 p-10 backdrop-blur-2xl">
            <div>
              <h2 className="text-2xl font-bold">
                Platform Usage
              </h2>

              <p className="mt-4 text-zinc-400 leading-relaxed">
                Users must use the platform responsibly and avoid
                uploading harmful, illegal, or unauthorized content.
              </p>
            </div>

            <div>
              <h2 className="text-2xl font-bold">
                AI Predictions
              </h2>

              <p className="mt-4 text-zinc-400 leading-relaxed">
                LipSpeak AI predictions are generated using machine
                learning systems and may not always be perfectly accurate.
              </p>
            </div>

            <div>
              <h2 className="text-2xl font-bold">
                Account Responsibility
              </h2>

              <p className="mt-4 text-zinc-400 leading-relaxed">
                Users are responsible for maintaining the security
                of their credentials and account activity.
              </p>
            </div>

            <div>
              <h2 className="text-2xl font-bold">
                Service Availability
              </h2>

              <p className="mt-4 text-zinc-400 leading-relaxed">
                Platform availability may change due to updates,
                maintenance, or infrastructure requirements.
              </p>
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}