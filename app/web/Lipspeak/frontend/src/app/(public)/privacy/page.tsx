// src/app/(public)/privacy/page.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Privacy policy page for LipSpeak AI platform.
 * @date: 10 May 2026
 * @returns: Privacy policy page component.
 */

import { Navbar } from "@/components/common/Navbar";
import { Footer } from "@/components/common/Footer";

export default function PrivacyPage() {
  return (
    <main className="min-h-screen bg-black text-white">
      <Navbar />

      <section className="relative px-6 py-24">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(34,211,238,0.12),transparent_45%)]" />

        <div className="relative mx-auto max-w-4xl">
          <div className="mb-16">
            <h1 className="text-5xl font-black tracking-tight">
              Privacy Policy
            </h1>

            <p className="mt-6 text-lg text-zinc-400">
              Your privacy and data security are important to LipSpeak AI.
            </p>
          </div>

          <div className="space-y-10 rounded-[32px] border border-white/10 bg-zinc-900/60 p-10 backdrop-blur-2xl">
            <div>
              <h2 className="text-2xl font-bold">
                Data Collection
              </h2>

              <p className="mt-4 text-zinc-400 leading-relaxed">
                LipSpeak AI may collect authentication details,
                uploaded video files, and usage analytics to improve
                AI prediction quality and platform performance.
              </p>
            </div>

            <div>
              <h2 className="text-2xl font-bold">
                Security
              </h2>

              <p className="mt-4 text-zinc-400 leading-relaxed">
                All sensitive data is securely encrypted and processed
                using protected cloud infrastructure and secure APIs.
              </p>
            </div>

            <div>
              <h2 className="text-2xl font-bold">
                AI Processing
              </h2>

              <p className="mt-4 text-zinc-400 leading-relaxed">
                Uploaded media may be analyzed by AI models solely
                for speech recognition and accessibility purposes.
              </p>
            </div>

            <div>
              <h2 className="text-2xl font-bold">
                User Rights
              </h2>

              <p className="mt-4 text-zinc-400 leading-relaxed">
                Users may request deletion of their data and account
                information at any time through account settings.
              </p>
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}