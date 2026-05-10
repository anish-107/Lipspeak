// src/app/(public)/signup/page.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Public signup page for new user registration.
 * @date: 10 May 2026
 * @returns: Signup page component.
 */

import { Navbar } from "@/components/common/Navbar";
import { Footer } from "@/components/common/Footer";
import { SignupForm } from "@/components/auth/SignupForm";

export default function SignupPage() {
  return (
    <main className="min-h-screen bg-black text-white">
      <Navbar />

      <section className="relative flex min-h-[calc(100vh-160px)] items-center justify-center px-6 py-20">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(34,211,238,0.15),transparent_40%)]" />

        <div className="relative w-full max-w-md">
          <div className="mb-10 text-center">
            <h1 className="text-4xl font-black tracking-tight">
              Create Account
            </h1>

            <p className="mt-4 text-zinc-400">
              Start using next-generation visual speech AI today.
            </p>
          </div>

          <SignupForm />
        </div>
      </section>

      <Footer />
    </main>
  );
}