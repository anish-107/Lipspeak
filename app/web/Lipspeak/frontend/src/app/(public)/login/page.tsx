// src/app/(public)/login/page.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Public login page for user authentication.
 * @date: 10 May 2026
 * @returns: Login page component.
 */

import { Navbar } from "@/components/common/Navbar";
import { Footer } from "@/components/common/Footer";
import { LoginForm } from "@/components/auth/LoginForm";

export default function LoginPage() {
  return (
    <main className="min-h-screen bg-black text-white">
      <Navbar />

      <section className="relative flex min-h-[calc(100vh-160px)] items-center justify-center px-6 py-20">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(99,102,241,0.18),transparent_40%)]" />

        <div className="relative w-full max-w-md">
          <div className="mb-10 text-center">
            <h1 className="text-4xl font-black tracking-tight">
              Welcome Back
            </h1>

            <p className="mt-4 text-zinc-400">
              Access your AI-powered lip reading dashboard.
            </p>
          </div>

          <LoginForm />
        </div>
      </section>

      <Footer />
    </main>
  );
}