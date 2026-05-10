// src/components/auth/SignupForm.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Reusable signup form component for user registration.
 * @date: 10 May 2026
 * @returns: Signup form component.
 */

"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/shadcn/button";
import { Input } from "@/components/ui/shadcn/input";
import { useAuth } from "@/hooks/useAuth";

export function SignupForm() {
  const {
    signupForm,
    signupLoading,
    handleSignupChange,
    handleSignupSubmit,
  } = useAuth();

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-[32px] border border-white/10 bg-zinc-900/60 p-8 backdrop-blur-2xl"
    >
      <form
        onSubmit={handleSignupSubmit}
        className="space-y-6"
      >
        <div className="space-y-2">
          <label className="text-sm text-zinc-300">
            Full Name
          </label>

          <Input
            type="text"
            name="name"
            placeholder="Enter your full name"
            value={signupForm.name}
            onChange={handleSignupChange}
            className="h-12 rounded-xl border-zinc-700 bg-zinc-950"
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm text-zinc-300">
            Email Address
          </label>

          <Input
            type="email"
            name="email"
            placeholder="Enter your email"
            value={signupForm.email}
            onChange={handleSignupChange}
            className="h-12 rounded-xl border-zinc-700 bg-zinc-950"
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm text-zinc-300">
            Password
          </label>

          <Input
            type="password"
            name="password"
            placeholder="Create a password"
            value={signupForm.password}
            onChange={handleSignupChange}
            className="h-12 rounded-xl border-zinc-700 bg-zinc-950"
          />
        </div>

        <Button
          type="submit"
          disabled={signupLoading}
          className="h-12 w-full rounded-xl bg-cyan-600 hover:bg-cyan-500"
        >
          {signupLoading ? (
            <Loader2 className="h-5 w-5 animate-spin" />
          ) : (
            "Create Account"
          )}
        </Button>
      </form>

      <div className="mt-8 text-center text-sm text-zinc-400">
        Already have an account?{" "}
        <Link
          href="/login"
          className="font-medium text-cyan-400 hover:text-cyan-300"
        >
          Login
        </Link>
      </div>
    </motion.div>
  );
}