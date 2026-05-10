// src/components/auth/LoginForm.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Reusable login form component with futuristic UI styling.
 * @date: 10 May 2026
 * @returns: Login form component.
 */

"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/shadcn/button";
import { Input } from "@/components/ui/shadcn/input";
import { useAuth } from "@/hooks/useAuth";

export function LoginForm() {
  const {
    loginForm,
    loginLoading,
    handleLoginChange,
    handleLoginSubmit,
  } = useAuth();

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-[32px] border border-white/10 bg-zinc-900/60 p-8 backdrop-blur-2xl"
    >
      <form
        onSubmit={handleLoginSubmit}
        className="space-y-6"
      >
        <div className="space-y-2">
          <label className="text-sm text-zinc-300">
            Email Address
          </label>

          <Input
            type="email"
            name="email"
            placeholder="Enter your email"
            value={loginForm.email}
            onChange={handleLoginChange}
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
            placeholder="Enter your password"
            value={loginForm.password}
            onChange={handleLoginChange}
            className="h-12 rounded-xl border-zinc-700 bg-zinc-950"
          />
        </div>

        <Button
          type="submit"
          disabled={loginLoading}
          className="h-12 w-full rounded-xl bg-indigo-600 hover:bg-indigo-500"
        >
          {loginLoading ? (
            <Loader2 className="h-5 w-5 animate-spin" />
          ) : (
            "Login"
          )}
        </Button>
      </form>

      <div className="mt-8 text-center text-sm text-zinc-400">
        Don&apos;t have an account?{" "}
        <Link
          href="/signup"
          className="font-medium text-indigo-400 hover:text-indigo-300"
        >
          Sign up
        </Link>
      </div>
    </motion.div>
  );
}