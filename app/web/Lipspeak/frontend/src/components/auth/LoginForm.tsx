/** LoginForm.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Authentication form for user login.
 * @date: 10 June 2026
 * @returns: Login form component.
 *
 */


// Client Component
"use client";


// Imports
import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Loader2, LogIn } from "lucide-react";

import { Button } from "@/components/ui/shadcn/button";
import { Input } from "@/components/ui/shadcn/input";

import { authService } from "@/services/api/auth.service";
import { useAuthStore } from "@/store/auth.store";


// LoginForm Component
export function LoginForm() {
  // Router
  const router = useRouter();

  // Store
  const login = useAuthStore(
    (state) => state.login,
  );

  // State
  const [username, setUsername] =
    useState("");

  const [password, setPassword] =
    useState("");

  const [error, setError] =
    useState("");

  const [loading, setLoading] =
    useState(false);

  /* ---------------------------------------------------------------------- */
  /*                             Form Submit                                */
  /* ---------------------------------------------------------------------- */

  const handleSubmit = async (
    event: React.FormEvent<HTMLFormElement>,
  ) => {
    event.preventDefault();

    setError("");

    if (!username.trim()) {
      setError("Username is required.");
      return;
    }

    if (!password.trim()) {
      setError("Password is required.");
      return;
    }

    try {
      setLoading(true);

      const response =
        await authService.login({
          username,
          password,
        });
      
      localStorage.setItem(
        "access_token",
        response.access_token,
      );
      
      const user =
        await authService.getCurrentUser();
      
      login(
        response.access_token,
        user,
      );
      
      router.push(
        "/dashboard",
      );
      
    } catch (error) {
      console.error(error);

      setError(
        "Invalid username or password.",
      );
    } finally {
      setLoading(false);
    }
  };

  /* ---------------------------------------------------------------------- */
  /*                                Render                                  */
  /* ---------------------------------------------------------------------- */

  return (
    <motion.div
      initial={{
        opacity: 0,
        y: 20,
      }}
      animate={{
        opacity: 1,
        y: 0,
      }}
      transition={{
        duration: 0.4,
      }}
      className="
        glass-card
        ai-border
        rounded-3xl
        p-6
        shadow-2xl
        md:p-8
      "
    >
      {/* Header */}
      <div className="mb-8 text-center">
        <h2
          className="
            text-2xl
            font-black
            tracking-tight
            md:text-3xl
          "
        >
          Sign In
        </h2>

        <p
          className="
            mt-3
            text-sm
            text-muted-foreground
          "
        >
          Continue to your LipSpeak AI dashboard.
        </p>
      </div>

      {/* Form */}
      <form
        onSubmit={handleSubmit}
        className="space-y-5"
      >
        {/* Username */}
        <div className="space-y-2">
          <label
            htmlFor="username"
            className="
              text-sm
              font-medium
            "
          >
            Username
          </label>

          <Input
            id="username"
            type="text"
            placeholder="Enter your username"
            value={username}
            onChange={(event) =>
              setUsername(
                event.target.value,
              )
            }
            className="
              h-12
              rounded-xl
            "
          />
        </div>

        {/* Password */}
        <div className="space-y-2">
          <label
            htmlFor="password"
            className="
              text-sm
              font-medium
            "
          >
            Password
          </label>

          <Input
            id="password"
            type="password"
            placeholder="Enter your password"
            value={password}
            onChange={(event) =>
              setPassword(
                event.target.value,
              )
            }
            className="
              h-12
              rounded-xl
            "
          />
        </div>

        {/* Error */}
        {error && (
          <div
            className="
              rounded-xl
              border
              border-red-500/20
              bg-red-500/10
              px-4
              py-3
              text-sm
              text-red-500
            "
          >
            {error}
          </div>
        )}

        {/* Submit */}
        <Button
          type="submit"
          disabled={loading}
          className="
            h-12
            w-full
            rounded-xl
            bg-linear-to-r
            from-indigo-600
            via-purple-600
            to-cyan-600
            transition-all
            duration-300
            hover:scale-[1.02]
          "
        >
          {loading ? (
            <Loader2
              className="
                h-5
                w-5
                animate-spin
              "
            />
          ) : (
            <>
              <LogIn className="mr-2 h-4 w-4" />
              Login
            </>
          )}
        </Button>
      </form>

      {/* Footer */}
      <div
        className="
          mt-8
          text-center
          text-sm
          text-muted-foreground
        "
      >
        Don&apos;t have an account?{" "}
        <Link
          href="/signup"
          className="
            font-medium
            text-primary
            hover:underline
          "
        >
          Create Account
        </Link>
      </div>
    </motion.div>
  );
}