/** SignupForm.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Authentication form for user registration.
 * @date: 10 June 2026
 * @returns: Signup form component.
 *
 */


// Client Component
"use client";


// Imports
import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Loader2, UserPlus } from "lucide-react";

import { Button } from "@/components/ui/shadcn/button";
import { Input } from "@/components/ui/shadcn/input";

import { authService } from "@/services/api/auth.service";


// SignupForm Component
export function SignupForm() {
  // Router
  const router = useRouter();

  // State
  const [username, setUsername] =
    useState("");

  const [name, setName] =
    useState("");

  const [password, setPassword] =
    useState("");

  const [confirmPassword, setConfirmPassword] =
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

    if (!name.trim()) {
      setError("Full name is required.");
      return;
    }

    if (!password.trim()) {
      setError("Password is required.");
      return;
    }

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    try {
      setLoading(true);

      await authService.signup({
        username,
        name,
        password,
      });

      router.push("/login");
    } catch (error) {
      console.error(error);

      setError(
        "Failed to create account.",
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
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="
        glass-card
        ai-border
        rounded-3xl
        p-6
        shadow-2xl
        md:p-8
      "
    >
      <div className="mb-8 text-center">
        <h2
          className="
            text-2xl
            font-black
            tracking-tight
            md:text-3xl
          "
        >
          Create Account
        </h2>

        <p
          className="
            mt-3
            text-sm
            text-muted-foreground
          "
        >
          Start using LipSpeak AI today.
        </p>
      </div>

      <form
        onSubmit={handleSubmit}
        className="space-y-5"
      >
        <Input
          placeholder="Username"
          value={username}
          onChange={(e) =>
            setUsername(e.target.value)
          }
          className="h-12 rounded-xl"
        />

        <Input
          placeholder="Full Name"
          value={name}
          onChange={(e) =>
            setName(e.target.value)
          }
          className="h-12 rounded-xl"
        />

        <Input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) =>
            setPassword(e.target.value)
          }
          className="h-12 rounded-xl"
        />

        <Input
          type="password"
          placeholder="Confirm Password"
          value={confirmPassword}
          onChange={(e) =>
            setConfirmPassword(
              e.target.value,
            )
          }
          className="h-12 rounded-xl"
        />

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
          "
        >
          {loading ? (
            <Loader2 className="h-5 w-5 animate-spin" />
          ) : (
            <>
              <UserPlus className="mr-2 h-4 w-4" />
              Create Account
            </>
          )}
        </Button>
      </form>

      <div
        className="
          mt-8
          text-center
          text-sm
          text-muted-foreground
        "
      >
        Already have an account?{" "}
        <Link
          href="/login"
          className="
            font-medium
            text-primary
            hover:underline
          "
        >
          Sign In
        </Link>
      </div>
    </motion.div>
  );
}