/**
 * Navbar.tsx
 * @authors: Anish Kumar, Bidipta Barua,
 * Dibyasmita Hati, Arpan Haldar
 * @description: Public navigation bar component.
 * @date: 10 May 2026
 * @returns: Navbar component.
 */

"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import { Menu } from "lucide-react";

import { Button } from "@/components/ui/shadcn/button";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
} from "@/components/ui/shadcn/sheet";

const ThemeToggle = dynamic(
  () =>
    import("@/components/common/ThemeToggle").then(
      (mod) => mod.ThemeToggle,
    ),
  {
    ssr: false,
  },
);

export function Navbar() {
  return (
    <header
      className="
        sticky
        top-0
        z-50
        w-full
        border-b
        border-border/50
        bg-background/80
        backdrop-blur-xl
      "
    >
      <div
        className="
          mx-auto
          flex
          h-18
          max-w-7xl
          items-center
          justify-between
          px-4
          sm:px-6
          lg:px-8
        "
      >
        {/* Logo */}
        <Link
          href="/"
          className="
            shrink-0
            text-xl
            font-black
            tracking-tight
            md:text-2xl
          "
        >
          <span
            className="
              bg-linear-to-r
              from-indigo-500
              via-cyan-500
              to-purple-500
              bg-clip-text
              text-transparent
            "
          >
            LipSpeak AI
          </span>
        </Link>

        {/* Desktop Nav */}
        <nav
          className="
            hidden
            items-center
            gap-8
            md:flex
          "
        >
          <Link
            href="/"
            className="
              text-sm
              font-medium
              text-muted-foreground
              transition-colors
              hover:text-primary
            "
          >
            Home
          </Link>

          <Link
            href="/privacy"
            className="
              text-sm
              font-medium
              text-muted-foreground
              transition-colors
              hover:text-primary
            "
          >
            Privacy
          </Link>

          <Link
            href="/terms"
            className="
              text-sm
              font-medium
              text-muted-foreground
              transition-colors
              hover:text-primary
            "
          >
            Terms
          </Link>
        </nav>

        {/* Desktop Actions */}
        <div
          className="
            hidden
            items-center
            gap-3
            md:flex
          "
        >
          <ThemeToggle />

          <Link href="/login">
            <Button
              variant="ghost"
              className="
                rounded-xl
              "
            >
              Login
            </Button>
          </Link>

          <Link href="/signup">
            <Button
              className="
                rounded-xl
                bg-linear-to-r
                from-indigo-600
                via-purple-600
                to-cyan-600
              "
            >
              Get Started
            </Button>
          </Link>
        </div>

        {/* Mobile */}
        <div
          className="
            flex
            items-center
            gap-2
            md:hidden
          "
        >
          <ThemeToggle />

          <Sheet>
            <SheetTrigger
              className="
                inline-flex
                h-10
                w-10
                items-center
                justify-center
                rounded-xl
                border
                border-border
                hover:bg-accent
              "
            >
              <Menu className="h-5 w-5" />
            </SheetTrigger>

            <SheetContent
              side="right"
              className="w-72"
            >
              <div
                className="
                  mt-10
                  flex
                  flex-col
                  gap-6
                "
              >
                <Link
                  href="/"
                  className="
                    text-lg
                    font-medium
                  "
                >
                  Home
                </Link>

                <Link
                  href="/privacy"
                  className="
                    text-lg
                    font-medium
                  "
                >
                  Privacy
                </Link>

                <Link
                  href="/terms"
                  className="
                    text-lg
                    font-medium
                  "
                >
                  Terms
                </Link>

                <div
                  className="
                    mt-4
                    flex
                    flex-col
                    gap-3
                  "
                >
                  <Link href="/login">
                    <Button
                      variant="outline"
                      className="
                        w-full
                        rounded-xl
                      "
                    >
                      Login
                    </Button>
                  </Link>

                  <Link href="/signup">
                    <Button
                      className="
                        w-full
                        rounded-xl
                        bg-linear-to-r
                        from-indigo-600
                        via-purple-600
                        to-cyan-600
                      "
                    >
                      Get Started
                    </Button>
                  </Link>
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </header>
  );
}