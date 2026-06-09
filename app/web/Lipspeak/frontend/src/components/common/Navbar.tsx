/** Navbar.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Public navigation bar component.
 * @date: 10 May 2026
 * @returns: Navbar component.
 *
 */

// Client Component
"use client";


// Imports
import Link from "next/link";
import { buttonVariants } from "@/components/ui/shadcn/button";
import { cn } from "@/lib/utils";
import dynamic from "next/dynamic";
import { Menu } from "lucide-react";
import { Button } from "@/components/ui/shadcn/button";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
} from "@/components/ui/shadcn/sheet";


// Dynamic Imports
const ThemeToggle = dynamic(
  () =>
    import("@/components/common/ThemeToggle").then((mod) => mod.ThemeToggle),
  {
    ssr: false,
  },
);


// Navbar Component
export function Navbar() {
  // Logic

  
  // Render
  return (
    <header
      className="
        sticky
        top-0
        z-50
        border-b
        border-border/50
        bg-background/80
        backdrop-blur-2xl
      "
    >
      <div
        className="
          mx-auto
          flex
          h-20
          max-w-7xl
          items-center
          justify-between
          px-4
          sm:px-6
        "
      >
        {/* Logo */}
        <Link
          href="/"
          className="
            text-xl
            font-black
            tracking-tight
            transition-all
            duration-300
            hover:scale-105
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

        {/* Desktop Navigation */}
        <nav className="hidden items-center gap-8 md:flex">
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
        <div className="hidden items-center gap-3 md:flex">
          <ThemeToggle />

          <Link href="/login">
            <Button
              variant="ghost"
              className="
                rounded-xl
                font-medium
                hover:bg-primary/10
                hover:text-primary
              "
            >
              Login
            </Button>
          </Link>

          <Link href="/signup">
            <Button
              className="
                rounded-xl
                font-medium
                shadow-lg
                shadow-indigo-500/20
                bg-linear-to-r
                from-indigo-600
                via-purple-600
                to-cyan-600
                hover:opacity-90
                transition-all
                duration-300
              "
            >
              Get Started
            </Button>
          </Link>
        </div>

        {/* Mobile Actions */}
        <div className="flex items-center gap-2 md:hidden">
          <ThemeToggle />

          <Sheet>
            <SheetTrigger
              className={cn(
                buttonVariants({ variant: "ghost", size: "icon" }),
                "rounded-xl",
              )}
            >
              <Menu className="h-5 w-5" />
            </SheetTrigger>

            <SheetContent side="right">
              <div className="mt-10 flex flex-col gap-6">
                <Link href="/" className="text-lg font-medium">
                  Home
                </Link>

                <Link href="/privacy" className="text-lg font-medium">
                  Privacy
                </Link>

                <Link href="/terms" className="text-lg font-medium">
                  Terms
                </Link>

                <div className="mt-4 flex flex-col gap-3">
                  <Link href="/login">
                    <Button variant="outline" className="w-full rounded-xl">
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
