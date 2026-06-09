/** Footer.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Footer component for public pages.
 * @date: 10 May 2026
 * @returns: Footer component.
 *
 */

 
// Client Component
"use client";


// Imports
import Link from "next/link";
import { ArrowUp } from "lucide-react";
import { Button } from "@/components/ui/shadcn/button";


// Footer Component
export function Footer() {
  // Handlers
  const handleScrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  };

  // Render
  return (
    <footer
      className="
        border-t
        border-border
        bg-background
      "
    >
      <div className="mx-auto max-w-7xl px-6 py-14">
        <div
          className="
            flex
            flex-col
            gap-10
            md:flex-row
            md:items-start
            md:justify-between
          "
        >
          {/* Brand Section */}
          <div className="max-w-md">
            <h3
              className="
                text-2xl
                font-black
                tracking-tight
                bg-linear-to-r
                from-indigo-500
                via-cyan-500
                to-purple-500
                bg-clip-text
                text-transparent
              "
            >
              LipSpeak AI
            </h3>

            <p className="mt-4 text-sm leading-6 text-muted-foreground">
              AI-powered visual speech recognition platform designed to
              transform human communication through cutting-edge lip-reading
              technology and intelligent video analysis.
            </p>
          </div>

          {/* Navigation */}
          <div className="flex flex-col gap-3 text-sm">
            <Link
              href="/"
              className="
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
                text-muted-foreground
                transition-colors
                hover:text-primary
              "
            >
              Privacy Policy
            </Link>

            <Link
              href="/terms"
              className="
                text-muted-foreground
                transition-colors
                hover:text-primary
              "
            >
              Terms of Service
            </Link>

            <Link
              href="/login"
              className="
                text-muted-foreground
                transition-colors
                hover:text-primary
              "
            >
              Login
            </Link>
          </div>
        </div>

        {/* Bottom Section */}
        <div
          className="
            mt-10
            flex
            flex-col
            gap-4
            border-t
            border-border
            pt-6
            text-sm
            md:flex-row
            md:items-center
            md:justify-between
          "
        >
          <p className="text-muted-foreground">
            © {new Date().getFullYear()} LipSpeak AI. All rights reserved.
          </p>

          <Button
            variant="outline"
            size="sm"
            onClick={handleScrollToTop}
            className="
              w-fit
              rounded-xl
              transition-all
              duration-300
              hover:scale-105
            "
          >
            <ArrowUp className="mr-2 h-4 w-4" />
            Back to Top
          </Button>
        </div>
      </div>
    </footer>
  );
}
