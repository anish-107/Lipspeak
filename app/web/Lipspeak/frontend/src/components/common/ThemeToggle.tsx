/** ThemeToggle.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Theme switch button component.
 * @date: 04 June 2026
 * @returns: ThemeToggle component.
 *
 */

 
// Client Component
"use client";


// Imports
import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/shadcn/button";


// ThemeToggle Component
export function ThemeToggle() {
  // Theme
  const { resolvedTheme, setTheme } = useTheme();

  // Logic
  const isDark = resolvedTheme === "dark";

  // Render
  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      className="
        rounded-xl
        border
        border-border
        hover:bg-accent
        transition-all
        duration-300
      "
      aria-label="Toggle Theme"
    >
      {isDark ? (
        <Sun className="h-5 w-5 text-yellow-500" />
      ) : (
        <Moon className="h-5 w-5 text-indigo-600" />
      )}
    </Button>
  );
}
