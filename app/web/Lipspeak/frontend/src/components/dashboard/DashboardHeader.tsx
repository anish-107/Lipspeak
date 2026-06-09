/** DashboardHeader.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Dashboard top navigation header.
 * @date: 10 June 2026
 * @returns: Dashboard header component.
 *
 */


// Client Component
"use client";


// Imports
import { ThemeToggle } from "@/components/common/ThemeToggle";
import { useAuthStore } from "@/store/auth.store";


// Dashboard Header Component
export function DashboardHeader() {
  // Store
  const user = useAuthStore(
    (state) => state.user,
  );

  // Logic
  const currentHour =
    new Date().getHours();

  const greeting =
    currentHour < 12
      ? "Good Morning"
      : currentHour < 18
      ? "Good Afternoon"
      : "Good Evening";

  const avatarLetter =
    user?.name?.charAt(0)?.toUpperCase() ??
    "U";

  // Render
  return (
    <header
      className="
        sticky
        top-0
        z-40
        border-b
        border-border
        bg-background/80
        backdrop-blur-xl
      "
    >
      <div
        className="
          flex
          h-20
          items-center
          justify-between
          px-4
          md:px-6
          lg:px-8
        "
      >
        {/* Left */}
        <div
          className="
            min-w-0
            flex-1
            pl-12
            lg:pl-0
          "
        >
          <h1
            className="
              truncate
              text-lg
              font-bold
              md:text-xl
            "
          >
            {greeting}
            {user?.name
              ? `, ${user.name}`
              : ""}
          </h1>

          <p
            className="
              text-sm
              text-muted-foreground
            "
          >
            Welcome to your LipSpeak AI dashboard.
          </p>
        </div>

        {/* Right */}
        <div
          className="
            flex
            items-center
            gap-3
          "
        >
          {/* Theme Toggle */}
          <ThemeToggle />

          {/* User Avatar */}
          <div
            className="
              flex
              h-11
              w-11
              items-center
              justify-center
              rounded-full
              bg-linear-to-r
              from-indigo-500
              via-cyan-500
              to-purple-500
              text-sm
              font-black
              text-white
              shadow-lg
            "
          >
            {avatarLetter}
          </div>
        </div>
      </div>
    </header>
  );
}