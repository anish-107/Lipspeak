// src/components/dashboard/Sidebar.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Sidebar navigation for dashboard routes.
 * @date: 10 May 2026
 * @returns: Sidebar component.
 */

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Camera,
  Video,
  LogOut,
} from "lucide-react";

const links = [
  {
    label: "Dashboard",
    href: "/dashboard",
    icon: LayoutDashboard,
  },
  {
    label: "Real-Time",
    href: "/dashboard/real-time",
    icon: Camera,
  },
  {
    label: "Pre-Recorded",
    href: "/dashboard/pre-recorded",
    icon: Video,
  },
];

export function Sidebar() {
  const pathname = usePathname();

  const handleLogout = () => {
    document.cookie =
      "token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";

    window.location.href = "/login";
  };

  return (
    <aside className="hidden min-h-screen w-[280px] border-r border-white/10 bg-zinc-950/80 backdrop-blur-2xl lg:block">
      <div className="flex h-24 items-center border-b border-white/10 px-8">
        <h2 className="text-2xl font-black gradient-text">
          LipSpeak AI
        </h2>
      </div>

      <div className="flex flex-col justify-between p-6">
        <nav className="space-y-3">
          {links.map((link) => {
            const Icon = link.icon;

            const active = pathname === link.href;

            return (
              <Link
                key={link.href}
                href={link.href}
                className={`flex items-center gap-4 rounded-2xl px-5 py-4 transition-all duration-300 ${
                  active
                    ? "bg-indigo-600 text-white"
                    : "text-zinc-400 hover:bg-zinc-900 hover:text-white"
                }`}
              >
                <Icon className="h-5 w-5" />

                <span className="font-medium">
                  {link.label}
                </span>
              </Link>
            );
          })}
        </nav>

        <button
          onClick={handleLogout}
          className="mt-10 flex items-center gap-4 rounded-2xl border border-red-500/20 bg-red-500/10 px-5 py-4 text-red-300 transition-all hover:bg-red-500/20"
        >
          <LogOut className="h-5 w-5" />

          <span className="font-medium">
            Logout
          </span>
        </button>
      </div>
    </aside>
  );
}