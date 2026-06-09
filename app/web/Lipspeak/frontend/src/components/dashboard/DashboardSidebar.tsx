/** DashboardSidebar.tsx
* @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
* @description: Responsive dashboard sidebar navigation.
* @date: 10 June 2026
* @returns: Dashboard sidebar component.
* 
*/

// Client Component
"use client";

// Imports
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";
import {
  LayoutDashboard,
  Video,
  Camera,
  LogOut,
  Menu,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/shadcn/button";
import { useAuthStore } from "@/store/auth.store";
import type { User } from "@/types/auth.types";

// Navigation Links
const links = [
  {
    label: "Dashboard",
    href: "/dashboard",
    icon: LayoutDashboard,
  },
  {
    label: "Pre-Recorded",
    href: "/dashboard/pre-recorded",
    icon: Video,
  },
  {
    label: "Real-Time",
    href: "/dashboard/real-time",
    icon: Camera,
  },
];

// Sidebar Content Props
interface SidebarContentProps {
  pathname: string;
  user: User | null;
  onLogout: () => void;
  onNavigate?: () => void;
}

// Sidebar Content Component
function SidebarContent({
  pathname,
  user,
  onLogout,
  onNavigate,
}: Readonly<SidebarContentProps>) {
  return (
    <div className="flex h-full flex-col justify-between">
      {/* Top Section */}
      <div>
        <div className="p-6 border-b border-border">
          <h1 className="text-xl font-bold tracking-tight text-foreground">
            LipSpeak AI
          </h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Visual Speech Recognition
          </p>
        </div>

        {/* Navigation links */}
        <nav className="p-4">
          <div className="space-y-2">
            {links.map((link) => {
              const Icon = link.icon;
              const active = pathname === link.href;

              return (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={onNavigate}
                  className={`flex items-center gap-3 rounded-2xl px-4 py-3 transition-all duration-300 ${
                    active
                      ? "bg-primary text-primary-foreground"
                      : "hover:bg-muted text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <Icon className="h-5 w-5" />
                  <span className="font-medium">{link.label}</span>
                </Link>
              );
            })}
          </div>
        </nav>
      </div>

      {/* Bottom Section */}
      <div className="border-t border-border p-4">
        <div className="mb-4 rounded-2xl border border-border p-4">
          <p className="text-sm text-muted-foreground">Logged In As</p>
          <p className="mt-1 font-semibold text-foreground">
            {user?.name ?? "User"}
          </p>
          <p className="text-sm text-muted-foreground">
            @{user?.username ?? "guest"}
          </p>
        </div>

        <Button
          variant="outline"
          onClick={onLogout}
          className="w-full justify-start rounded-2xl"
        >
          <LogOut className="mr-2 h-4 w-4" />
          Logout
        </Button>
      </div>
    </div>
  );
}

// Main Dashboard Sidebar Component
export function DashboardSidebar() {
  const router = useRouter();
  const pathname = usePathname();
  const user = useAuthStore((state) => state.user);
  const logout = useAuthStore((state) => state.logout);
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleLogout = () => {
    logout();
    router.push("/login");
  };

  return (
    <>
      {/* Mobile Menu Toggle Button */}
      <button
        onClick={() => setMobileOpen(true)}
        className="fixed left-4 top-4 z-50 rounded-xl border border-border bg-background p-2 shadow-lg lg:hidden text-foreground hover:bg-muted"
      >
        <Menu className="h-5 w-5" />
      </button>

      {/* Desktop Sidebar Layout */}
      <aside className="hidden h-screen w-72 border-r border-border bg-background/80 backdrop-blur-xl lg:block fixed left-0 top-0">
        <SidebarContent
          pathname={pathname}
          user={user}
          onLogout={handleLogout}
        />
      </aside>

      {/* Mobile Drawer Navigation */}
      {mobileOpen && (
        <>
          {/* Backdrop Overlay */}
          <div
            className="fixed inset-0 z-50 bg-black/50 lg:hidden backdrop-blur-sm"
            onClick={() => setMobileOpen(false)}
          />

          {/* Drawer Element */}
          <aside className="fixed left-0 top-0 z-50 h-screen w-72 border-r border-border bg-background lg:hidden flex flex-col">
            <div className="flex justify-end p-4">
              <button 
                onClick={() => setMobileOpen(false)}
                className="p-1 rounded-lg hover:bg-muted text-muted-foreground hover:text-foreground"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto">
              <SidebarContent
                pathname={pathname}
                user={user}
                onLogout={handleLogout}
                onNavigate={() => setMobileOpen(false)}
              />
            </div>
          </aside>
        </>
      )}
    </>
  );
}