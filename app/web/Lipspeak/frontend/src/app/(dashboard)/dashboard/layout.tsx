// src/app/(dashboard)/dashboard/layout.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Dashboard layout with sidebar and header.
 * @date: 10 May 2026
 * @returns: Dashboard layout component.
 */

import { Sidebar } from "@/components/dashboard/Sidebar";
import { DashboardHeader } from "@/components/dashboard/DashboardHeader";

interface DashboardLayoutProps {
  children: React.ReactNode;
}

export default function DashboardLayout({
  children,
}: Readonly<DashboardLayoutProps>) {
  return (
    <div className="min-h-screen bg-black text-white">
      <div className="flex">
        <Sidebar />

        <div className="flex-1">
          <DashboardHeader />

          <main className="p-6 md:p-10">
            {children}
          </main>
        </div>
      </div>
    </div>
  );
}