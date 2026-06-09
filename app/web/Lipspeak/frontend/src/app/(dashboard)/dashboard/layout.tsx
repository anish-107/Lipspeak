/** layout.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Dashboard layout with responsive sidebar and header.
 * @date: 10 June 2026
 * @returns: Dashboard layout component.
 *
 */


// Imports
import { DashboardSidebar } from "@/components/dashboard/DashboardSidebar";
import { DashboardHeader } from "@/components/dashboard/DashboardHeader";


// Props Interface
interface DashboardLayoutProps {
  children: React.ReactNode;
}


// Dashboard Layout
export default function DashboardLayout({
  children,
}: Readonly<DashboardLayoutProps>) {
  // Render
  return (
    <div
      className="
        min-h-screen
        bg-background
        text-foreground
      "
    >
      <div className="flex">
        {/* Desktop Sidebar */}
        <DashboardSidebar />

        {/* Content */}
        <div className="min-w-0 flex-1">
          <DashboardHeader />

          <main
            className="
              mx-auto
              max-w-7xl
              p-4
              md:p-6
              lg:p-8
            "
          >
            {children}
          </main>
        </div>
      </div>
    </div>
  );
}