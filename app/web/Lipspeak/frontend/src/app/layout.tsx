/** Layout.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Root application layout for global providers and metadata.
 * @date: 10 May 2026
 * @returns: Root layout component.
 * 
 */
 
 
// Imports
import type { Metadata } from "next";
import { cookies } from "next/headers"; 
import "./globals.css";
import { ThemeProvider } from "@/components/providers/ThemeProvider";
import { AuthProvider } from "@/providers/AuthProvider";



// Metadata
export const metadata: Metadata = {
  title: "LipSpeak AI",
  description:
    "AI-powered lip reading SaaS platform for real-time and pre-recorded visual speech recognition.",
};


// Props Interface
interface RootLayoutProps {
  children: React.ReactNode;
}


// Root Layout Component (Converted to async to access server cookies)
export default async function RootLayout({
  children,
}: Readonly<RootLayoutProps>) {
  // Logic
  const cookieStore = await cookies();
  const theme = cookieStore.get("theme")?.value || "dark"; 

  // Render
  return (
    <html 
      lang="en" 
      data-scroll-behavior="smooth" 
      className={theme} 
      style={{ colorScheme: theme }}
      suppressHydrationWarning
    >
      <body className="min-h-screen bg-background text-foreground">
        <ThemeProvider>
          <AuthProvider>
            <div className="fixed inset-0 -z-50 grid-background opacity-30" />
  
            <div className="fixed left-1/2 top-0 -z-50 h-125 w-125 -translate-x-1/2 rounded-full hero-glow blur-3xl" />
  
            {children}
          </AuthProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
