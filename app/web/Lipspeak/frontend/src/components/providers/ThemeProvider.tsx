/** ThemeProvider.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Application theme provider component.
 * @date: 04 June 2026
 * @returns: ThemeProvider component.
 * 
 */


// Client Component
"use client";


// Imports
import { ThemeProvider as NextThemesProvider } from "next-themes";


// ThemeProvider Component
export function ThemeProvider({
  children,
}: {
  children: React.ReactNode;
  }) {
  // Logic
  
  
  // Render
  return (
    <NextThemesProvider
      attribute="class"
      defaultTheme="dark"
      enableSystem={false}
    >
      {children}
    </NextThemesProvider>
  );
}