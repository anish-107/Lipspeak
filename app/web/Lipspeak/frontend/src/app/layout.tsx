// src/app/layout.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Root application layout for global providers and metadata.
 * @date: 10 May 2026
 * @returns: Root layout component.
 */

import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LipSpeak AI",
  description:
    "AI-powered lip reading SaaS platform for real-time and pre-recorded visual speech recognition.",
};

interface RootLayoutProps {
  children: React.ReactNode;
}

export default function RootLayout({
  children,
}: Readonly<RootLayoutProps>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-black text-white">
        <div className="fixed inset-0 -z-50 grid-background opacity-30" />

        <div className="fixed left-1/2 top-0 -z-50 h-[500px] w-[500px] -translate-x-1/2 rounded-full hero-glow blur-3xl" />

        {children}
      </body>
    </html>
  );
}