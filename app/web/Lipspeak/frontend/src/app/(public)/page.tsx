/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Home landing page for the AI lip reading SaaS platform.
 * @date: 10 May 2026
 * @returns: Public landing page component.
 */

import { HeroSection } from "@/components/common/HeroSection";
import { FeaturesSection } from "@/components/common/FeaturesSection";
import { StatsSection } from "@/components/common/StatsSection";
import { CTASection } from "@/components/common/CTASection";
import { Navbar } from "@/components/common/Navbar";
import { Footer } from "@/components/common/Footer";

export default function HomePage() {
  return (
    <main className="min-h-screen bg-black text-white overflow-hidden">
      <Navbar />

      <div className="relative">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(99,102,241,0.15),transparent_40%)] pointer-events-none" />

        <HeroSection />
        <FeaturesSection />
        <StatsSection />
        <CTASection />
      </div>

      <Footer />
    </main>
  );
}