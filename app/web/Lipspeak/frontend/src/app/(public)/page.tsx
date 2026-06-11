/** page.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Home landing page for the AI lip reading SaaS platform.
 * @date: 10 May 2026
 * @returns: Public landing page component.
 */

// Imports
import { Navbar } from "@/components/common/Navbar";
import { Footer } from "@/components/common/Footer";
import { HeroSection } from "@/components/common/HeroSection";
import { FeaturesSection } from "@/components/common/FeaturesSection";
import { DemoSection } from "@/components/common/DemoSection";
import { CTASection } from "@/components/common/CTASection";


// HomePage Component
export default function HomePage() {
  // Logic

  // Render
  return (
    <main
      className="
        min-h-screen
        overflow-hidden
        bg-background
        text-foreground
      "
    >
      <Navbar />

      <div className="relative">
        <div
          className="
            absolute
            inset-0
            pointer-events-none
            bg-[radial-gradient(circle_at_top,rgba(99,102,241,0.08),transparent_40%)]
            dark:bg-[radial-gradient(circle_at_top,rgba(99,102,241,0.15),transparent_40%)]
          "
        />

        <HeroSection />
        
        <FeaturesSection />
        
        <DemoSection />
        
        <CTASection />
        
      </div>

      <Footer />
    </main>
  );
}
