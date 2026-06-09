/** CTASection.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: CTA section encouraging user signup.
 * @date: 10 May 2026
 * @returns: CTA section component.
 * 
 */

 
// Imports
import Link from "next/link";
import { ArrowRight, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/shadcn/button";
import { BackgroundBeams } from "@/components/ui/aceternity/background-beams";
import { LampContainer } from "../ui/aceternity/lamp";


// CTASection Component
export function CTASection() {
  // Logic
  
  
  // Render
  return (
    <section className="relative overflow-hidden">
      <section className="relative py-32">
        <BackgroundBeams />
        <LampContainer>
          <div
            className="
            relative
            z-10
            mx-auto
            flex
            max-w-5xl
            flex-col
            items-center
            px-6
            text-center
          "
          >
            {/* Badge */}
            <div
              className="
              inline-flex
              items-center
              gap-2
              rounded-full
              border
              border-primary/20
              bg-primary/10
              px-4
              py-2
              text-sm
              font-medium
              text-primary
            "
            >
              <Sparkles className="h-4 w-4" />
              AI-Powered Lip Reading Platform
            </div>

            {/* Heading */}
            <h2
              className="
              mt-8
              max-w-4xl
              text-4xl
              font-black
              leading-tight
              tracking-tight
              md:text-7xl
            "
            >
              Transform Silent Videos Into
              <span
                className="
                block
                bg-linear-to-r
                from-indigo-500
                via-cyan-500
                to-purple-500
                bg-clip-text
                text-transparent
              "
              >
                Actionable Intelligence
              </span>
            </h2>

            {/* Description */}
            <p
              className="
              mt-8
              max-w-3xl
              text-lg
              leading-8
              text-muted-foreground
              md:text-xl
            "
            >
              Unlock speech insights from video using advanced AI-powered lip
              reading. Built for accessibility, research, communication systems,
              and next-generation intelligent applications.
            </p>

            {/* Stats */}
            <div
              className="
              mt-12
              grid
              grid-cols-1
              gap-8
              text-center
              sm:grid-cols-3
            "
            >
              <div>
                <p className="text-4xl font-black">95%+</p>

                <p className="mt-2 text-sm text-muted-foreground">
                  Recognition Accuracy
                </p>
              </div>

              <div>
                <p className="text-4xl font-black">Real-Time</p>

                <p className="mt-2 text-sm text-muted-foreground">
                  AI Processing
                </p>
              </div>

              <div>
                <p className="text-4xl font-black">24/7</p>

                <p className="mt-2 text-sm text-muted-foreground">
                  Cloud Availability
                </p>
              </div>
            </div>

            {/* CTA Buttons */}
            <div
              className="
              mt-12
              flex
              flex-col
              gap-4
              sm:flex-row
            "
            >
              <Link href="/signup">
                <Button
                  size="lg"
                  className="
                  rounded-xl
                  bg-linear-to-r
                  from-indigo-600
                  via-purple-600
                  to-cyan-600
                  px-8
                  shadow-xl
                  shadow-indigo-500/20
                  transition-all
                  duration-300
                  hover:scale-105
                "
                >
                  Get Started Free
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>

              <Link href="/login">
                <Button
                  variant="outline"
                  size="lg"
                  className="
                  rounded-xl
                  px-8
                "
                >
                  Sign In
                </Button>
              </Link>
            </div>

            {/* Footer Text */}
            <p
              className="
              mt-6
              text-sm
              text-muted-foreground
            "
            >
              No credit card required • Free trial available • Setup in minutes
            </p>
          </div>
        </LampContainer>
      </section>
    </section>
  );
}
