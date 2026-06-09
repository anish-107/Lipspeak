/** HeroSection.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Futuristic hero section for landing page.
 * @date: 10 May 2026
 * @returns: Hero section component.
 *
 */

 
// Client Component
"use client";


// Imports
import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, CheckCircle2, PlayCircle, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/shadcn/button";
import { BackgroundBeams } from "@/components/ui/aceternity/background-beams";
import { LampContainer } from "../ui/aceternity/lamp";


// HeroSection Component
export function HeroSection() {
  // Logic

  
  // Render
  return (
    <section
      className="
        relative
        overflow-hidden
        px-6
        pt-24
        pb-16
        md:pt-32
        md:pb-24
      "
    >
      {/* Background */}

      <BackgroundBeams />

      <div className="relative z-10 mx-auto max-w-7xl">
        <div
          className="
            grid
            items-center
            gap-12
            lg:grid-cols-2
            lg:gap-20
          "
        >
          {/* Left Content */}
          <LampContainer>
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7 }}
              className="space-y-8"
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
                backdrop-blur-xl
              "
              >
                <Sparkles className="h-4 w-4" />
                AI-Powered Lip Reading Platform
              </div>

              {/* Heading */}
              <div className="space-y-6">
                <h1
                  className="
                  text-4xl
                  font-black
                  leading-tight
                  tracking-tight
                  sm:text-5xl
                  lg:text-7xl
                "
                >
                  Understand Speech
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
                    Without Audio
                  </span>
                </h1>

                <p
                  className="
                  max-w-2xl
                  text-base
                  leading-8
                  text-muted-foreground
                  sm:text-lg
                  lg:text-xl
                "
                >
                  Transform silent video into meaningful speech predictions
                  using advanced AI-powered visual speech recognition. Built for
                  accessibility, research, surveillance analysis, and next
                  generation communication systems.
                </p>
              </div>

              {/* Buttons */}
              <div
                className="
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
                    w-full
                    rounded-xl
                    bg-linear-to-r
                    from-indigo-600
                    via-purple-600
                    to-cyan-600
                    shadow-xl
                    shadow-indigo-500/20
                    transition-all
                    duration-300
                    hover:scale-105
                    sm:w-auto
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
                    w-full
                    rounded-xl
                    sm:w-auto
                  "
                  >
                    <PlayCircle className="mr-2 h-4 w-4" />
                    Watch Demo
                  </Button>
                </Link>
              </div>

              {/* Trust Indicators */}
              <div
                className="
                flex
                flex-wrap
                gap-4
                pt-2
                text-sm
                text-muted-foreground
              "
              >
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  Real-Time Processing
                </div>

                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  Secure AI Pipelines
                </div>

                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  Cloud Accelerated
                </div>
              </div>
            </motion.div>
          </LampContainer>

          {/* Right Content */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8 }}
            className="relative"
          >
            {/* Glow */}
            <div
              className="
                absolute
                inset-0
                rounded-[40px]
                bg-linear-to-r
                from-indigo-500/20
                via-cyan-500/20
                to-purple-500/20
                blur-3xl
              "
            />

            {/* Product Card */}
            <div
              className="
                relative
                overflow-hidden
                rounded-4xl
                border
                border-border
                bg-background/70
                p-4
                backdrop-blur-2xl
                shadow-2xl
              "
            >
              {/* Demo Video */}
              <div className="relative overflow-hidden rounded-3xl">
                <video
                  autoPlay
                  muted
                  loop
                  playsInline
                  className="
                    aspect-video
                    w-full
                    object-cover
                  "
                >
                  <source src="/videos/lipspeak-demo.mp4" type="video/mp4" />
                </video>

                {/* Overlay */}
                <div
                  className="
                    absolute
                    inset-0
                    bg-linear-to-t
                    from-black/70
                    via-black/10
                    to-transparent
                  "
                />

                {/* Prediction Box */}
                <div
                  className="
                    absolute
                    bottom-4
                    left-4
                    rounded-2xl
                    border
                    border-white/10
                    bg-black/50
                    px-4
                    py-3
                    backdrop-blur-xl
                  "
                >
                  <p className="text-xs text-zinc-400">Predicted Speech</p>

                  <p className="mt-1 font-medium text-white">Hello everyone</p>

                  <p className="mt-1 text-xs text-green-400">
                    Confidence • 87%
                  </p>
                </div>
              </div>

              {/* Stats */}
              <div
                className="
                  mt-4
                  grid
                  grid-cols-3
                  gap-3
                "
              >
                <div
                  className="
                    rounded-2xl
                    border
                    border-border
                    bg-card
                    p-4
                  "
                >
                  <p className="text-xs text-muted-foreground">Accuracy</p>

                  <h4 className="mt-2 text-xl font-black">87%</h4>
                </div>

                <div
                  className="
                    rounded-2xl
                    border
                    border-border
                    bg-card
                    p-4
                  "
                >
                  <p className="text-xs text-muted-foreground">Latency</p>

                  <h4 className="mt-2 text-xl font-black">1500ms</h4>
                </div>

                <div
                  className="
                    rounded-2xl
                    border
                    border-border
                    bg-card
                    p-4
                  "
                >
                  <p className="text-xs text-muted-foreground">Models</p>

                  <h4 className="mt-2 text-xl font-black">AI</h4>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
