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

          {/* Right Content */}
          <motion.div
            initial={{
              opacity: 0,
              x: 40,
            }}
            animate={{
              opacity: 1,
              x: 0,
            }}
            transition={{
              duration: 0.8,
            }}
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
          
            {/* Main Card */}
            <div
              className="
                relative
                overflow-hidden
                rounded-4xl
                border
                border-border
                bg-background/70
                p-8
                backdrop-blur-2xl
                shadow-2xl
              "
            >
              <div
                className="
                  flex
                  flex-col
                  gap-6
                "
              >
                {/* Step 1 */}
                <div
                  className="
                    rounded-2xl
                    border
                    border-border
                    bg-card
                    p-5
                  "
                >
                  <p
                    className="
                      text-xs
                      uppercase
                      tracking-wider
                      text-muted-foreground
                    "
                  >
                    Input Video
                  </p>
          
                  <h3
                    className="
                      mt-2
                      text-lg
                      font-bold
                    "
                  >
                    Silent Visual Speech
                  </h3>
                </div>
          
                {/* Arrow */}
                <div
                  className="
                    flex
                    justify-center
                    text-cyan-500
                    text-2xl
                  "
                >
                  ↓
                </div>
          
                {/* Step 2 */}
                <div
                  className="
                    rounded-2xl
                    border
                    border-cyan-500/20
                    bg-cyan-500/5
                    p-5
                  "
                >
                  <p
                    className="
                      text-xs
                      uppercase
                      tracking-wider
                      text-muted-foreground
                    "
                  >
                    AI Model
                  </p>
          
                  <h3
                    className="
                      mt-2
                      text-lg
                      font-bold
                    "
                  >
                    GRID + TensorFlow
                  </h3>
                </div>
          
                {/* Arrow */}
                <div
                  className="
                    flex
                    justify-center
                    text-cyan-500
                    text-2xl
                  "
                >
                  ↓
                </div>
          
                {/* Output */}
                <div
                  className="
                    rounded-2xl
                    border
                    border-green-500/20
                    bg-green-500/5
                    p-6
                  "
                >
                  <p
                    className="
                      text-xs
                      uppercase
                      tracking-wider
                      text-muted-foreground
                    "
                  >
                    Generated Transcript
                  </p>
          
                  <h2
                    className="
                      mt-3
                      text-3xl
                      font-black
                    "
                  >
                    Hello Everyone
                  </h2>
          
                  <p
                    className="
                      mt-3
                      text-green-500
                    "
                  >
                    Confidence • 87%
                  </p>
                </div>
              </div>
          
              {/* Stats */}
              <div
                className="
                  mt-6
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
                  <p
                    className="
                      text-xs
                      text-muted-foreground
                    "
                  >
                    Accuracy
                  </p>
          
                  <h4
                    className="
                      mt-2
                      text-xl
                      font-black
                    "
                  >
                    87%
                  </h4>
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
                  <p
                    className="
                      text-xs
                      text-muted-foreground
                    "
                  >
                    Latency
                  </p>
          
                  <h4
                    className="
                      mt-2
                      text-xl
                      font-black
                    "
                  >
                    ~7s
                  </h4>
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
                  <p
                    className="
                      text-xs
                      text-muted-foreground
                    "
                  >
                    Model
                  </p>
          
                  <h4
                    className="
                      mt-2
                      text-xl
                      font-black
                    "
                  >
                    GRID
                  </h4>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
