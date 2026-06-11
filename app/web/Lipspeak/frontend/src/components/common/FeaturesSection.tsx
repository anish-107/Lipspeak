/** FeaturesSection.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Features showcase section for landing page.
 * @date: 10 May 2026
 * @returns: Features section component.
 *
 */

 
// Client Component
"use client";


// Imports
import { motion } from "framer-motion";
import {
  ArrowRight,
  Brain,
  Camera,
  Shield,
  Video,
  MessageSquareText,
} from "lucide-react";



// FeaturesSection Component
export function FeaturesSection() {
  // Logic

  
  // Render
  return (
      <section className="relative z-50 w-full px-6 py-28">
        {" "}
        <div className="mx-auto max-w-7xl">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="relative z-50 mx-auto mb-20 max-w-3xl text-center"
          >
            {" "}
            <div
              className="
             inline-flex
             items-center
             rounded-full
             border
             border-primary/20
             bg-primary/10
             px-4
             py-2
             text-sm
             font-semibold
             text-primary
           "
            >
              Advanced AI Capabilities{" "}
            </div>
            <h2
              className="
            mt-6
            text-4xl
            font-black
            tracking-tight
            md:text-6xl
          "
            >
              How
              <span className="gradient-text block">LipSpeak AI Works</span>
            </h2>
            <p
              className="
            mt-6
            text-lg
            leading-8
            text-muted-foreground
          "
            >
              A powerful visual speech recognition pipeline built using deep
              learning, computer vision, and secure cloud infrastructure.
            </p>
          </motion.div>

          {/* Features Grid */}
          <div className="grid gap-8 md:grid-cols-3">
            {/* Pipeline Card */}
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="
            glass-card
            ai-border
            rounded-3xl
            p-8
            md:col-span-2
            transition-all
            duration-300
            hover:-translate-y-1
          "
            >
              <div className="flex h-full flex-col justify-between">
                <div>
                  <h3 className="text-2xl font-bold">
                    Visual Speech Recognition Pipeline
                  </h3>

                  <p className="mt-3 text-muted-foreground">
                    Every frame is processed through a specialized AI pipeline
                    to convert visual speech patterns into meaningful text.
                  </p>
                </div>

                <div className="mt-12">
                  <div className="grid grid-cols-4 gap-4">
                    {[
                      {
                        icon: Camera,
                        label: "Video",
                      },
                      {
                        icon: Video,
                        label: "Face",
                      },
                      {
                        icon: Brain,
                        label: "AI",
                      },
                      {
                        icon: MessageSquareText,
                        label: "Speech",
                      },
                    ].map((step, index) => {
                      const Icon = step.icon;

                      return (
                        <div key={step.label} className="flex items-center">
                          <div className="flex flex-1 flex-col items-center">
                            <div
                              className="
                            relative
                            flex
                            h-16
                            w-16
                            items-center
                            justify-center
                            rounded-2xl
                            border
                            border-primary/20
                            bg-linear-to-br
                            from-indigo-500/20
                            to-cyan-500/20
                          "
                            >
                              <div className="absolute inset-0 rounded-2xl bg-primary/10 blur-xl" />

                              <Icon className="relative z-10 h-7 w-7 text-primary" />
                            </div>

                            <p className="mt-3 text-sm font-medium">
                              {step.label}
                            </p>
                          </div>

                          {index !== 3 && (
                            <ArrowRight
                              className="
                            hidden
                            md:block
                            h-5
                            w-5
                            text-muted-foreground
                          "
                            />
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Security Card */}
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 }}
              className="
            glass-card
            ai-border
            rounded-3xl
            p-8
            transition-all
            duration-300
            hover:-translate-y-1
          "
            >
              <div
                className="
              relative
              flex
              h-24
              w-24
              items-center
              justify-center
              rounded-3xl
              bg-linear-to-br
              from-indigo-500/20
              to-cyan-500/20
            "
              >
                <div className="absolute inset-0 rounded-3xl bg-primary/10 blur-2xl" />

                <Shield className="relative z-10 h-12 w-12 text-primary" />
              </div>

              <h3 className="mt-8 text-2xl font-bold">Secure Infrastructure</h3>

              <p className="mt-4 text-muted-foreground">
                Enterprise-grade authentication, encrypted processing, and
                privacy-first architecture.
              </p>
            </motion.div>

            {/* Neural Network Card */}
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
              className="
            glass-card
            ai-border
            rounded-3xl
            p-8
            transition-all
            duration-300
            hover:-translate-y-1
          "
            >
              <div className="relative h-40">
                <div className="absolute left-10 top-10 h-3 w-3 rounded-full bg-primary" />
                <div className="absolute left-20 top-20 h-3 w-3 rounded-full bg-primary" />
                <div className="absolute left-10 top-32 h-3 w-3 rounded-full bg-primary" />

                <div className="absolute right-10 top-16 h-3 w-3 rounded-full bg-cyan-400" />
                <div className="absolute right-10 top-28 h-3 w-3 rounded-full bg-cyan-400" />

                <div className="absolute left-1/2 top-20 h-4 w-4 -translate-x-1/2 rounded-full bg-purple-500" />

                <div className="absolute left-12 top-12 h-px w-24 rotate-12 bg-primary/50" />
                <div className="absolute left-12 top-32 h-px w-24 -rotate-12 bg-primary/50" />
                <div className="absolute right-12 top-24 h-px w-20 -rotate-12 bg-cyan-400/50" />
              </div>

              <h3 className="text-2xl font-bold">Deep Learning Engine</h3>

              <p className="mt-4 text-muted-foreground">
                Specialized neural architectures trained for visual speech
                recognition and contextual inference.
              </p>
            </motion.div>

            {/* Video Analysis Card */}
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3 }}
              className="
            glass-card
            ai-border
            rounded-3xl
            p-8
            md:col-span-2
            transition-all
            duration-300
            hover:-translate-y-1
            hover:shadow-2xl
          "
            >
              <div
                className="
              relative
              overflow-hidden
              rounded-3xl
              border
              border-border
            "
              >
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

                <div
                  className="
                absolute
                inset-0
                bg-linear-to-t
                from-black/80
                via-black/10
                to-transparent
              "
                />

                <div
                  className="
                absolute
                left-4
                top-4
                rounded-full
                border
                border-green-500/30
                bg-green-500/20
                px-3
                py-1
                text-xs
                font-medium
                text-green-400
                backdrop-blur-md
              "
                >
                  ● LIVE ANALYSIS
                </div>

                <div
                  className="
                absolute
                right-4
                top-4
                rounded-full
                border
                border-primary/20
                bg-background/70
                px-3
                py-1
                text-xs
                font-medium
                backdrop-blur-md
              "
                >
                  AI Processing
                </div>

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

              <h3 className="mt-8 text-2xl font-bold">
                See LipSpeak In Action
              </h3>

              <p className="mt-4 text-muted-foreground">
                Watch LipSpeak AI analyze visual speech patterns and generate
                real-time predictions from uploaded video content.
              </p>
            </motion.div>
          </div>
        </div>
      </section>
  );
}
