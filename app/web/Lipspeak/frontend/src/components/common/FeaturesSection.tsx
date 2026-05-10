// src/components/common/FeaturesSection.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Features showcase section for landing page.
 * @date: 10 May 2026
 * @returns: Features section component.
 */

"use client";

import { motion } from "framer-motion";
import { Brain, Camera, Video, Shield } from "lucide-react";

const features = [
  {
    title: "Real-Time Prediction",
    description:
      "Analyze live webcam feeds instantly using optimized AI inference pipelines.",
    icon: Camera,
  },
  {
    title: "Pre-Recorded Analysis",
    description:
      "Upload videos and generate detailed lip-reading predictions effortlessly.",
    icon: Video,
  },
  {
    title: "Deep Learning Engine",
    description:
      "Powered by state-of-the-art neural architectures for maximum accuracy.",
    icon: Brain,
  },
  {
    title: "Secure Infrastructure",
    description:
      "Enterprise-grade security with encrypted processing and authentication.",
    icon: Shield,
  },
];

export function FeaturesSection() {
  return (
    <section className="px-6 py-28">
      <div className="mx-auto max-w-7xl">
        <div className="text-center max-w-3xl mx-auto mb-20">
          <h2 className="text-4xl md:text-5xl font-black">
            Advanced AI Capabilities
          </h2>

          <p className="mt-6 text-zinc-400 text-lg">
            Built with modern deep learning systems for scalable and accurate
            lip-reading intelligence.
          </p>
        </div>

        <div className="grid md:grid-cols-2 xl:grid-cols-4 gap-8">
          {features.map((feature, index) => {
            const Icon = feature.icon;

            return (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                className="rounded-3xl border border-white/10 bg-zinc-900/60 p-8 backdrop-blur-xl"
              >
                <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-indigo-500/10 border border-indigo-500/20">
                  <Icon className="h-6 w-6 text-indigo-400" />
                </div>

                <h3 className="mt-6 text-2xl font-bold">
                  {feature.title}
                </h3>

                <p className="mt-4 text-zinc-400 leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}