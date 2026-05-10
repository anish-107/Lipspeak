// src/components/common/HeroSection.tsx

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Futuristic hero section for landing page.
 * @date: 10 May 2026
 * @returns: Hero section component.
 */

"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/shadcn/button";

export function HeroSection() {
  return (
    <section className="relative flex min-h-[90vh] items-center justify-center px-6 py-24">
      <div className="mx-auto max-w-7xl grid lg:grid-cols-2 gap-16 items-center">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="space-y-8"
        >
          <div className="inline-flex items-center rounded-full border border-indigo-500/30 bg-indigo-500/10 px-4 py-2 text-sm text-indigo-300 backdrop-blur-xl">
            AI-Powered Lip Reading Platform
          </div>

          <div className="space-y-6">
            <h1 className="text-5xl md:text-7xl font-black leading-tight tracking-tight">
              Understand Speech
              <span className="block bg-gradient-to-r from-indigo-400 via-cyan-400 to-purple-500 bg-clip-text text-transparent">
                Without Audio
              </span>
            </h1>

            <p className="max-w-2xl text-lg md:text-xl text-zinc-400 leading-relaxed">
              Real-time and pre-recorded AI lip reading powered by deep
              learning. Analyze speech visually with futuristic precision and
              accessibility-first technology.
            </p>
          </div>

          <div className="flex flex-wrap gap-4">
            <Link href="/signup">
              <Button className="h-12 px-8 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white">
                Get Started
              </Button>
            </Link>

            <Link href="/login">
              <Button
                variant="outline"
                className="h-12 px-8 rounded-xl border-zinc-700 bg-zinc-900/40 hover:bg-zinc-800"
              >
                Live Demo
              </Button>
            </Link>
          </div>

          <div className="flex items-center gap-8 pt-4 text-sm text-zinc-500">
            <div>Real-Time Processing</div>
            <div>Secure AI Pipelines</div>
            <div>Cloud Accelerated</div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.92 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8 }}
          className="relative"
        >
          <div className="absolute inset-0 rounded-[40px] bg-gradient-to-r from-indigo-500/20 to-cyan-500/20 blur-3xl" />

          <div className="relative overflow-hidden rounded-[32px] border border-white/10 bg-zinc-900/70 backdrop-blur-2xl p-6 shadow-2xl">
            <div className="rounded-3xl border border-zinc-800 bg-black p-6">
              <div className="aspect-video rounded-2xl bg-gradient-to-br from-indigo-500/20 via-black to-cyan-500/20 border border-zinc-800 flex items-center justify-center">
                <div className="space-y-4 text-center">
                  <div className="mx-auto h-24 w-24 rounded-full bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center">
                    <div className="h-10 w-10 rounded-full bg-indigo-400 animate-pulse" />
                  </div>

                  <div>
                    <h3 className="text-xl font-semibold">
                      Neural Speech Analysis
                    </h3>

                    <p className="text-sm text-zinc-500 mt-2">
                      Real-time visual speech recognition pipeline
                    </p>
                  </div>
                </div>
              </div>

              <div className="mt-6 grid grid-cols-3 gap-4">
                <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-4">
                  <p className="text-sm text-zinc-500">Accuracy</p>
                  <h4 className="text-2xl font-bold mt-2">97%</h4>
                </div>

                <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-4">
                  <p className="text-sm text-zinc-500">Latency</p>
                  <h4 className="text-2xl font-bold mt-2">40ms</h4>
                </div>

                <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-4">
                  <p className="text-sm text-zinc-500">Models</p>
                  <h4 className="text-2xl font-bold mt-2">AI</h4>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}