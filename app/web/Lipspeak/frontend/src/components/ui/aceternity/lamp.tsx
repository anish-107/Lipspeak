/** Lamp.tsx
* @authors: Aceternity UI
* @adapted_by: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
* @description: Animated lamp spotlight container component used to highlight
* content with gradient lighting effects and motion animations.
* @date: 04 June 2026
* @returns: LampContainer component.
*
*/

// Client Component
"use client";

// Imports
import React from "react";
import { motion } from "motion/react";
import { cn } from "@/lib/utils";

// Props Interface
interface LampContainerProps {
  children: React.ReactNode;
  className?: string;
}

// LampContainer Component
export const LampContainer = ({ children, className }: LampContainerProps) => {
  // Render
  return (
    <div
      className={cn(
        "relative flex min-h-[500px] md:min-h-[700px] w-full flex-col items-center justify-center overflow-hidden rounded-3xl bg-background z-0",
        className
      )}
    >
      <div className="relative flex w-full flex-1 scale-y-125 items-center justify-center isolate z-0">
        
        {/* Left Beam */}
        <motion.div
          initial={{ opacity: 0.5, width: "15rem" }}
          whileInView={{ opacity: 1, width: "30rem" }}
          transition={{
            delay: 0.3,
            duration: 0.8,
            ease: "easeInOut",
          }}
          style={{
            backgroundImage:
              "conic-gradient(var(--conic-position), var(--tw-gradient-stops))",
          }}
          className="absolute right-1/2 h-56 w-[18rem] md:w-[30rem] overflow-visible bg-gradient-conic from-cyan-500 via-transparent to-transparent [--conic-position:from_70deg_at_center_top]"
        >
          <div className="absolute bottom-0 left-0 z-20 h-40 w-full bg-background mask-[linear-gradient(to_top,white,transparent)]" />
          <div className="absolute bottom-0 left-0 z-20 h-full w-40 bg-background mask-[linear-gradient(to_right,white,transparent)]" />
        </motion.div>

        {/* Right Beam */}
        <motion.div
          initial={{ opacity: 0.5, width: "15rem" }}
          whileInView={{ opacity: 1, width: "30rem" }}
          transition={{
            delay: 0.3,
            duration: 0.8,
            ease: "easeInOut",
          }}
          style={{
            backgroundImage:
              "conic-gradient(var(--conic-position), var(--tw-gradient-stops))",
          }}
          className="absolute left-1/2 h-56 w-[18rem] md:w-[30rem] bg-gradient-conic from-transparent via-transparent to-cyan-500 [--conic-position:from_290deg_at_center_top]"
        >
          <div className="absolute bottom-0 right-0 z-20 h-full w-40 bg-background mask-[linear-gradient(to_left,white,transparent)]" />
          <div className="absolute bottom-0 right-0 z-20 h-40 w-full bg-background mask-[linear-gradient(to_top,white,transparent)]" />
        </motion.div>

        {/* Glow */}
        <div className="absolute top-1/2 h-48 w-full translate-y-12 scale-x-150 bg-background blur-2xl" />
        <div className="absolute top-1/2 z-50 h-48 w-full bg-transparent opacity-10 backdrop-blur-md" />
        <div className="absolute z-50 h-36 w-md -translate-y-1/2 rounded-full bg-cyan-500/40 blur-3xl" />

        <motion.div
          initial={{ width: "8rem" }}
          whileInView={{ width: "16rem" }}
          transition={{
            delay: 0.3,
            duration: 0.8,
            ease: "easeInOut",
          }}
          className="absolute z-30 h-36 w-64 -translate-y-24 rounded-full bg-cyan-400/40 blur-2xl"
        />

        <motion.div
          initial={{ width: "15rem" }}
          whileInView={{ width: "30rem" }}
          transition={{
            delay: 0.3,
            duration: 0.8,
            ease: "easeInOut",
          }}
          className="absolute z-50 h-0.5 w-120 -translate-y-28 bg-cyan-400"
        />
        
        {/* Note: Masking div deleted here to prevent hidden content */}
      </div>

      {/* Content */}
      <div className="relative z-50 flex flex-col items-center px-5">
        {children}
      </div>
    </div>
  );
};