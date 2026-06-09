/** DemoSection.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Interactive demo section for testing AI lip reading models.
 * @date: 04 June 2026
 * @returns: Demo section component.
 * 
 */

 
// Client Component
"use client";


// Imports
import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Upload, FileVideo, Brain, Sparkles, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/shadcn/button";
import { BackgroundBeams } from "@/components/ui/aceternity/background-beams";


// DemoSection Component
export function DemoSection() {
  // Logic
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [model, setModel] = useState("lipnet");
  const [transcript] = useState("Hello everyone and welcome to LipSpeak AI.");
  const [confidence] = useState("87.4%");

  // Generate a preview URL when a file is selected (safely avoids cascading renders)
  useEffect(() => {
    if (!selectedFile) return;

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);

    // Cleanup function
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  // Render
  return (
    <section className="relative overflow-hidden py-28">
      <BackgroundBeams />

      <div className="relative z-10 mx-auto max-w-7xl px-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mx-auto mb-20 max-w-3xl text-center"
        >
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
            Interactive Product Demo
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
            Try
            <span className="gradient-text block">LipSpeak AI</span>
          </h2>

          <p
            className="
              mt-6
              text-lg
              leading-8
              text-muted-foreground
            "
          >
            Upload a video, choose a model, and experience real-time visual
            speech recognition.
          </p>
        </motion.div>

        {/* Demo Grid */}
        <div className="grid gap-8 lg:grid-cols-2">
          {/* Upload Card */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="
              glass-card
              ai-border
              rounded-3xl
              p-8
            "
          >
            <h3 className="text-2xl font-bold">Upload Video</h3>

            <p className="mt-2 text-muted-foreground">
              Supported formats: MP4, AVI, MOV, MKV
            </p>

            {!selectedFile ? (
              <label
                className="
                  mt-8
                  flex
                  aspect-video
                  cursor-pointer
                  flex-col
                  items-center
                  justify-center
                  rounded-3xl
                  border-2
                  border-dashed
                  border-primary/20
                  bg-primary/5
                  transition-all
                  hover:border-primary/40
                "
              >
                <Upload className="h-12 w-12 text-primary" />

                <p className="mt-4 font-medium">Drag & Drop Video</p>

                <p className="text-sm text-muted-foreground">
                  or click to browse
                </p>

                <input
                  type="file"
                  accept="video/*"
                  className="hidden"
                  onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                />
              </label>
            ) : (
              <div className="mt-8 flex flex-col gap-4">
                <div className="relative aspect-video w-full overflow-hidden rounded-3xl border border-border bg-black/5 dark:bg-black/40">
                  <video
                    src={previewUrl || ""}
                    controls
                    className="h-full w-full object-contain"
                  />
                </div>

                <div className="flex items-center justify-between gap-4">
                  <div className="flex items-center gap-2 truncate text-sm text-muted-foreground">
                    <FileVideo className="h-4 w-4 shrink-0 text-primary" />
                    <span className="truncate">{selectedFile.name}</span>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSelectedFile(null)}
                    className="shrink-0"
                  >
                    Remove Video
                  </Button>
                </div>
              </div>
            )}
          </motion.div>

          {/* Results Card */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="
              glass-card
              ai-border
              rounded-3xl
              p-8
            "
          >
            <h3 className="text-2xl font-bold">Analysis Settings</h3>

            <div className="mt-8">
              <label className="text-sm font-medium">Select Model</label>

              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="
                  mt-2
                  w-full
                  rounded-xl
                  border
                  border-border
                  bg-white
                  px-4
                  py-3
                  text-zinc-900
                  focus:border-primary
                  focus:outline-none
                  focus:ring-1
                  focus:ring-primary
                  dark:bg-zinc-950
                  dark:text-zinc-100
                "
              >
                <option
                  value="lipnet"
                  className="bg-white text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100"
                >
                  LipNet (Fast)
                </option>

                <option
                  value="transformer"
                  className="bg-white text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100"
                >
                  Transformer (Balanced)
                </option>

                <option
                  value="production"
                  className="bg-white text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100"
                >
                  Production Model (Best Accuracy)
                </option>
              </select>
            </div>

            <Button
              disabled={!selectedFile}
              className="
                mt-6
                w-full
                rounded-xl
                bg-linear-to-r
                from-indigo-600
                via-purple-600
                to-cyan-600
                disabled:opacity-50
              "
            >
              <Brain className="mr-2 h-4 w-4" />
              Analyze Video
            </Button>

            {/* Output */}
            <div
              className="
                mt-8
                rounded-3xl
                border
                border-border
                bg-card
                p-6
              "
            >
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-green-500" />

                <span className="font-medium">Prediction Complete</span>
              </div>

              <div className="mt-6">
                <p className="text-sm text-muted-foreground">Transcript</p>

                <p className="mt-2 leading-7">{transcript}</p>
              </div>

              <div className="mt-6 grid grid-cols-2 gap-4">
                <div
                  className="
                    rounded-xl
                    border
                    border-border
                    p-4
                  "
                >
                  <p className="text-sm text-muted-foreground">Confidence</p>

                  <h4 className="mt-2 text-xl font-bold">{confidence}</h4>
                </div>

                <div
                  className="
                    rounded-xl
                    border
                    border-border
                    p-4
                  "
                >
                  <p className="text-sm text-muted-foreground">Model</p>

                  <h4 className="mt-2 text-xl font-bold capitalize">{model}</h4>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
