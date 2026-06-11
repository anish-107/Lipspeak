/**
 * DemoSection.tsx
 * @authors: Anish Kumar, Bidipta Barua,
 * Dibyasmita Hati, Arpan Haldar
 * @description: Interactive homepage demo.
 * @date: 11 June 2026
 * @returns: Demo section component.
 *
 */


"use client";


// Imports
import {
  useState,
} from "react";

import {
  Upload,
  Loader2,
  FileVideo,
  Cpu,
  Zap,
  Network,
} from "lucide-react";

import {
  Button,
} from "@/components/ui/shadcn/button";

import {
  demoService,
} from "@/services/api/demo.service";


// Demo Section
export function DemoSection() {
  const [
    file,
    setFile,
  ] = useState<
    File | null
  >(null);

  const [
    transcript,
    setTranscript,
  ] = useState("");

  const [
    isLoading,
    setIsLoading,
  ] = useState(false);

  const handleUpload =
    async () => {
      if (!file) {
        return;
      }

      try {
        setIsLoading(
          true,
        );

        setTranscript(
          "",
        );

        const response =
          await demoService.transcribe(
            file,
          );

        setTranscript(
          response.transcript,
        );

      } catch (
        error
      ) {
        console.error(
          error,
        );
      } finally {
        setIsLoading(
          false,
        );
      }
    };

  return (
    <section
      className="
        px-6
        py-24
      "
    >
      <div
        className="
          mx-auto
          max-w-6xl
        "
      >
        {/* Header */}
        <div
          className="
            mb-12
            text-center
          "
        >
          <h2
            className="
              text-4xl
              font-black
            "
          >
            Try LipSpeak AI
          </h2>

          <p
            className="
              mt-4
              text-muted-foreground
            "
          >
            Upload a video and
            generate a transcript
            using our live GRID
            inference pipeline.
          </p>
        </div>

        {/* Main Card */}
        <div
          className="
            rounded-3xl
            border
            border-border
            bg-card
            p-8
          "
        >
          {/* Upload Area */}
          <div
            className="
              rounded-3xl
              border
              border-dashed
              border-primary/30
              bg-primary/5
              p-10
            "
          >
            <div
              className="
                flex
                flex-col
                items-center
                gap-6
                text-center
              "
            >
              <div
                className="
                  flex
                  h-20
                  w-20
                  items-center
                  justify-center
                  rounded-full
                  bg-primary/10
                "
              >
                <FileVideo
                  className="
                    h-10
                    w-10
                    text-primary
                  "
                />
              </div>
          
              <div>
                <h3
                  className="
                    text-2xl
                    font-bold
                  "
                >
                  Upload a Video
                </h3>
          
                <p
                  className="
                    mt-2
                    text-muted-foreground
                  "
                >
                  Supported formats:
                  MP4, MPG, MPEG
                </p>
              </div>
          
              <label
                className="
                  cursor-pointer
                "
              >
                <input
                  type="file"
                  accept="
                    .mp4,
                    .mpeg,
                    .mpg
                  "
                  className="hidden"
                  onChange={(event) =>
                    setFile(
                      event.target.files?.[0] ??
                        null
                    )
                  }
                />
          
                <div
                  className="
                    rounded-xl
                    border
                    px-6
                    py-3
                    transition
                    hover:bg-accent
                  "
                >
                  Choose Video
                </div>
              </label>
          
              {file && (
                <div
                  className="
                    rounded-xl
                    border
                    bg-background
                    px-4
                    py-3
                    text-sm
                  "
                >
                  📹 {file.name}
                </div>
              )}
          
              <Button
                onClick={handleUpload}
                disabled={
                  !file ||
                  isLoading
                }
                size="lg"
                className="
                  h-12
                  min-w-[240px]
                  rounded-xl
                  bg-linear-to-r
                  from-indigo-600
                  via-violet-600
                  to-cyan-600
                  font-semibold
                  text-white
                  transition-all
                  hover:scale-[1.02]
                "
              >
                {isLoading ? (
                  <>
                    <Loader2
                      className="
                        mr-2
                        h-5
                        w-5
                        animate-spin
                      "
                    />
                    Running GRID...
                  </>
                ) : (
                  <>
                    <Cpu
                      className="
                        mr-2
                        h-5
                        w-5
                      "
                    />
                    Generate Transcript
                  </>
                )}
              </Button>
          
              <p
                className="
                  text-xs
                  text-muted-foreground
                "
              >
                Powered by GRID • TensorFlow • gRPC
              </p>
            </div>
          </div>

          {/* Feature Cards */}
          <div
            className="
              mt-6
              grid
              gap-4
              md:grid-cols-3
            "
          >
            <div
              className="
                rounded-2xl
                border
                p-4
              "
            >
              <Cpu
                className="
                  mb-2
                  h-5
                  w-5
                "
              />

              <h4
                className="
                  font-semibold
                "
              >
                GRID Model
              </h4>

              <p
                className="
                  text-sm
                  text-muted-foreground
                "
              >
                TensorFlow
                inference
              </p>
            </div>

            <div
              className="
                rounded-2xl
                border
                p-4
              "
            >
              <Network
                className="
                  mb-2
                  h-5
                  w-5
                "
              />

              <h4
                className="
                  font-semibold
                "
              >
                gRPC Pipeline
              </h4>

              <p
                className="
                  text-sm
                  text-muted-foreground
                "
              >
                Dedicated
                inference server
              </p>
            </div>

            <div
              className="
                rounded-2xl
                border
                p-4
              "
            >
              <Zap
                className="
                  mb-2
                  h-5
                  w-5
                "
              />

              <h4
                className="
                  font-semibold
                "
              >
                ~7 Seconds
              </h4>

              <p
                className="
                  text-sm
                  text-muted-foreground
                "
              >
                Current CPU
                inference time
              </p>
            </div>
          </div>

          {/* Transcript */}
          {transcript && (
            <div
              className="
                mt-8
                rounded-3xl
                border
                border-green-500/20
                bg-green-500/5
                p-6
              "
            >
              <div
                className="
                  flex
                  items-center
                  justify-between
                "
              >
                <h3
                  className="
                    text-lg
                    font-bold
                  "
                >
                  Generated Transcript
                </h3>
            
                <span
                  className="
                    rounded-full
                    bg-green-500/10
                    px-3
                    py-1
                    text-xs
                    text-green-500
                  "
                >
                  Success
                </span>
              </div>
            
              <p
                className="
                  mt-4
                  leading-8
                  text-lg
                "
              >
                {transcript}
              </p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}