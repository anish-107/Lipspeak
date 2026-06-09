/** page.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Video details page displaying transcript and metadata.
 * @date: 10 June 2026
 * @returns: Video details page component.
 *
 */


// Client Component
"use client";


// Imports
import { useState } from "react";

import {
  Calendar,
  Copy,
  Check,
  FileVideo,
} from "lucide-react";

import { Button } from "@/components/ui/shadcn/button";


// Video Details Page
export default function VideoDetailsPage() {
  /* ---------------------------------------------------------------------- */
  /*                            Mock Data                                   */
  /* ---------------------------------------------------------------------- */

  // TODO:
  // Replace with videoService.getVideoById(id)

  const video = {
    id: 1,

    username: "anish",

    video_link:
      "https://www.w3schools.com/html/mov_bbb.mp4",

    transcript:
      "Welcome to LipSpeak AI. This transcript was generated from a pre-recorded video and is currently mock data until the backend integration is completed.",

    created_at:
      new Date().toISOString(),
  };

  /* ---------------------------------------------------------------------- */
  /*                               State                                    */
  /* ---------------------------------------------------------------------- */

  const [copied, setCopied] =
    useState(false);

  /* ---------------------------------------------------------------------- */
  /*                              Handlers                                  */
  /* ---------------------------------------------------------------------- */

  const handleCopy = async () => {
    await navigator.clipboard.writeText(
      video.transcript,
    );

    setCopied(true);

    setTimeout(() => {
      setCopied(false);
    }, 2000);
  };

  /* ---------------------------------------------------------------------- */
  /*                                Render                                  */
  /* ---------------------------------------------------------------------- */

  return (
    <section className="space-y-8">
      {/* Header */}
      <div>
        <h1
          className="
            text-4xl
            font-black
            tracking-tight
          "
        >
          Video Details
        </h1>

        <p
          className="
            mt-3
            text-muted-foreground
          "
        >
          Review generated transcripts
          and uploaded video details.
        </p>
      </div>

      {/* Metadata */}
      <div
        className="
          glass-card
          ai-border
          rounded-3xl
          p-6
        "
      >
        <div
          className="
            flex
            flex-col
            gap-4
            md:flex-row
            md:items-center
            md:justify-between
          "
        >
          <div>
            <div
              className="
                flex
                items-center
                gap-2
              "
            >
              <FileVideo
                className="
                  h-5
                  w-5
                  text-primary
                "
              />

              <span className="font-semibold">
                Video #{video.id}
              </span>
            </div>

            <p
              className="
                mt-2
                text-muted-foreground
              "
            >
              Uploaded by @{video.username}
            </p>
          </div>

          <div
            className="
              flex
              items-center
              gap-2
              text-sm
              text-muted-foreground
            "
          >
            <Calendar className="h-4 w-4" />

            {new Date(
              video.created_at,
            ).toLocaleString()}
          </div>
        </div>
      </div>

      {/* Video Preview */}
      <div
        className="
          glass-card
          ai-border
          rounded-3xl
          p-6
        "
      >
        <h2
          className="
            text-2xl
            font-bold
          "
        >
          Video Preview
        </h2>

        <div className="mt-6">
          <video
            controls
            className="
              w-full
              rounded-2xl
              border
              border-border
            "
          >
            <source
              src={video.video_link}
              type="video/mp4"
            />

            Your browser does not
            support video playback.
          </video>
        </div>
      </div>

      {/* Transcript */}
      <div
        className="
          glass-card
          ai-border
          rounded-3xl
          p-6
        "
      >
        <div
          className="
            flex
            flex-col
            gap-4
            sm:flex-row
            sm:items-center
            sm:justify-between
          "
        >
          <div>
            <h2
              className="
                text-2xl
                font-bold
              "
            >
              Transcript
            </h2>

            <p
              className="
                mt-2
                text-sm
                text-muted-foreground
              "
            >
              AI-generated transcript.
            </p>
          </div>

          <Button
            variant="outline"
            onClick={handleCopy}
          >
            {copied ? (
              <>
                <Check
                  className="
                    mr-2
                    h-4
                    w-4
                  "
                />

                Copied
              </>
            ) : (
              <>
                <Copy
                  className="
                    mr-2
                    h-4
                    w-4
                  "
                />

                Copy
              </>
            )}
          </Button>
        </div>

        <div
          className="
            mt-6
            rounded-2xl
            border
            border-border
            p-5
          "
        >
          <p
            className="
              whitespace-pre-wrap
              leading-7
            "
          >
            {video.transcript}
          </p>
        </div>
      </div>
    </section>
  );
}