/** page.tsx
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Video details page displaying transcript and metadata.
 * @date: 11 June 2026
 * @returns: Video details page component.
 *
 */


"use client";


import {
  useState,
} from "react";

import {
  useParams,
} from "next/navigation";

import {
  Calendar,
  Copy,
  Check,
  FileVideo,
} from "lucide-react";

import {
  Button,
} from "@/components/ui/shadcn/button";

import {
  useVideo,
} from "@/hooks/useVideo";


export default function VideoDetailsPage() {
  /* ---------------------------------------------------------------------- */
  /*                                Params                                  */
  /* ---------------------------------------------------------------------- */

  const params =
    useParams();

  const id =
    params.id as string;

  /* ---------------------------------------------------------------------- */
  /*                                 Hook                                   */
  /* ---------------------------------------------------------------------- */

  const {
    video,
    loading,
  } = useVideo(
    id,
  );

  /* ---------------------------------------------------------------------- */
  /*                                State                                   */
  /* ---------------------------------------------------------------------- */

  const [
    copied,
    setCopied,
  ] = useState(
    false,
  );

  /* ---------------------------------------------------------------------- */
  /*                              Loading                                   */
  /* ---------------------------------------------------------------------- */

  if (
    loading
  ) {
    return (
      <p>
        Loading...
      </p>
    );
  }

  if (
    !video
  ) {
    return (
      <p>
        Video not found.
      </p>
    );
  }

  /* ---------------------------------------------------------------------- */
  /*                               Handlers                                 */
  /* ---------------------------------------------------------------------- */

  const handleCopy =
    async () => {
      await navigator.clipboard.writeText(
        video.transcript,
      );

      setCopied(
        true,
      );

      setTimeout(
        () => {
          setCopied(
            false,
          );
        },
        2000,
      );
    };

  /* ---------------------------------------------------------------------- */
  /*                                Render                                  */
  /* ---------------------------------------------------------------------- */

  return (
    <section className="space-y-8">
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
          </video>
        </div>
      </div>

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
            items-center
            justify-between
          "
        >
          <h2
            className="
              text-2xl
              font-bold
            "
          >
            Transcript
          </h2>

          <Button
            variant="outline"
            onClick={
              handleCopy
            }
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