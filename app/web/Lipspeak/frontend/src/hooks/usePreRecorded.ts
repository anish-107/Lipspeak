/** usePreRecorded.ts
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: State management hook for pre-recorded video analysis.
 * @date: 10 June 2026
 * @returns: Pre-recorded analysis state and actions.
 *
 */


// Client Hook
"use client";


// Imports
import { useState } from "react";
import {
  videoService,
} from "@/services/api/video.service";


// Pre Recorded Hook
export function usePreRecorded() {
  /* ---------------------------------------------------------------------- */
  /*                                State                                   */
  /* ---------------------------------------------------------------------- */

  const [videoFile, setVideoFile] =
    useState<File | null>(null);

  const [videoUrl, setVideoUrl] =
    useState("");

  const [transcript, setTranscript] =
    useState("");

  const [loading, setLoading] =
    useState(false);

  /* ---------------------------------------------------------------------- */
  /*                           Video Selection                              */
  /* ---------------------------------------------------------------------- */

  const handleVideoSelect = (
    file: File,
  ) => {
    setVideoFile(file);

    const url =
      URL.createObjectURL(file);

    setVideoUrl(url);

    setTranscript("");
  };

  /* ---------------------------------------------------------------------- */
  /*                           Process Video                                */
  /* ---------------------------------------------------------------------- */

  const processVideo =
    async () => {
      if (
        !videoFile
      ) {
        return;
      }
  
      try {
        setLoading(
          true,
        );
  
        const response =
          await videoService.uploadVideo(
            videoFile,
          );
  
        setTranscript(
          response.transcript,
        );
  
      } catch (error) {
        console.error(
          error,
        );
  
        setTranscript(
          "Failed to process video.",
        );
  
      } finally {
        setLoading(
          false,
        );
      }
    };

  /* ---------------------------------------------------------------------- */
  /*                                Return                                  */
  /* ---------------------------------------------------------------------- */

  return {
    videoFile,

    videoUrl,

    transcript,

    loading,

    handleVideoSelect,

    processVideo,
  };
}