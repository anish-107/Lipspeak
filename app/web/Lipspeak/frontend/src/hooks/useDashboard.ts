/** useDashboard.ts
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Dashboard data management hook.
 * @date: 10 June 2026
 * @returns: Dashboard state and actions.
 *
 */


// Client Hook
"use client";


// Imports
import { useEffect, useState } from "react";

// import { videoService } from "@/services/api/video.service";

import type {
  Video,
} from "@/types/video.types";


// Dashboard Hook
export function useDashboard() {
  /* ---------------------------------------------------------------------- */
  /*                                State                                   */
  /* ---------------------------------------------------------------------- */

  const [videos, setVideos] =
    useState<Video[]>([]);

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");


  /* ---------------------------------------------------------------------- */
  /*                             Load Videos                                */
  /* ---------------------------------------------------------------------- */

  
  const fetchVideos = async () => {
    try {
      setLoading(true);

      // const data =
      //   await videoService.getVideos();

      const data = [
        {
          id: 1,
          username: "anish",
          video_link: "",
          transcript:
            "Welcome to LipSpeak AI",
          created_at:
            new Date().toISOString(),
        },
        {
          id: 2,
          username: "anish",
          video_link: "",
          transcript:
            "Real-time recognition test",
          created_at:
            new Date().toISOString(),
        },
      ];
      
      setVideos(data);

      setError("");
    } catch (error) {
      console.error(error);

      setError(
        "Failed to load videos.",
      );
    } finally {
      setLoading(false);
    }
  };

  /* ---------------------------------------------------------------------- */
  /*                               Effects                                  */
  /* ---------------------------------------------------------------------- */

  useEffect(() => {
    const loadVideos = async () => {
      await fetchVideos();
    };
  
    void loadVideos();
  }, []);

  /* ---------------------------------------------------------------------- */
  /*                              Statistics                                */
  /* ---------------------------------------------------------------------- */

  const totalVideos =
    videos.length;

  const latestTranscript =
    videos[0]?.transcript ??
    "No transcripts available";

  /* ---------------------------------------------------------------------- */
  /*                               Return                                   */
  /* ---------------------------------------------------------------------- */

  return {

    videos,

    loading,

    error,

    totalVideos,

    latestTranscript,

    refreshVideos:
      fetchVideos,
  };
}