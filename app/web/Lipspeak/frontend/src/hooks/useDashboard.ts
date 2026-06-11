/** useDashboard.ts
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Dashboard data management hook.
 * @date: 11 June 2026
 * @returns: Dashboard state and actions.
 *
 */


// Client Hook
"use client";


// Imports
import {
  useEffect,
  useState,
} from "react";

import {
  dashboardService,
} from "@/services/api/dashboard.service";

import type {
  Video,
} from "@/types/video.types";


// Dashboard Hook
export function useDashboard() {
  /* ---------------------------------------------------------------------- */
  /*                                State                                   */
  /* ---------------------------------------------------------------------- */

  const [
    videos,
    setVideos,
  ] = useState<
    Video[]
  >([]);

  const [
    totalVideos,
    setTotalVideos,
  ] = useState(
    0,
  );

  const [
    latestTranscript,
    setLatestTranscript,
  ] = useState(
    "No transcripts available",
  );

  const [
    loading,
    setLoading,
  ] = useState(
    true,
  );

  const [
    error,
    setError,
  ] = useState(
    "",
  );

  /* ---------------------------------------------------------------------- */
  /*                             Load Dashboard                             */
  /* ---------------------------------------------------------------------- */

  const fetchDashboard =
    async () => {
      try {
        setLoading(
          true,
        );

        const data =
          await dashboardService.getOverview();

        setVideos(
          data.recent_videos,
        );

        setTotalVideos(
          data.total_videos,
        );

        setLatestTranscript(
          data.latest_transcript,
        );

        setError(
          "",
        );

      } catch (error) {
        console.error(
          error,
        );

        setError(
          "Failed to load dashboard.",
        );

      } finally {
        setLoading(
          false,
        );
      }
    };

  /* ---------------------------------------------------------------------- */
  /*                               Effects                                  */
  /* ---------------------------------------------------------------------- */

  useEffect(() => {
    dashboardService
      .getOverview()
      .then((data) => {
        setVideos(
          data.recent_videos,
        );
  
        setTotalVideos(
          data.total_videos,
        );
  
        setLatestTranscript(
          data.latest_transcript,
        );
      })
      .catch(() => {
        setError(
          "Failed to load dashboard.",
        );
      })
      .finally(() => {
        setLoading(
          false,
        );
      });
  }, []);

  /* ---------------------------------------------------------------------- */
  /*                               Return                                   */
  /* ---------------------------------------------------------------------- */

  return {
    videos,

    totalVideos,

    latestTranscript,

    loading,

    error,

    refreshVideos:
      fetchDashboard,
  };
}