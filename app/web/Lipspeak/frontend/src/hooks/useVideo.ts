/** useVideo.ts
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Video details hook.
 * @date: 11 June 2026
 * @returns: Video details state and actions.
 *
 */


"use client";


import {
  useEffect,
  useState,
} from "react";

import {
  videoService,
} from "@/services/api/video.service";

import type {
  Video,
} from "@/types/video.types";


export function useVideo(
  id: string,
) {
  const [
    video,
    setVideo,
  ] = useState<
    Video | null
  >(null);

  const [
    loading,
    setLoading,
  ] = useState(
    true,
  );

  useEffect(() => {
    const fetchVideo =
      async () => {
        try {
          const response =
            await videoService.getVideoById(
              id,
            );

          setVideo(
            response,
          );

        } catch (error) {
          console.error(
            error,
          );

        } finally {
          setLoading(
            false,
          );
        }
      };

    if (
      id
    ) {
      void fetchVideo();
    }
  }, [
    id,
  ]);

  return {
    video,
    loading,
  };
}