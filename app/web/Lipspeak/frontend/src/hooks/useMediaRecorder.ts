/** useMediaRecorder.ts
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: MediaRecorder hook for capturing video chunks from webcam.
 * @date: 10 June 2026
 * @returns: Media recording state and actions.
 *
 */


// Client Hook
"use client";


// Imports
import {
  useCallback,
  useRef,
  useState,
} from "react";


// Hook
export function useMediaRecorder() {
  /* ---------------------------------------------------------------------- */
  /*                                Refs                                    */
  /* ---------------------------------------------------------------------- */

  const mediaRecorderRef =
    useRef<MediaRecorder | null>(
      null,
    );

  const streamRef =
    useRef<MediaStream | null>(
      null,
    );

  /* ---------------------------------------------------------------------- */
  /*                                State                                   */
  /* ---------------------------------------------------------------------- */

  const [isRecording, setIsRecording] =
    useState(false);

  /* ---------------------------------------------------------------------- */
  /*                           Start Recording                              */
  /* ---------------------------------------------------------------------- */

  const startRecording =
    useCallback(
      async (
        stream: MediaStream,

        onChunk: (
          chunk: Blob,
        ) => void,
      ) => {
        try {
          streamRef.current =
            stream;

          const recorder =
            new MediaRecorder(
              stream,
              {
                mimeType:
                  "video/webm",
              },
            );

          mediaRecorderRef.current =
            recorder;

          recorder.ondataavailable =
            (event) => {
              if (
                event.data.size > 0
              ) {
                onChunk(
                  event.data,
                );
              }
            };

          recorder.start(1000);

          setIsRecording(
            true,
          );
        } catch (error) {
          console.error(
            "Failed to start recording:",
            error,
          );
        }
      },
      [],
    );

  /* ---------------------------------------------------------------------- */
  /*                           Stop Recording                               */
  /* ---------------------------------------------------------------------- */

  const stopRecording =
    useCallback(() => {
      mediaRecorderRef.current?.stop();

      setIsRecording(false);
    }, []);

  /* ---------------------------------------------------------------------- */
  /*                                Return                                  */
  /* ---------------------------------------------------------------------- */

  return {
    isRecording,

    startRecording,

    stopRecording,

    mediaRecorderRef,

    streamRef,
  };
}