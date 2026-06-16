/** useRealtime.ts
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: State management hook for real-time speech recognition.
 * @date: 10 June 2026
 * @returns: Real-time recognition state and actions.
 *
 */


// Client Hook
"use client";


// Imports
import {
  useEffect,
  useRef,
  useState,
} from "react";

import { realtimeService }
  from "@/services/websocket/realtime.service";

import { useMediaRecorder }
  from "@/hooks/useMediaRecorder";


// Realtime Hook
export function useRealtime() {
  /* ---------------------------------------------------------------------- */
  /*                                Refs                                    */
  /* ---------------------------------------------------------------------- */

  const videoRef =
    useRef<HTMLVideoElement>(null);

  const streamRef =
    useRef<MediaStream | null>(
      null,
    );

  /* ---------------------------------------------------------------------- */
  /*                                Hooks                                   */
  /* ---------------------------------------------------------------------- */

  const {
    startRecording:
      startMediaRecording,

    stopRecording:
      stopMediaRecording,
  } = useMediaRecorder();

  /* ---------------------------------------------------------------------- */
  /*                                State                                   */
  /* ---------------------------------------------------------------------- */

  const [isRecording, setIsRecording] =
    useState(false);

  const [transcript, setTranscript] =
    useState("");

  const [
    connectionStatus,
    setConnectionStatus,
  ] = useState<
    | "disconnected"
    | "connecting"
    | "connected"
  >("disconnected");

  /* ---------------------------------------------------------------------- */
  /*                           Camera Setup                                 */
  /* ---------------------------------------------------------------------- */

  useEffect(() => {
    const initializeCamera =
      async () => {
        try {
          const stream =
            await navigator.mediaDevices.getUserMedia(
              {
                video: true,
                audio: false,
              },
            );

          streamRef.current =
            stream;

          if (
            videoRef.current
          ) {
            videoRef.current.srcObject =
              stream;
          }
        } catch (error) {
          console.error(
            "Failed to access camera:",
            error,
          );
        }
      };

    void initializeCamera();

    return () => {
      streamRef.current
        ?.getTracks()
        .forEach((track) =>
          track.stop(),
        );

      realtimeService.disconnect();
    };
  }, []);

  /* ---------------------------------------------------------------------- */
  /*                           Start Recording                              */
  /* ---------------------------------------------------------------------- */

  const startRecording =
    async () => {
      try {
        setConnectionStatus(
          "connecting",
        );

        realtimeService.connect({
          onOpen: () => {
            setConnectionStatus(
              "connected",
            );
          },

          onClose: () => {
            setConnectionStatus(
              "disconnected",
            );
          },

          onError: () => {
            setConnectionStatus(
              "disconnected",
            );
          },

          onTranscript: (newTranscript) => {
            // FIX: Just replace the state, do not append to it.
            setTranscript(newTranscript);
          }
        });

        if (
          streamRef.current
        ) {
          await startMediaRecording(
            streamRef.current,

            (chunk) => {
              realtimeService.send(
                chunk,
              );
            },
          );
        }

        setIsRecording(
          true,
        );
      } catch (error) {
        console.error(error);

        setConnectionStatus(
          "disconnected",
        );
      }
    };

  /* ---------------------------------------------------------------------- */
  /*                           Stop Recording                               */
  /* ---------------------------------------------------------------------- */
  
  const stopRecording = () => {
    stopMediaRecording();
  
    realtimeService.disconnect();
  
    setIsRecording(false);
  
    setConnectionStatus(
      "disconnected",
    );
  };

  /* ---------------------------------------------------------------------- */
  /*                                Return                                  */
  /* ---------------------------------------------------------------------- */

  return {
    videoRef,

    isRecording,

    transcript,

    connectionStatus,

    startRecording,

    stopRecording,
  };
}