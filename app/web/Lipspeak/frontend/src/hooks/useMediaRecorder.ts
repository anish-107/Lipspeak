// /** useMediaRecorder.ts
//  * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
//  * @description: MediaRecorder hook for capturing complete video windows.
//  * @date: 10 June 2026
//  * @returns: Media recording state and actions.
//  */

// "use client";

// import { useCallback, useRef, useState } from "react";

// export function useMediaRecorder() {
//   const mediaRecorderRef = useRef<MediaRecorder | null>(null);
//   const streamRef = useRef<MediaStream | null>(null);
//   const recordingLoopRef = useRef<NodeJS.Timeout | null>(null);

//   const [isRecording, setIsRecording] = useState(false);

//   const startRecording = useCallback(
//     async (stream: MediaStream, onChunk: (chunk: Blob) => void) => {
//       try {
//         streamRef.current = stream;
//         setIsRecording(true);

//         const recordWindow = () => {
//           if (!streamRef.current) return;

//           // const recorder = new MediaRecorder(streamRef.current, {
//           //   mimeType: "video/webm",
//           // });
//           // 
//           const recorder = new MediaRecorder(streamRef.current);
          
//           const chunks: Blob[] = [];

//           recorder.ondataavailable = (event) => {
//             if (event.data.size > 0) {
//               chunks.push(event.data);
//             }
//           };

//           recorder.onstop = () => {
//             if (chunks.length > 0) {
//               // Create a complete, playable video file with headers
//               const fullVideoBlob = new Blob(chunks, { type: recorder.mimeType });
//               onChunk(fullVideoBlob);
//             }

//             // Loop to start the next 5-second window if still active
//             if (streamRef.current) {
//               recordWindow();
//             }
//           };

//           mediaRecorderRef.current = recorder;
//           recorder.start();

//           // Stop and package the video every 5 seconds
//           recordingLoopRef.current = setTimeout(() => {
//             if (recorder.state === "recording") {
//               recorder.stop();
//             }
//           }, 5000);
//         };

//         // Kick off the first recording window
//         recordWindow();
//       } catch (error) {
//         console.error("Failed to start recording:", error);
//         setIsRecording(false);
//       }
//     },
//     []
//   );

//   const stopRecording = useCallback(() => {
//     // Clear the timeout loop
//     if (recordingLoopRef.current) {
//       clearTimeout(recordingLoopRef.current);
//     }
    
//     // Stop the stream and recorder
//     streamRef.current = null;
//     if (mediaRecorderRef.current?.state === "recording") {
//       mediaRecorderRef.current.stop();
//     }

//     setIsRecording(false);
//   }, []);

//   return {
//     isRecording,
//     startRecording,
//     stopRecording,
//     mediaRecorderRef,
//     streamRef,
//   };
// }
// 
// 


/** useMediaRecorder.ts
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: MediaRecorder hook for capturing continuous video chunks.
 * @date: 10 June 2026
 * @returns: Media recording state and actions.
 */

"use client";

import { useCallback, useRef, useState } from "react";

export function useMediaRecorder() {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [isRecording, setIsRecording] = useState(false);

  const startRecording = useCallback(
    async (stream: MediaStream, onChunk: (chunk: Blob) => void) => {
      try {
        streamRef.current = stream;
        setIsRecording(true);

        // 1. Initialize ONE recorder. Let the browser pick its optimal native format.
        const recorder = new MediaRecorder(stream);

        // 2. Listen for the native timeslice chunks
        recorder.ondataavailable = (event) => {
          if (event.data && event.data.size > 0) {
            console.log(`[MediaRecorder] Generated real chunk: ${event.data.size} bytes`);
            onChunk(event.data);
          } else {
            console.warn("[MediaRecorder] Ignored 0-byte ghost frame.");
          }
        };

        mediaRecorderRef.current = recorder;

        // 3. Start with a 5000ms timeslice.
        // This natively cuts a chunk every 5 seconds WITHOUT stopping the camera!
        recorder.start(5000);

      } catch (error) {
        console.error("Failed to start recording:", error);
        setIsRecording(false);
      }
    },
    []
  );

  const stopRecording = useCallback(() => {
    streamRef.current = null;
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  }, []);

  return {
    isRecording,
    startRecording,
    stopRecording,
    mediaRecorderRef,
    streamRef,
  };
}