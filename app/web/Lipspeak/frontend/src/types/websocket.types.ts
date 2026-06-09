/** websocket.types.ts
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: WebSocket related TypeScript types and interfaces.
 * @date: 09 June 2026
 * @returns: WebSocket type definitions.
 *
 */


/* -------------------------------------------------------------------------- */
/*                             Client Messages                                */
/* -------------------------------------------------------------------------- */

export interface TranscriptChunkMessage {
  type: "chunk";

  data: Blob;
}

export interface StartRecordingMessage {
  type: "start";
}

export interface StopRecordingMessage {
  type: "stop";
}


/* -------------------------------------------------------------------------- */
/*                             Server Messages                                */
/* -------------------------------------------------------------------------- */

export interface TranscriptResponse {
  transcript: string;

  confidence?: number;

  isFinal?: boolean;
}