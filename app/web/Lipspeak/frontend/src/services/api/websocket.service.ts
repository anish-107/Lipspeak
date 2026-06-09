/** websocket.service.ts
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: WebSocket service for real-time lip-reading communication.
 * @date: 09 June 2026
 * @returns: WebSocket communication methods.
 *
 */


// Imports
import { env } from "@/config/env";
import type {
  TranscriptResponse,
} from "@/types/websocket.types";


// WebSocket Service
class WebSocketService {
  private socket: WebSocket | null = null;


  /* ------------------------------------------------------------------------ */
  /*                                 Connect                                  */
  /* ------------------------------------------------------------------------ */

  connect(
    onMessage: (data: TranscriptResponse) => void,
    onOpen?: () => void,
    onClose?: () => void,
  ) {
    if (this.socket) {
      this.disconnect();
    }

    this.socket = new WebSocket(
      `${env.WS_URL}/transcribe`,
    );

    this.socket.onopen = () => {
      console.log("WebSocket Connected");

      onOpen?.();
    };

    this.socket.onmessage = (event) => {
      try {
        const data: TranscriptResponse = JSON.parse(
          event.data,
        );

        onMessage(data);
      } catch (error) {
        console.error(
          "Failed to parse WebSocket message:",
          error,
        );
      }
    };

    this.socket.onclose = () => {
      console.log("WebSocket Disconnected");

      onClose?.();
    };

    this.socket.onerror = (error) => {
      console.error(
        "WebSocket Error:",
        error,
      );
    };
  }


  /* ------------------------------------------------------------------------ */
  /*                             Send Start Event                             */
  /* ------------------------------------------------------------------------ */

  startRecording() {
    this.send({
      type: "start",
    });
  }


  /* ------------------------------------------------------------------------ */
  /*                             Send Stop Event                              */
  /* ------------------------------------------------------------------------ */

  stopRecording() {
    this.send({
      type: "stop",
    });
  }


  /* ------------------------------------------------------------------------ */
  /*                              Send Chunk                                  */
  /* ------------------------------------------------------------------------ */

  sendChunk(blob: Blob) {
    if (
      !this.socket ||
      this.socket.readyState !== WebSocket.OPEN
    ) {
      return;
    }

    this.socket.send(blob);
  }


  /* ------------------------------------------------------------------------ */
  /*                               Send JSON                                  */
  /* ------------------------------------------------------------------------ */

  send(data: unknown) {
    if (
      !this.socket ||
      this.socket.readyState !== WebSocket.OPEN
    ) {
      return;
    }

    this.socket.send(
      JSON.stringify(data),
    );
  }


  /* ------------------------------------------------------------------------ */
  /*                               Disconnect                                 */
  /* ------------------------------------------------------------------------ */

  disconnect() {
    if (this.socket) {
      this.socket.close();

      this.socket = null;
    }
  }


  /* ------------------------------------------------------------------------ */
  /*                              Connection State                            */
  /* ------------------------------------------------------------------------ */

  isConnected() {
    return (
      this.socket?.readyState ===
      WebSocket.OPEN
    );
  }
}


// Export Singleton
export const websocketService =
  new WebSocketService();