/** realtime.service.ts
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: WebSocket service for real-time speech recognition.
 * @date: 10 June 2026
 * @returns: Realtime websocket operations.
 *
 */


// Types
interface RealtimeServiceOptions {
  onOpen?: () => void;

  onClose?: () => void;

  onError?: (
    event: Event,
  ) => void;

  onTranscript?: (
    transcript: string,
  ) => void;
}


// Realtime Service
export class RealtimeService {
  /* ---------------------------------------------------------------------- */
  /*                                Fields                                  */
  /* ---------------------------------------------------------------------- */

  private socket:
    | WebSocket
    | null = null;

  private reconnectAttempts = 0;

  private readonly maxReconnects =
    5;

  /* ---------------------------------------------------------------------- */
  /*                               Connect                                  */
  /* ---------------------------------------------------------------------- */

  connect(
    options?: RealtimeServiceOptions,
  ) {
    if (
      this.socket &&
      this.socket.readyState ===
        WebSocket.OPEN
    ) {
      return;
    }

    const websocketUrl =
      process.env
        .NEXT_PUBLIC_WS_URL ??
      "ws://localhost:8000/ws/realtime";

    this.socket =
      new WebSocket(
        websocketUrl,
      );

    this.socket.onopen = () => {
      this.reconnectAttempts = 0;

      options?.onOpen?.();
    };

    this.socket.onmessage = (
      event: MessageEvent,
    ) => {
      try {
        const data =
          JSON.parse(
            event.data,
          );

        if (
          data.transcript
        ) {
          options?.onTranscript?.(
            data.transcript,
          );
        }
      } catch (error) {
        console.error(
          "Failed to parse websocket message:",
          error,
        );
      }
    };

    this.socket.onerror = (
      event,
    ) => {
      options?.onError?.(
        event,
      );
    };

    this.socket.onclose = () => {
      options?.onClose?.();

      this.tryReconnect(
        options,
      );
    };
  }

  /* ---------------------------------------------------------------------- */
  /*                             Reconnect                                  */
  /* ---------------------------------------------------------------------- */

  private tryReconnect(
    options?: RealtimeServiceOptions,
  ) {
    if (
      this.reconnectAttempts >=
      this.maxReconnects
    ) {
      return;
    }

    this.reconnectAttempts++;

    setTimeout(() => {
      this.connect(
        options,
      );
    }, 2000);
  }

  /* ---------------------------------------------------------------------- */
  /*                             Send Data                                  */
  /* ---------------------------------------------------------------------- */

  send(
    data:
      | Blob
      | ArrayBuffer
      | string,
  ) {
    if (
      !this.socket ||
      this.socket.readyState !==
        WebSocket.OPEN
    ) {
      return;
    }

    this.socket.send(data);
  }

  /* ---------------------------------------------------------------------- */
  /*                            Disconnect                                  */
  /* ---------------------------------------------------------------------- */

  disconnect() {
    this.socket?.close();

    this.socket = null;
  }

  /* ---------------------------------------------------------------------- */
  /*                            Connection                                  */
  /* ---------------------------------------------------------------------- */

  isConnected() {
    return (
      this.socket
        ?.readyState ===
      WebSocket.OPEN
    );
  }
}


// Singleton
export const realtimeService =
  new RealtimeService();