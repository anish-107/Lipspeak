/** realtime.service.ts
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: WebSocket service for real-time speech recognition.
 * @date: 10 June 2026
 * @returns: Realtime websocket operations.
 *
 */

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

export class RealtimeService {
  private socket:
    | WebSocket
    | null = null;

  private reconnectAttempts = 0;

  private readonly maxReconnects =
    5;

  private manuallyDisconnected =
    false;

  connect(
    options?: RealtimeServiceOptions,
  ) {
    this.manuallyDisconnected =
      false;

    if (
      this.socket &&
      this.socket.readyState ===
        WebSocket.OPEN
    ) {
      console.log(
        "[WS] Already connected",
      );
      return;
    }

    const websocketUrl =
      process.env
        .NEXT_PUBLIC_WS_URL ??
      "ws://localhost:8000/ws/realtime";

    console.log(
      "[WS] Connecting:",
      websocketUrl,
    );

    this.socket =
      new WebSocket(
        websocketUrl,
      );

    this.socket.onopen = () => {
      console.log(
        "[WS] Connected",
      );

      this.reconnectAttempts = 0;

      options?.onOpen?.();
    };

    this.socket.onmessage = (
      event: MessageEvent,
    ) => {
      try {
        console.log(
          "[WS] RAW MESSAGE:",
          event.data,
        );

        const data =
          JSON.parse(
            event.data,
          );

        console.log(
          "[WS] PARSED MESSAGE:",
          data,
        );

        if (
          data.transcript
        ) {
          console.log(
            "[WS] TRANSCRIPT RECEIVED:",
            data.transcript,
          );

          options?.onTranscript?.(
            data.transcript,
          );
        }
      } catch (error) {
        console.error(
          "[WS] Failed to parse message:",
          error,
        );
      }
    };

    this.socket.onerror = (
      event,
    ) => {
      console.error(
        "[WS] ERROR:",
        event,
      );

      options?.onError?.(
        event,
      );
    };

    this.socket.onclose = (
      event,
    ) => {
      console.warn(
        "[WS] CLOSED:",
        {
          code: event.code,
          reason:
            event.reason,
          wasClean:
            event.wasClean,
        },
      );

      options?.onClose?.();

      if (
        !this.manuallyDisconnected
      ) {
        console.log(
          "[WS] Attempting reconnect...",
        );

        this.tryReconnect(
          options,
        );
      }
    };
  }

  private tryReconnect(
    options?: RealtimeServiceOptions,
  ) {
    if (
      this.reconnectAttempts >=
      this.maxReconnects
    ) {
      console.error(
        "[WS] Max reconnect attempts reached",
      );
      return;
    }

    this.reconnectAttempts++;

    console.log(
      `[WS] Reconnect attempt ${this.reconnectAttempts}/${this.maxReconnects}`,
    );

    setTimeout(() => {
      this.connect(
        options,
      );
    }, 2000);
  }

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
      console.warn(
        "[WS] Tried to send while disconnected",
      );
      return;
    }

    console.log(
      "[WS] Sending chunk",
    );

    this.socket.send(
      data,
    );
  }

  disconnect() {
    console.log(
      "[WS] Manual disconnect",
    );

    this.manuallyDisconnected =
      true;

    this.socket?.close();

    this.socket = null;
  }

  isConnected() {
    return (
      this.socket
        ?.readyState ===
      WebSocket.OPEN
    );
  }
}

export const realtimeService =
  new RealtimeService();