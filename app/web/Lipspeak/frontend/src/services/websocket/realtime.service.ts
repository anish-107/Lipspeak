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

      if (
        !this.manuallyDisconnected
      ) {
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
      return;
    }

    this.reconnectAttempts++;

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
      return;
    }

    this.socket.send(data);
  }

  disconnect() {
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