import { SatQueryResult, TraceStep } from '../types/api';
import { SatQueryAPI } from './api';

export interface StreamQueryCallbacks {
  onAck?: (data: { query: string; image_count: number; status: string }) => void;
  onStep?: (step: TraceStep) => void;
  onResult: (result: SatQueryResult) => void;
  onError: (error: string) => void;
  onClose?: () => void;
}

export class SatQueryWebSocket {
  private ws: WebSocket | null = null;
  private isClosedManually = false;

  /**
   * Execute query with streaming execution trace via WebSocket.
   * Automatically falls back to standard REST POST if WebSocket is unavailable.
   */
  public execute(
    query: string,
    imageIds: string[],
    callbacks: StreamQueryCallbacks,
    history?: { role: string; content: string }[]
  ): () => void {
    if (this.ws) {
      try {
        this.ws.close();
      } catch {
        // ignore
      }
      this.ws = null;
    }
    this.isClosedManually = false;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/api/v1/ws/query`;

    let stepIndexCounter = 0;
    let isFallingBack = false;
    let hasReceivedResult = false;
    let hasReceivedError = false;
    let connectTimeout: any = null;

    try {
      this.ws = new WebSocket(wsUrl);

      connectTimeout = setTimeout(() => {
        if (this.ws && this.ws.readyState === WebSocket.CONNECTING && !this.isClosedManually) {
          console.warn('WebSocket connection timed out, falling back to REST');
          isFallingBack = true;
          try {
            this.ws.close();
          } catch {
            // ignore
          }
          this.ws = null;
          this.fallbackToRest(query, imageIds, callbacks, history);
        }
      }, 6000);

      this.ws.onopen = () => {
        if (connectTimeout) clearTimeout(connectTimeout);
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(
            JSON.stringify({
              query,
              image_ids: imageIds,
              history: history || [],
            })
          );
        }
      };

      this.ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          const type = payload.type;
          const data = payload.data;

          if (this.isClosedManually) return;

          if (type === 'ack') {
            callbacks.onAck?.(data);
          } else if (type === 'step') {
            stepIndexCounter++;
            const step: TraceStep = {
              step_index: data.step_index ?? stepIndexCounter,
              step_name: data.step_name || 'PROCESSING',
              status: data.status || 'SUCCESS',
              duration_ms: data.duration_ms ?? 0,
              details: data.details || {},
              message: data.message,
            };
            callbacks.onStep?.(step);
          } else if (type === 'result') {
            hasReceivedResult = true;
            callbacks.onResult(data as SatQueryResult);
          } else if (type === 'error') {
            hasReceivedError = true;
            callbacks.onError(data.message || 'An error occurred during analysis');
          }
        } catch (parseErr) {
          console.error('Error parsing WebSocket message:', parseErr);
        }
      };

      this.ws.onerror = () => {
        if (connectTimeout) clearTimeout(connectTimeout);
        // Fallback to REST API if WebSocket fails.
        // Set flag BEFORE closing so onclose doesn't fire onClose() prematurely.
        if (!this.isClosedManually && !isFallingBack) {
          isFallingBack = true;
          try {
            this.ws?.close();
          } catch {
            // ignore
          }
          this.ws = null;
          this.fallbackToRest(query, imageIds, callbacks, history);
        }
      };

      this.ws.onclose = (event) => {
        if (connectTimeout) clearTimeout(connectTimeout);
        if (isFallingBack || this.isClosedManually) return;

        // If closed unexpectedly without result or error, fallback to REST
        if (!hasReceivedResult && !hasReceivedError) {
          console.warn(`WebSocket closed unexpectedly (code ${event.code}). Falling back to REST.`);
          isFallingBack = true;
          this.fallbackToRest(query, imageIds, callbacks, history);
          return;
        }

        callbacks.onClose?.();
      };
    } catch {
      if (connectTimeout) clearTimeout(connectTimeout);
      // Fallback to REST if opening failed
      this.fallbackToRest(query, imageIds, callbacks, history);
    }

    // Return cleanup/cancel function
    return () => {
      this.isClosedManually = true;
      if (connectTimeout) clearTimeout(connectTimeout);
      if (this.ws) {
        this.ws.close();
        this.ws = null;
      }
    };
  }

  private async fallbackToRest(
    query: string,
    imageIds: string[],
    callbacks: StreamQueryCallbacks,
    history?: { role: string; content: string }[]
  ) {
    if (this.ws) {
      try {
        this.ws.close();
      } catch {
        // ignore
      }
      this.ws = null;
    }
    if (this.isClosedManually) return;

    try {
      callbacks.onStep?.({
        step_index: 1,
        step_name: 'REST_DISPATCH',
        status: 'IN_PROGRESS',
        duration_ms: 0,
        details: { mode: 'fallback_http' },
        message: 'Dispatching query via HTTP REST gateway...',
      });

      const result = await SatQueryAPI.executeQuery(query, imageIds, history);
      if (this.isClosedManually) return;

      // Emit simulated trace steps if backend returned trace
      if (result.audit_trace?.execution_steps) {
        for (const step of result.audit_trace.execution_steps) {
          if (this.isClosedManually) return;
          callbacks.onStep?.(step);
        }
      }

      callbacks.onResult(result);
    } catch (err: any) {
      if (!this.isClosedManually) {
        callbacks.onError(err.message || 'HTTP Query execution failed');
      }
    } finally {
      if (!this.isClosedManually) {
        callbacks.onClose?.();
      }
    }
  }
}
