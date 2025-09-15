import { io, Socket } from 'socket.io-client';
import { WebSocketMessage } from '../types';

class WebSocketService {
  private socket: Socket | null = null;
  private isConnected = false;
  private messageHandlers: Map<string, (data: any) => void> = new Map();

  connect(userId?: string): void {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';
    const clientId = `client-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    this.socket = io(wsUrl, {
      transports: ['websocket'],
      query: { client_id: clientId, user_id: userId },
    });

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.isConnected = true;
    });

    this.socket.on('disconnect', () => {
      console.log('WebSocket disconnected');
      this.isConnected = false;
    });

    this.socket.on('message', (message: WebSocketMessage) => {
      this.handleMessage(message);
    });

    this.socket.on('solution_complete', (data) => {
      this.handleMessage({ type: 'solution_complete', data, timestamp: Date.now() });
    });

    this.socket.on('feedback_processed', (data) => {
      this.handleMessage({ type: 'feedback_processed', data, timestamp: Date.now() });
    });

    this.socket.on('error', (error) => {
      console.error('WebSocket error:', error);
    });
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.isConnected = false;
    }
  }

  sendMessage(type: string, data?: any): void {
    if (this.socket && this.isConnected) {
      this.socket.emit('message', { type, data, timestamp: Date.now() });
    }
  }

  onMessage(type: string, handler: (data: any) => void): void {
    this.messageHandlers.set(type, handler);
  }

  offMessage(type: string): void {
    this.messageHandlers.delete(type);
  }

  private handleMessage(message: WebSocketMessage): void {
    const handler = this.messageHandlers.get(message.type);
    if (handler) {
      handler(message.data);
    }
  }

  isSocketConnected(): boolean {
    return this.isConnected;
  }
}

export const websocketService = new WebSocketService();
