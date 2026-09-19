import { ImageUploadResponse, SatQueryResult, TaskType, TraceStep } from './api';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  images?: ImageUploadResponse[];
  result?: SatQueryResult;
  reportId?: string;
  isStreaming?: boolean;
  streamingSteps?: TraceStep[];
  error?: string;
}

export interface AnalysisSession {
  id: string;
  title: string;
  taskType?: TaskType;
  createdAt: string;
  messages: ChatMessage[];
  images: ImageUploadResponse[];
  lastResult?: SatQueryResult;
  lastReportId?: string;
}
