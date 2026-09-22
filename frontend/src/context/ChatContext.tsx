import React, { createContext, useContext, useEffect, useState, useRef } from 'react';
import { ImageUploadResponse, SatQueryResult, SystemHealth, TraceStep } from '../types/api';
import { AnalysisSession, ChatMessage } from '../types/chat';
import { SatQueryAPI } from '../services/api';
import { SatQueryWebSocket } from '../services/websocket';

interface ChatContextType {
  sessions: AnalysisSession[];
  activeSessionId: string;
  currentSession: AnalysisSession;
  activeImages: ImageUploadResponse[];
  isProcessing: boolean;
  systemHealth: SystemHealth | null;
  activeAnalysisResult: SatQueryResult | null;
  isSidebarOpen: boolean;
  isAnalysisPanelOpen: boolean;
  activeImageForAnalysis: ImageUploadResponse | null;
  /** Pre-filled query text to populate the chat input (without auto-submitting) */
  pendingQueryText: string;
  setPendingQueryText: (text: string) => void;
  setSidebarOpen: (open: boolean) => void;
  setAnalysisPanelOpen: (open: boolean) => void;
  createNewSession: () => void;
  selectSession: (id: string) => void;
  deleteSession: (id: string) => void;
  renameSession: (id: string, newTitle: string) => void;
  selectMessageResult: (message: ChatMessage) => void;
  attachImage: (img: ImageUploadResponse) => void;
  removeImage: (fileId: string) => void;
  clearAttachedImages: () => void;
  submitQuery: (queryText: string, overrideImages?: ImageUploadResponse[], targetSessionIdParam?: string) => Promise<void>;
  cancelProcessing: () => void;
  setActiveAnalysisResult: (res: SatQueryResult | null) => void;
  setActiveImageForAnalysis: (img: ImageUploadResponse | null) => void;
  loadPresetAnalysis: (presetType: string) => Promise<void>;
  activeReportId: string | null;
  setActiveReportId: (id: string | null) => void;
  isGeneratingReport: boolean;
  generateReportForCurrentResult: (result?: SatQueryResult | null, images?: ImageUploadResponse[]) => Promise<string | null>;
  isPdfModalOpen: boolean;
  pdfModalResult: SatQueryResult | null;
  pdfModalImage: ImageUploadResponse | null;
  openPdfModal: (result: SatQueryResult, image?: ImageUploadResponse | null) => void;
  closePdfModal: () => void;
}

const STORAGE_KEY = 'satquery_sessions_v1';

const createDefaultSession = (): AnalysisSession => ({
  id: `session_${Date.now()}`,
  title: 'New Analysis',
  createdAt: new Date().toISOString(),
  messages: [],
  images: [],
});

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export const ChatProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [sessions, setSessions] = useState<AnalysisSession[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
      }
    } catch {
      // ignore
    }
    return [createDefaultSession()];
  });

  const [activeSessionId, setActiveSessionId] = useState<string>(() => sessions[0]?.id || `session_${Date.now()}`);
  const [activeImages, setActiveImages] = useState<ImageUploadResponse[]>(() => sessions[0]?.images || []);
  const [isProcessing, setIsProcessing] = useState(false);
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [activeAnalysisResult, setActiveAnalysisResult] = useState<SatQueryResult | null>(null);
  const [activeImageForAnalysis, setActiveImageForAnalysis] = useState<ImageUploadResponse | null>(null);
  const [activeReportId, setActiveReportId] = useState<string | null>(null);
  const [isGeneratingReport, setIsGeneratingReport] = useState<boolean>(false);
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  const [isAnalysisPanelOpen, setAnalysisPanelOpen] = useState(false);
  const [isPdfModalOpen, setPdfModalOpen] = useState(false);
  const [pdfModalResult, setPdfModalResult] = useState<SatQueryResult | null>(null);
  const [pdfModalImage, setPdfModalImage] = useState<ImageUploadResponse | null>(null);
  const [pendingQueryText, setPendingQueryText] = useState('');

  const sessionsRef = useRef(sessions);
  useEffect(() => {
    sessionsRef.current = sessions;
  }, [sessions]);

  const activeSessionIdRef = useRef(activeSessionId);
  useEffect(() => {
    activeSessionIdRef.current = activeSessionId;
  }, [activeSessionId]);

  const generateReportForCurrentResult = async (
    res?: SatQueryResult | null,
    imgs?: ImageUploadResponse[]
  ): Promise<string | null> => {
    const targetRes = res || activeAnalysisResult;
    if (!targetRes) return null;
    const targetImgs = imgs || activeImages;
    setIsGeneratingReport(true);
    try {
      const thumbPaths: string[] = [];
      if (targetRes.thumbnail_urls) {
        targetRes.thumbnail_urls.forEach((u) => {
          if (u && !thumbPaths.includes(u)) thumbPaths.push(u);
        });
      }
      targetImgs.forEach((img) => {
        const u = img.thumbnail_url || img.file_id;
        if (u && !thumbPaths.includes(u)) thumbPaths.push(u);
      });

      const resp = await SatQueryAPI.generateReport({
        query: targetRes.query,
        text_response: targetRes.text_response,
        audit_trace: targetRes.audit_trace as any,
        spatial_evidence: targetRes.spatial_evidence as any,
        mask_image_path: targetRes.spatial_evidence?.mask_url || undefined,
        thumbnail_paths: thumbPaths.length > 0 ? thumbPaths : undefined,
        image_metadata: targetImgs.map((img) => ({
          file_id: img.file_id,
          filename: img.filename,
          modality: img.modality,
          width: img.width,
          height: img.height,
          band_count: img.band_count,
          crs: img.crs,
          thumbnail_url: img.thumbnail_url,
        })),
        classification: 'RESTRICTED',
        layout_mode: 'rapid_assessment',
      });
      setActiveReportId(resp.report_id);
      return resp.report_id;
    } catch (err) {
      console.warn('Failed to compile SAC-ISRO report:', err);
      return null;
    } finally {
      setIsGeneratingReport(false);
    }
  };

  const openPdfModal = (result: SatQueryResult, image?: ImageUploadResponse | null) => {
    setPdfModalResult(result);
    setPdfModalImage(image || activeImageForAnalysis || activeImages[0] || null);
    setPdfModalOpen(true);
  };

  const closePdfModal = () => {
    setPdfModalOpen(false);
  };

  const activeCancelRef = useRef<(() => void) | null>(null);

  // Sync sessions to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    } catch (e) {
      console.error('Failed to save sessions to localStorage:', e);
    }
  }, [sessions]);

  // Initial health check
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await SatQueryAPI.getHealth();
        setSystemHealth(health);
      } catch {
        setSystemHealth(null);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const currentSession = sessions.find((s) => s.id === activeSessionId) || sessions[0] || createDefaultSession();

  const createNewSession = () => {
    const newSession = createDefaultSession();
    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newSession.id);
    setActiveImages([]);
    setActiveAnalysisResult(null);
    setActiveImageForAnalysis(null);
    setActiveReportId(null);
    // Always close the right panel for a fresh session
    setAnalysisPanelOpen(false);
  };

  const selectSession = (id: string) => {
    setActiveSessionId(id);
    const s = sessionsRef.current.find((item) => item.id === id);
    if (s) {
      if (!isProcessing) {
        setActiveImages(s.images || []);
      }
      const lastRes = s.lastResult || [...(s.messages || [])].reverse().find((m) => m.result)?.result || null;
      setActiveAnalysisResult(lastRes);
      if (s.images && s.images.length > 0) {
        setActiveImageForAnalysis(s.images[0]);
      } else {
        setActiveImageForAnalysis(null);
      }
      setActiveReportId(s.lastReportId || null);
      // Only keep panel open if the session already has a result to display
      if (!lastRes) {
        setAnalysisPanelOpen(false);
      }
    }
  };

  const renameSession = (id: string, newTitle: string) => {
    if (!newTitle.trim()) return;
    setSessions((prev) =>
      prev.map((s) => (s.id === id ? { ...s, title: newTitle.trim() } : s))
    );
  };

  const selectMessageResult = (message: ChatMessage) => {
    if (message.result) {
      setActiveAnalysisResult(message.result);
      if (message.images && message.images.length > 0) {
        setActiveImageForAnalysis(message.images[0]);
      }
      setActiveReportId(message.reportId || null);
      setAnalysisPanelOpen(true);
    }
  };

  const deleteSession = (id: string) => {
    const filtered = sessions.filter((s) => s.id !== id);
    let nextSessions: AnalysisSession[];
    let nextActiveId: string | null = null;
    let nextImages: ImageUploadResponse[] = [];
    let nextResult: SatQueryResult | null = null;
    let nextImageForAnalysis: ImageUploadResponse | null = null;

    if (filtered.length === 0) {
      const fresh = createDefaultSession();
      nextSessions = [fresh];
      nextActiveId = fresh.id;
    } else {
      nextSessions = filtered;
      if (activeSessionIdRef.current === id) {
        nextActiveId = filtered[0].id;
        nextImages = filtered[0].images || [];
        nextResult = filtered[0].lastResult || null;
        nextImageForAnalysis = filtered[0].images?.[0] || null;
      }
    }

    setSessions(nextSessions);

    if (nextActiveId) {
      setActiveSessionId(nextActiveId);
      setActiveImages(nextImages);
      setActiveAnalysisResult(nextResult);
      setActiveImageForAnalysis(nextImageForAnalysis);
      setActiveReportId(null);
    }
  };

  const attachImage = (img: ImageUploadResponse) => {
    const currentActiveId = activeSessionIdRef.current;
    setActiveImages((prev) => {
      // Limit to 2 images max as enforced by backend.
      // Keep the first image (slot 0) and replace slot 1 with the new one.
      const existing = prev.filter((item) => item.file_id !== img.file_id);
      return existing.length >= 2 ? [existing[0], img] : [...existing, img];
    });

    setSessions((sPrev) =>
      sPrev.map((s) => {
        if (s.id === currentActiveId) {
          const existing = (s.images || []).filter((item) => item.file_id !== img.file_id);
          const updated = existing.length >= 2 ? [existing[0], img] : [...existing, img];
          return { ...s, images: updated };
        }
        return s;
      })
    );

    setActiveImageForAnalysis(img);
  };

  const removeImage = (fileId: string) => {
    const currentActiveId = activeSessionIdRef.current;
    setActiveImages((prev) => prev.filter((img) => img.file_id !== fileId));
    setSessions((sPrev) =>
      sPrev.map((s) =>
        s.id === currentActiveId
          ? { ...s, images: (s.images || []).filter((img) => img.file_id !== fileId) }
          : s
      )
    );
    if (activeImageForAnalysis?.file_id === fileId) {
      setActiveImageForAnalysis(null);
    }
  };

  const clearAttachedImages = () => {
    const currentActiveId = activeSessionIdRef.current;
    setActiveImages([]);
    setActiveImageForAnalysis(null);
    setSessions((prev) =>
      prev.map((s) => (s.id === currentActiveId ? { ...s, images: [] } : s))
    );
  };

  const cancelProcessing = () => {
    if (activeCancelRef.current) {
      activeCancelRef.current();
      activeCancelRef.current = null;
    }
    setIsProcessing(false);
  };

  const submitQuery = async (
    queryText: string,
    overrideImages?: ImageUploadResponse[],
    targetSessionIdParam?: string
  ) => {
    if (!queryText.trim() || isProcessing) return;

    // Lock to target session at dispatch time to avoid stale closure if user navigates
    const targetSessionId = targetSessionIdParam || activeSessionIdRef.current;
    setActiveReportId(null);

    const ts = Date.now();
    const uniqueSuffix = Math.random().toString(36).slice(2, 9);
    const userMessageId = `msg_${ts}_${uniqueSuffix}_user`;
    const assistantMessageId = `msg_${ts}_${uniqueSuffix}_assistant`;

    const targetSessionBefore = sessionsRef.current.find((s) => s.id === targetSessionId) || currentSession;

    const effectiveImages =
      overrideImages && overrideImages.length > 0
        ? [...overrideImages]
        : activeImages.length > 0
        ? [...activeImages]
        : targetSessionBefore.images && targetSessionBefore.images.length > 0
        ? [...targetSessionBefore.images]
        : [];

    const userMessage: ChatMessage = {
      id: userMessageId,
      role: 'user',
      content: queryText.trim(),
      timestamp: new Date().toISOString(),
      images: [...effectiveImages],
    };

    const initialAssistantMessage: ChatMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
      images: [...effectiveImages],
      isStreaming: true,
      streamingSteps:
        effectiveImages.length > 0
          ? [
              {
                step_index: 0,
                step_name: 'DISPATCHING',
                status: 'IN_PROGRESS',
                duration_ms: 0,
                details: {},
                message: 'Connecting to SatQuery Agentic Orchestrator...',
              },
            ]
          : [],
    };

    // Update session title if first message
    const currentMsgs = targetSessionBefore.messages || [];
    const updatedTitle =
      currentMsgs.length === 0
        ? queryText.slice(0, 32) + (queryText.length > 32 ? '...' : '')
        : (targetSessionBefore.title || 'Analysis');

    setSessions((prev) =>
      prev.map((s) => {
        if (s.id === targetSessionId) {
          return {
            ...s,
            title: updatedTitle,
            images: effectiveImages,
            messages: [...s.messages, userMessage, initialAssistantMessage],
          };
        }
        return s;
      })
    );

    setIsProcessing(true);

    const imageIds = effectiveImages.map((img) => img.file_id);
    const wsClient = new SatQueryWebSocket();

    // Extract prior multi-turn context for agent memory
    const conversationHistory = currentMsgs
      .filter((m) => m.content && !m.isStreaming && !m.error)
      .slice(-6)
      .map((m) => ({
        role: m.role,
        content: m.content,
      }));

    let isCancelled = false;

    const cleanup = wsClient.execute(
      queryText.trim(),
      imageIds,
      {
        onStep: (step: TraceStep) => {
          if (isCancelled) return;
          setSessions((prev) =>
            prev.map((s) => {
              if (s.id === targetSessionId) {
                return {
                  ...s,
                  messages: s.messages.map((m) => {
                    if (m.id === assistantMessageId) {
                      const existingSteps = m.streamingSteps || [];
                      return {
                        ...m,
                        streamingSteps: [...existingSteps, step],
                      };
                    }
                    return m;
                  }),
                };
              }
              return s;
            })
          );
        },
        onResult: (result: SatQueryResult) => {
          if (isCancelled) return;
          setIsProcessing(false);
          setActiveAnalysisResult(result);

          setSessions((prev) =>
            prev.map((s) => {
              if (s.id === targetSessionId) {
                return {
                  ...s,
                  lastResult: result,
                  taskType: result.audit_trace?.task_identified,
                  messages: s.messages.map((m) => {
                    if (m.id === assistantMessageId) {
                      return {
                        ...m,
                        content: result.text_response,
                        result,
                        images: [...effectiveImages],
                        isStreaming: false,
                      };
                    }
                    return m;
                  }),
                };
              }
              return s;
            })
          );

          // Detect if the user explicitly requested a document or briefing dossier in their prompt
          const isExplicitDocRequest = /\b(report|dossier|briefing|bulletin|pdf|document|docx|executive summary)\b/i.test(queryText);
          const isSpatialTask = Boolean(result.audit_trace?.task_identified && result.audit_trace.task_identified !== 'AGENT_ASSISTANT');

          // Auto-compile official SAC-ISRO Intelligence Bulletin only when explicitly requested
          if (isExplicitDocRequest && isSpatialTask && effectiveImages.length > 0) {
            generateReportForCurrentResult(result, effectiveImages).then((repId) => {
              if (!isCancelled && repId) {
                setSessions((prev) =>
                  prev.map((s) => {
                    if (s.id === targetSessionId) {
                      return {
                        ...s,
                        lastReportId: repId,
                        messages: s.messages.map((m) => {
                          if (m.id === assistantMessageId) {
                            return { ...m, reportId: repId };
                          }
                          return m;
                        }),
                      };
                    }
                    return s;
                  })
                );
              }
            });
          }
        },
        onError: (errorMsg: string) => {
          if (isCancelled) return;
          setIsProcessing(false);
          setSessions((prev) =>
            prev.map((s) => {
              if (s.id === targetSessionId) {
                return {
                  ...s,
                  messages: s.messages.map((m) => {
                    if (m.id === assistantMessageId) {
                      return {
                        ...m,
                        content: `**Analysis Error:** ${errorMsg}`,
                        images: [...effectiveImages],
                        isStreaming: false,
                        error: errorMsg,
                      };
                    }
                    return m;
                  }),
                };
              }
              return s;
            })
          );
        },
        onClose: () => {
          if (isCancelled) return;
          setIsProcessing(false);
        },
      },
      conversationHistory
    );

    activeCancelRef.current = () => {
      isCancelled = true;
      try {
        cleanup();
      } catch {
        // ignore
      }
    };
  };

  /**
   * Helper to load pre-bundled server samples and attach them to the chat input bar
   * without auto-submitting or auto-writing queries, allowing the user to type their query freely.
   */
  const loadPresetAnalysis = async (presetType: string) => {
    interface PresetItem {
      files: Array<{ url: string; name: string }>;
      prompt: string;
      title: string;
    }

    const PRESETS: Record<string, PresetItem> = {
      flood: {
        files: [
          { url: '/static/samples/flood_t1.tif', name: 'pre_flood_t1.tif' },
          { url: '/static/samples/flood_t2.tif', name: 'post_flood_t2.tif' },
        ],
        prompt: 'Detect flood inundation, map submerged parcel boundaries, and quantify displacement between pre- and post-disaster dates.',
        title: 'Flood Disaster Inundation Mapping',
      },
      fusion: {
        files: [
          { url: '/static/samples/fusion_optical.tif', name: 'sentinel2_optical_cloudy.tif' },
          { url: '/static/samples/fusion_sar.tif', name: 'sentinel1_sar_backscatter.tif' },
        ],
        prompt: 'Execute cross-modal optical and microwave SAR fusion to penetrate dense cloud cover and reconstruct ground terrain.',
        title: 'Cloud-Penetrating SAR Fusion',
      },
      grounding: {
        files: [
          { url: '/static/samples/port_grounding.tif', name: 'port_grounding.tif' },
        ],
        prompt: 'Locate, outline, and delineate all maritime vessels, storage facilities, and harbor infrastructure with bounding boxes.',
        title: 'Maritime Port & Infrastructure Grounding',
      },
      vqa: {
        files: [
          { url: '/static/samples/forest_vqa.tif', name: 'sentinel2_forest_vqa.tif' },
        ],
        prompt: 'What land cover classes dominate this scene, and what is the estimated vegetation canopy density and spectral signature?',
        title: 'Forest Canopy & Spectral VQA',
      },
      urban: {
        files: [
          { url: '/static/samples/urban_t1.tif', name: 'urban_baseline_t1.tif' },
          { url: '/static/samples/urban_t2.tif', name: 'urban_expansion_t2.tif' },
        ],
        prompt: 'Analyze bi-temporal urban expansion, new residential footprints, and roadway corridors between these acquisition dates.',
        title: 'Urban Sprawl & Expansion Analysis',
      },
      // Aliases for backwards compatibility
      sar: {
        files: [
          { url: '/static/samples/fusion_optical.tif', name: 'cloudy_optical_pass.tif' },
          { url: '/static/samples/fusion_sar.tif', name: 'sentinel1_sar_radar.tif' },
        ],
        prompt: 'Execute cross-modal optical and microwave SAR fusion to penetrate dense cloud cover and reconstruct ground terrain.',
        title: 'Cloud-Penetrating SAR Fusion',
      },
      vegetation: {
        files: [
          { url: '/static/samples/forest_vqa.tif', name: 'sentinel2_forest_vqa.tif' },
        ],
        prompt: 'What land cover classes dominate this scene, and what is the estimated vegetation canopy density and spectral signature?',
        title: 'Forest Canopy & Spectral VQA',
      },
      cloud_free_flood: {
        files: [
          { url: '/static/samples/fusion_optical.tif', name: 'cloudy_optical_pass.tif' },
          { url: '/static/samples/fusion_sar.tif', name: 'sentinel1_sar_radar.tif' },
        ],
        prompt: 'Penetrate storm clouds using Sentinel-1 SAR and Sentinel-2 optical imagery, reconstruct a cloud-free ground view, calculate total flooded area, and pinpoint elevated safe evacuation zones.',
        title: 'Cloud-Free SAR Flood & Safe Zones',
      },
      canopy: {
        files: [
          { url: '/static/samples/forest_vqa.tif', name: 'sentinel2_forest_vqa.tif' },
        ],
        prompt: 'Analyze the vegetation canopy health and delineate land cover categories.',
        title: 'Vegetation & Land Cover Analysis',
      },
    };

    const targetPreset = PRESETS[presetType] || PRESETS.flood;

    try {
      const imagesToSubmit: ImageUploadResponse[] = [];
      for (const item of targetPreset.files) {
        const resp = await fetch(`${item.url}?t=${Date.now()}`, { cache: 'no-store' });
        if (!resp.ok) {
          throw new Error(`Failed to load preset sample raster: ${item.url} (${resp.status} ${resp.statusText})`);
        }
        const blob = await resp.blob();
        const file = new File([blob], item.name, { type: 'image/tiff' });
        const uploaded = await SatQueryAPI.uploadImage(file);
        imagesToSubmit.push(uploaded);
      }

      setActiveImages(imagesToSubmit);
      if (imagesToSubmit.length > 0) {
        setActiveImageForAnalysis(imagesToSubmit[0]);
      }

      setSessions((prev) =>
        prev.map((s) => (s.id === activeSessionId ? { ...s, images: imagesToSubmit } : s))
      );
    } catch (err: any) {
      console.error('Failed to attach sample preset:', err);
    }
  };

  return (
    <ChatContext.Provider
      value={{
        sessions,
        activeSessionId,
        currentSession,
        activeImages,
        isProcessing,
        systemHealth,
        activeAnalysisResult,
        activeImageForAnalysis,
        activeReportId,
        setActiveReportId,
        isGeneratingReport,
        generateReportForCurrentResult,
        isSidebarOpen,
        isAnalysisPanelOpen,
        setSidebarOpen,
        setAnalysisPanelOpen,
        createNewSession,
        selectSession,
        deleteSession,
        renameSession,
        selectMessageResult,
        attachImage,
        removeImage,
        clearAttachedImages,
        submitQuery,
        cancelProcessing,
        setActiveAnalysisResult,
        setActiveImageForAnalysis,
        loadPresetAnalysis,
        pendingQueryText,
        setPendingQueryText,
        isPdfModalOpen,
        pdfModalResult,
        pdfModalImage,
        openPdfModal,
        closePdfModal,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
};

export const useChat = (): ChatContextType => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};
