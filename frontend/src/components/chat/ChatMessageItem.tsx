import React, { useState } from 'react';
import {
  Copy,
  Check,
  Activity,
  FileText,
  Share2,
  Download,
  AlertTriangle,
  Compass,
  Sparkles,
  Maximize2,
  X,
  Info,
  ArrowRight,
  Satellite,
  Layers,
} from 'lucide-react';
import { MarkdownContent } from './MarkdownContent';
import type { ChatMessage } from '../../types/chat';
import { StreamingStatus } from './StreamingStatus';
import { CartographicIntelligenceViewer } from '../visualization/CartographicIntelligenceViewer';
import { SatQueryLogo } from '../ui/SatQueryLogo';
import { Badge } from '../ui/Badge';
import { useChat } from '../../context/ChatContext';
import { EvidenceGraphModal } from './EvidenceGraphModal';
import { BitemporalChangeCard } from './BitemporalChangeCard';
import { DisasterAssessmentCard } from './DisasterAssessmentCard';
import { OpticalSarFusionCard } from './OpticalSarFusionCard';
import { GroundingDinoCard } from './GroundingDinoCard';
import { MultiModelAnalysisCard } from './MultiModelAnalysisCard';
import { DynamicTelemetryChart } from '../visualization/DynamicTelemetryChart';
import { SatQueryAPI } from '../../services/api';

interface ChatMessageItemProps {
  message: ChatMessage;
}

export const ChatMessageItem: React.FC<ChatMessageItemProps> = ({ message }) => {
  const { selectMessageResult, openPdfModal, submitQuery } = useChat();
  const [copied, setCopied] = useState(false);
  const [isEvidenceGraphOpen, setIsEvidenceGraphOpen] = useState(false);
  const [vqaZoomOpen, setVqaZoomOpen] = useState(false);
  const [showDetailedDossier, setShowDetailedDossier] = useState(false);

  const isUser = message.role === 'user';
  const result = message.result;

  const handleCopy = () => {
    navigator.clipboard
      .writeText(message.content)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      })
      .catch(() => {
        // Silently fail if clipboard permission is denied or non-HTTPS
      });
  };

  const handleOpenPdf = () => {
    if (!result) return;
    openPdfModal(result, message.images?.[0]);
  };

  const handleOpenTelemetry = () => {
    if (result) {
      selectMessageResult(message);
    }
  };

  const handleDownloadGeoJson = () => {
    if (!result?.geojson_data) return;
    const blob = new Blob([JSON.stringify(result.geojson_data, null, 2)], {
      type: 'application/geo+json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `satquery_features_${result.audit_trace?.trace_id || 'export'}.geojson`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 200);
  };

  // ── Render User Message (Claude Signature Warm Bubble) ─────
  if (isUser) {
    return (
      <div className="flex justify-end my-5 px-3 sm:px-6">
        <div className="max-w-2xl bg-[var(--bg-user-bubble)] text-[var(--text-main)] rounded-[22px] px-5 py-3.5 shadow-subtle border border-[var(--border-subtle)] hover:border-[var(--border-hover)] transition-colors">
          {message.images && message.images.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-2.5">
              {message.images.map((img) => (
                <div
                  key={img.file_id}
                  className="flex items-center gap-2 py-1 px-2.5 bg-[var(--bg-app)] rounded-lg border border-[var(--border-subtle)] text-[11px]"
                >
                  {img.thumbnail_url && (
                    <img
                      src={img.thumbnail_url}
                      alt={img.filename}
                      className="w-5 h-5 rounded object-cover border border-[var(--border-subtle)]"
                    />
                  )}
                  <span className="font-mono text-[var(--text-main)] font-medium truncate max-w-[150px]">
                    {img.filename}
                  </span>
                  <span className="font-mono text-[10px] text-[#0EA5E9] bg-[#0EA5E9]/10 px-1.5 py-0.2 rounded border border-[#0EA5E9]/20">
                    {img.modality}
                  </span>
                </div>
              ))}
            </div>
          )}

          <p className="text-[14.5px] leading-relaxed whitespace-pre-wrap font-normal text-[var(--text-main)]">
            {message.content}
          </p>
        </div>
      </div>
    );
  }

  // ── Render Assistant Message (Claude Editorial Document Layout) ────
  const spatial = result?.spatial_evidence;
  const trace = result?.audit_trace;
  const primaryImage = message.images?.[0];
  const secondaryImage = message.images?.[1];
  const resolveThumb = (img?: typeof primaryImage, fallbackUrl?: string) => {
    if (img?.thumbnail_url) return img.thumbnail_url;
    if (fallbackUrl) return fallbackUrl;
    if (img?.file_id) return `/api/v1/preview/${img.file_id}`;
    if (img?.filename) {
      const base = img.filename.replace(/\.(tif|tiff|geotiff|png|jpg|jpeg)$/i, '');
      return `/api/v1/preview/${base}.tif`;
    }
    return '';
  };
  const primaryThumb = resolveThumb(primaryImage, result?.thumbnail_urls?.[0]);
  const secondaryThumb = resolveThumb(secondaryImage, result?.thumbnail_urls?.[1]);
  const boxes = spatial?.bounding_boxes || (spatial as any)?.boxes || [];
  const clusters = spatial?.clusters || (spatial?.extra?.clusters as any) || [];

  const confDecomp = result?.confidence_decomposition || trace?.confidence_decomposition;
  const evidenceVerif = result?.evidence_verification || trace?.evidence_verification;
  const evidenceGraph = result?.evidence_graph || trace?.evidence_graph;
  const interpreted = result?.interpreted_query || trace?.interpreted_query;

  // Determine if this is a basic conversational question or actual spatial imagery analysis
  const isConversational =
    !message.isStreaming &&
    (trace?.task_identified === 'AGENT_ASSISTANT' ||
      (!primaryImage && !spatial?.mask_url && (!boxes || boxes.length === 0)));

  const hasGeoJsonFeatures = Boolean(
    result?.geojson_data?.features &&
      Array.isArray(result.geojson_data.features) &&
      result.geojson_data.features.length > 0
  );

  const bitemporalCard = (spatial?.extra?.bitemporal_card as any) || (result?.spatial_evidence?.extra?.bitemporal_card as any);
  const disasterCard = (spatial?.extra?.disaster_card as any) || (result?.spatial_evidence?.extra?.disaster_card as any);
  const opticalSarCard = (spatial?.extra?.optical_sar_card as any) || (result?.spatial_evidence?.extra?.optical_sar_card as any);
  const groundingCard = (spatial?.extra?.grounding_card as any) || (result?.spatial_evidence?.extra?.grounding_card as any);
  const multiModelCard = (spatial?.extra?.multi_model_card as any) || (result?.spatial_evidence?.extra?.multi_model_card as any);
  const vqaGrounding = result?.vqa_grounding || (spatial as any)?.vqa_grounding;
  const chartData = (result?.chart_data || spatial?.chart_data || (spatial?.extra?.chart_data as any)) || null;
  const hasRichCard = Boolean(
    bitemporalCard ||
      disasterCard ||
      vqaGrounding ||
      opticalSarCard ||
      groundingCard ||
      multiModelCard ||
      chartData ||
      (!isConversational && (primaryThumb || spatial?.mask_url || (boxes && boxes.length > 0)))
  );

  return (
    <div className="my-6 px-3 sm:px-6">
      <div className={`w-full ${hasRichCard ? 'max-w-5xl' : 'max-w-3xl'} mx-auto space-y-3.5`}>
        {/* Header Branding */}
        <div className="flex items-center justify-between pb-2 border-b border-[var(--border-subtle)]">
          <div className="flex items-center gap-2.5">
            <SatQueryLogo
              size={18}
              isLoading={message.isStreaming}
            />

            <span className="font-semibold text-xs text-[var(--text-main)] tracking-tight">
              SatQuery AI
            </span>
            {!isConversational && trace?.task_identified && (
              <Badge variant="neutral">{trace.task_identified}</Badge>
            )}
            {!isConversational && trace?.confidence_score !== undefined && (
              <Badge variant="success">
                {(trace.confidence_score * 100).toFixed(0)}% Match
              </Badge>
            )}
          </div>

          <div className="flex items-center gap-1.5 text-[var(--text-muted)]">
            <button
              type="button"
              onClick={handleCopy}
              className="p-1.5 hover:text-[var(--text-main)] hover:bg-[var(--bg-surface)] rounded-md transition-colors text-xs cursor-pointer"
              title="Copy response"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-[#0EA5E9]" /> : <Copy className="w-3.5 h-3.5" />}
            </button>

            {!isConversational && evidenceGraph && (
              <button
                type="button"
                onClick={() => setIsEvidenceGraphOpen(true)}
                className="px-2.5 py-1 text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-[var(--bg-surface)] rounded-md transition-colors text-xs font-mono flex items-center gap-1.5 cursor-pointer"
                title="View Verifiable Evidence Graph DAG"
              >
                <Share2 className="w-3.5 h-3.5 text-[#0EA5E9]" />
                <span className="hidden sm:inline">Evidence Graph</span>
              </button>
            )}

            {!isConversational && hasGeoJsonFeatures && (
              <button
                type="button"
                onClick={handleDownloadGeoJson}
                className="px-2.5 py-1 text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-[var(--bg-surface)] rounded-md transition-colors text-xs font-mono flex items-center gap-1.5 cursor-pointer"
                title="Download GeoJSON Vector FeatureCollection"
              >
                <Download className="w-3.5 h-3.5 text-emerald-500" />
                <span className="hidden sm:inline">GeoJSON</span>
              </button>
            )}

            {!isConversational && result && (primaryThumb || spatial?.mask_url || boxes.length > 0) && (
              <button
                type="button"
                onClick={handleOpenPdf}
                className="px-2.5 py-1 text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-[var(--bg-surface)] rounded-md transition-colors text-xs font-mono flex items-center gap-1.5 cursor-pointer"
                title="View Executive Mission Briefing Dossier"
              >
                <FileText className="w-3.5 h-3.5 text-[#0EA5E9]" />
                <span className="hidden sm:inline">Briefing</span>
              </button>
            )}

            {!isConversational && result && (
              <button
                type="button"
                onClick={handleOpenTelemetry}
                className="px-2.5 py-1 text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-[var(--bg-surface)] rounded-md transition-colors text-xs font-mono flex items-center gap-1.5 cursor-pointer"
                title="View Telemetry Panel"
              >
                <Activity className="w-3.5 h-3.5 text-[#0EA5E9]" />
                <span className="hidden sm:inline">Telemetry</span>
              </button>
            )}
          </div>
        </div>

        {/* Observable Agent Activity Timeline (only for spatial imagery tasks) */}
        {!isConversational && message.streamingSteps && message.streamingSteps.length > 0 && (
          <StreamingStatus
            steps={message.streamingSteps}
            isStreaming={message.isStreaming}
          />
        )}

        {/* Loading Indicator (Clean ChatGPT style for conversational, detailed for satellite intelligence) */}
        {message.isStreaming && !message.content && (
          <div className="flex items-center gap-2.5 py-3 text-xs text-[var(--text-muted)] font-mono">
            {isConversational ? (
              <>
                <span className="w-2 h-2 rounded-full bg-[#0EA5E9] animate-ping" />
                <span className="text-[var(--text-main)]">SatQuery AI is thinking...</span>
              </>
            ) : (
              <>
                <SatQueryLogo size={18} isLoading={true} />
                <span className="text-[var(--text-main)] animate-pulse">
                  SatQuery AI is synthesizing multi-modal satellite intelligence...
                </span>
              </>
            )}
          </div>
        )}

        {/* 🎯 Query Explainability & Task Decomposition (spatial tasks only) */}
        {!isConversational && interpreted && (
          <div className="px-3.5 py-2.5 rounded-xl bg-[var(--bg-user-bubble)]/70 border border-[var(--border-subtle)] text-xs space-y-1 animate-fadeIn">
            <div className="flex items-center justify-between text-[11px] font-mono text-[var(--text-muted)]">
              <span className="flex items-center gap-1.5 text-[#0EA5E9] font-semibold">
                <Compass className="w-3.5 h-3.5" />
                Query Decomposed & Verified
              </span>
              <span className="bg-[var(--bg-surface)] px-2 py-0.5 rounded border border-[var(--border-subtle)] text-[10px]">
                Target: {interpreted.target_features || interpreted.task_type}
              </span>
            </div>
            <p className="text-[12px] text-[var(--text-main)] font-normal leading-relaxed">
              {interpreted.reasoning_summary}
            </p>
          </div>
        )}

        {/* ── Conversational AI Response Stream (Claude / ChatGPT Style) ── */}
        {message.content ? (
          <div className="text-[14.5px] text-[var(--text-main)] leading-[1.75] font-normal space-y-3.5">
            <MarkdownContent content={message.content} isStreaming={message.isStreaming} />
            {message.isStreaming && (
              <span className="inline-flex items-center ml-2 align-baseline">
                <SatQueryLogo size={14} isLoading={true} />
              </span>
            )}
          </div>
        ) : null}

        {/* ── Cartographic Satellite Intelligence Displays (Model-wise & Interactive) ── */}
        {!isConversational && !message.isStreaming && disasterCard && (
          <DisasterAssessmentCard
            data={disasterCard}
            onOpenPdf={result ? () => handleOpenPdf() : undefined}
            onDownloadGeoJson={hasGeoJsonFeatures ? handleDownloadGeoJson : undefined}
          />
        )}

        {!isConversational && !message.isStreaming && groundingCard && (
          <GroundingDinoCard
            data={groundingCard}
            onOpenPdf={result ? () => handleOpenPdf() : undefined}
          />
        )}

        {!isConversational && !message.isStreaming && bitemporalCard && (
          <BitemporalChangeCard
            data={bitemporalCard}
            onOpenPdf={result ? () => handleOpenPdf() : undefined}
          />
        )}

        {!isConversational && !message.isStreaming && !groundingCard && !bitemporalCard && multiModelCard && (
          <MultiModelAnalysisCard
            data={multiModelCard}
            onOpenPdf={result ? () => handleOpenPdf() : undefined}
            onQueryClick={(q) => submitQuery(q, message.images && message.images.length > 0 ? message.images : undefined)}
          />
        )}

        {!isConversational && !message.isStreaming && !multiModelCard && !groundingCard && !bitemporalCard && !disasterCard && (primaryThumb || spatial?.mask_url || vqaGrounding?.overlay_url || boxes.length > 0) && (
          <CartographicIntelligenceViewer
            image1Url={primaryThumb || spatial?.mask_url || ''}
            image2Url={secondaryThumb || undefined}
            maskUrl={vqaGrounding?.overlay_url || spatial?.mask_url}
            boxes={boxes}
            label1={primaryImage?.filename || (primaryThumb ? 'pre_event_t1.tif' : 'spatial_evidence.tif')}
            label2={secondaryImage?.filename || 'surveillance_t2.tif'}
            modality1={primaryImage?.modality || 'OPTICAL'}
            modality2={secondaryImage?.modality || 'SAR'}
            taskType={trace?.task_identified}
            clusters={clusters}
            changedAreaHectares={spatial?.changed_area_hectares}
            changedAreaPercent={spatial?.changed_area_percent}
            confidence={trace?.confidence_score}
            bounds={primaryImage?.bounds_latlon || null}
            gsd_m={(primaryImage as any)?.gsd_m || (primaryImage as any)?.spatial_resolution_m || null}
            fileId={primaryImage?.file_id || null}
            onOpenPdf={result ? () => handleOpenPdf() : undefined}
          />
        )}

        {!isConversational && !message.isStreaming && opticalSarCard && (
          <div className="pt-1">
            <button
              type="button"
              onClick={() => setShowDetailedDossier(!showDetailedDossier)}
              className="text-xs font-mono text-sky-400 hover:text-sky-300 flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sky-950/40 border border-sky-800/40 hover:bg-sky-900/40 transition-colors cursor-pointer"
            >
              <Layers className="w-3.5 h-3.5" />
              <span>{showDetailedDossier ? 'Hide Detailed Sensor Insets & Decomposition Dossier' : 'Inspect Detailed SAR-Optical Physical Insets & Decomposition Dossier'}</span>
            </button>
            {showDetailedDossier && (
              <div className="mt-3">
                <OpticalSarFusionCard
                  data={opticalSarCard}
                  onOpenPdf={result ? () => handleOpenPdf() : undefined}
                />
              </div>
            )}
          </div>
        )}

        {/* Dynamic Geospatial Telemetry Chart */}
        {!isConversational && !message.isStreaming && chartData && (
          <DynamicTelemetryChart data={chartData} />
        )}

        {/* Interactive Follow-Up Suggestion Chips (ChatGPT/Claude Style) */}
        {!message.isStreaming && result?.suggested_actions && result.suggested_actions.length > 0 && (
          <div className="pt-2 flex flex-wrap gap-2 animate-fadeIn">
            {result.suggested_actions.map((action) => (
              <button
                key={action}
                type="button"
                onClick={() => submitQuery(action, message.images && message.images.length > 0 ? message.images : undefined)}
                className="px-3 py-1.5 rounded-full text-xs bg-[var(--bg-user-bubble)]/70 hover:bg-[var(--bg-user-bubble)] text-[var(--text-main)] border border-[var(--border-subtle)] hover:border-[#0EA5E9]/40 hover:text-[#0EA5E9] transition-all flex items-center gap-1.5 group text-left cursor-pointer shadow-subtle hover:shadow-md"
              >
                <span className="text-[#0EA5E9] opacity-70 group-hover:opacity-100 font-mono text-[10px]">✦</span>
                <span>{action}</span>
              </button>
            ))}
          </div>
        )}

        {/* Subtle Footer (Execution Latency & Trace ID for audit verification) */}
        {!isConversational && trace && (
          <div className="pt-2 flex items-center justify-between text-[10.5px] font-mono text-[var(--text-dim)] border-t border-[var(--border-subtle)]/50">
            <span>Execution Latency: {trace.total_execution_time_ms.toFixed(0)} ms</span>
            <span>Trace ID: {trace.trace_id}</span>
          </div>
        )}
      </div>

      {/* Evidence Graph Modal */}
      <EvidenceGraphModal
        isOpen={isEvidenceGraphOpen}
        onClose={() => setIsEvidenceGraphOpen(false)}
        graph={evidenceGraph}
        traceId={trace?.trace_id}
        taskType={trace?.task_identified}
      />

      {/* VQA High-Resolution Overlay Zoom Modal */}
      {vqaZoomOpen && vqaGrounding && (
        <div
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4"
          onClick={() => setVqaZoomOpen(false)}
        >
          <div
            className="relative max-w-4xl w-full bg-slate-950 rounded-2xl overflow-hidden border border-white/20 shadow-2xl p-2"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              type="button"
              onClick={() => setVqaZoomOpen(false)}
              className="absolute top-4 right-4 p-2 rounded-full bg-black/60 text-white hover:bg-black/80 transition-colors z-10 cursor-pointer"
              title="Close"
            >
              <X className="w-5 h-5" />
            </button>
            <img
              src={SatQueryAPI.getRasterPreviewUrl(vqaGrounding.overlay_url)}
              alt={vqaGrounding.legend_label}
              className="w-full max-h-[80vh] object-contain rounded-xl"
              onError={(e) => {
                const target = e.currentTarget;
                if (!target.dataset.fallbackApplied) {
                  target.dataset.fallbackApplied = 'true';
                  target.src = primaryThumb || SatQueryAPI.getRasterPreviewUrl('cartosat_t1.tif');
                }
              }}
            />
            <div className="p-3 bg-slate-900 border-t border-slate-800 flex items-center justify-between text-xs font-mono text-white">
              <div className="flex items-center gap-2">
                <span
                  className="w-3 h-3 rounded-sm"
                  style={{ backgroundColor: vqaGrounding.legend_color || '#ef4444' }}
                />
                <span className="font-semibold">{vqaGrounding.legend_label}</span>
              </div>
              <span className="text-slate-400">Method: {vqaGrounding.method} | {(vqaGrounding.confidence * 100).toFixed(0)}% Confidence</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
