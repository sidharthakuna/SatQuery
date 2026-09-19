import React, { useState, useEffect, useRef } from 'react';
import {
  X,
  FileDown,
  Satellite,
  MapPin,
  CheckCircle2,
  Activity,
  Copy,
  Check,
  FileText,
  ExternalLink,
  Maximize2,
  Sparkles,
  AlertTriangle,
  ShieldCheck,
} from 'lucide-react';
import { useChat } from '../../context/ChatContext';
import { Badge } from '../ui/Badge';
import { CartographicIntelligenceViewer } from '../visualization/CartographicIntelligenceViewer';
import { MarkdownContent } from '../chat/MarkdownContent';
import { SatQueryAPI } from '../../services/api';

export const AnalysisPanel: React.FC = () => {
  const {
    isAnalysisPanelOpen,
    setAnalysisPanelOpen,
    activeAnalysisResult,
    activeImageForAnalysis,
    activeImages,
    openPdfModal,
    activeReportId,
    isGeneratingReport,
    generateReportForCurrentResult,
  } = useChat();

  const [activeTab, setActiveTab] = useState<'document' | 'canvas' | 'telemetry' | 'audit' | 'json'>('canvas');
  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false);
  const [copiedJson, setCopiedJson] = useState(false);
  const [isPanelExpanded, setIsPanelExpanded] = useState(false);
  const lastTraceIdRef = useRef<string | null>(null);

  useEffect(() => {
    const currentTraceId = activeAnalysisResult?.audit_trace?.trace_id;
    if (currentTraceId && currentTraceId !== lastTraceIdRef.current) {
      lastTraceIdRef.current = currentTraceId;
      if (activeReportId) {
        setActiveTab('document');
      } else {
        setActiveTab('canvas');
      }
    }
  }, [activeAnalysisResult, activeReportId]);

  if (!isAnalysisPanelOpen) return null;

  const result = activeAnalysisResult;
  const trace = result?.audit_trace;
  const targetImage = activeImageForAnalysis || activeImages[0];
  const secondaryImage = activeImages[1];
  const spatial = result?.spatial_evidence;

  const primaryThumb = targetImage?.thumbnail_url || result?.thumbnail_urls?.[0] || '';
  const secondaryThumb = secondaryImage?.thumbnail_url || result?.thumbnail_urls?.[1] || '';
  const boxes = spatial?.bounding_boxes || (spatial as any)?.boxes || [];
  const clusters = spatial?.clusters || (spatial?.extra?.clusters as any) || [];

  const handleDownloadPdf = async () => {
    if (!result) return;
    setIsGeneratingPdf(true);
    try {
      const imgs = activeImages.length > 0 ? activeImages : (targetImage ? [targetImage] : []);
      const imageMetadata = imgs.map((img) => ({
        file_id: img.file_id,
        filename: img.filename,
        modality: img.modality,
        width: img.width,
        height: img.height,
        band_count: img.band_count,
        crs: img.crs,
        thumbnail_url: img.thumbnail_url,
      }));

      const thumbPaths: string[] = [];
      if (result.thumbnail_urls) {
        result.thumbnail_urls.forEach((u) => { if (u && !thumbPaths.includes(u)) thumbPaths.push(u); });
      }
      imgs.forEach((img) => {
        const u = img.thumbnail_url || img.file_id;
        if (u && !thumbPaths.includes(u)) thumbPaths.push(u);
      });

      const maskPath = result.spatial_evidence?.mask_url || undefined;

      const resp = await SatQueryAPI.generateReport({
        query: result.query,
        text_response: result.text_response,
        audit_trace: result.audit_trace as any,
        spatial_evidence: result.spatial_evidence as any,
        mask_image_path: maskPath,
        thumbnail_paths: thumbPaths.length > 0 ? thumbPaths : undefined,
        image_metadata: imageMetadata.length > 0 ? imageMetadata : undefined,
      });
      const downloadUrl = SatQueryAPI.getReportDownloadUrl(resp.report_id);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = `SatQuery_Briefing_${resp.report_id}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (err: any) {
      alert(err.message || 'Failed to download PDF briefing');
    } finally {
      setIsGeneratingPdf(false);
    }
  };

  const handleCopyJson = () => {
    if (!result) return;
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    setCopiedJson(true);
    setTimeout(() => setCopiedJson(false), 2000);
  };

  return (
    <aside
      className={`h-full bg-[var(--bg-panel)] border-l border-[var(--border-subtle)] flex flex-col shrink-0 z-30 select-none overflow-hidden transition-all duration-300 ${
        isPanelExpanded
          ? 'w-full md:w-[700px] lg:w-[820px] xl:w-[940px]'
          : 'w-88 lg:w-[480px] xl:w-[540px]'
      }`}
    >
      {/* ── Top Header (Claude Style) ──────────────── */}
      <div className="px-4 py-3 border-b border-[var(--border-subtle)] flex items-center justify-between bg-[var(--bg-panel)] shrink-0">
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-lg bg-[var(--bg-card)] border border-[var(--border-subtle)] flex items-center justify-center text-[#cc785c]">
            <Activity className="w-3.5 h-3.5" />
          </div>
          <div>
            <h3 className="font-semibold text-xs text-[var(--text-main)] leading-none">
              Geospatial Workspace & Telemetry
            </h3>
            <span className="text-[10px] font-mono text-[var(--text-dim)] leading-tight mt-0.5 block">
              {trace?.task_identified || 'Mission Intelligence Canvas'}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          {result && (
            <button
              type="button"
              onClick={() => openPdfModal(result, targetImage)}
              className="px-2.5 py-1 rounded-lg bg-[#cc785c] hover:bg-[#b8674d] text-white text-[11px] font-medium flex items-center gap-1 shadow-sm transition-colors cursor-pointer"
              title="Open Executive Mission Briefing Dossier (PDF)"
            >
              <FileText className="w-3 h-3 text-white" />
              <span>PDF Dossier</span>
            </button>
          )}

          <button
            type="button"
            onClick={() => setIsPanelExpanded(!isPanelExpanded)}
            className="p-1.5 text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-[var(--bg-surface)] rounded-lg transition-colors cursor-pointer"
            title={isPanelExpanded ? 'Standard width' : 'Widen panel for document viewing'}
          >
            <Maximize2 className="w-4 h-4" />
          </button>

          <button
            type="button"
            onClick={() => setAnalysisPanelOpen(false)}
            className="p-1.5 text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-[var(--bg-surface)] rounded-lg transition-colors cursor-pointer"
            title="Close panel"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* ── Segmented Tab Switcher (Claude style) ──── */}
      <div className="px-3 pt-2.5 pb-2 border-b border-[var(--border-subtle)] bg-[var(--bg-panel)] shrink-0">
        <div className="flex items-center p-0.5 bg-[var(--bg-surface)] border border-[var(--border-subtle)] rounded-xl text-xs overflow-x-auto">
          <button
            type="button"
            onClick={() => setActiveTab('document')}
            className={`flex-1 py-1.5 px-2 rounded-lg font-medium text-[11px] transition-all cursor-pointer whitespace-nowrap flex items-center justify-center gap-1.5 ${
              activeTab === 'document'
                ? 'bg-[var(--bg-card)] text-[#cc785c] font-semibold shadow-subtle'
                : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
            }`}
            title="Official SAC-ISRO Satellite Rapid Surveillance Bulletin"
          >
            <FileText className="w-3.5 h-3.5 text-[#cc785c]" />
            <span>Output Document</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('canvas')}
            className={`flex-1 py-1.5 px-2 rounded-lg font-medium text-[11px] transition-all cursor-pointer whitespace-nowrap flex items-center justify-center gap-1.5 ${
              activeTab === 'canvas'
                ? 'bg-[var(--bg-card)] text-[#cc785c] font-semibold shadow-subtle'
                : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
            }`}
            title="Cartographic Studio (1-2-3 Flow, Swipe Compare, Slippy Map)"
          >
            <Satellite className="w-3.5 h-3.5 text-[#cc785c]" />
            <span>Studio</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('telemetry')}
            className={`flex-1 py-1.5 px-2 rounded-lg font-medium text-[11px] transition-all cursor-pointer whitespace-nowrap ${
              activeTab === 'telemetry'
                ? 'bg-[var(--bg-card)] text-[#cc785c] font-semibold shadow-subtle'
                : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
            }`}
          >
            Telemetry
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('audit')}
            className={`flex-1 py-1.5 px-2 rounded-lg font-medium text-[11px] transition-all cursor-pointer whitespace-nowrap ${
              activeTab === 'audit'
                ? 'bg-[var(--bg-card)] text-[#cc785c] font-semibold shadow-subtle'
                : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
            }`}
          >
            Audit Steps {trace?.execution_steps ? `(${trace.execution_steps.length})` : ''}
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('json')}
            className={`flex-1 py-1.5 px-2 rounded-lg font-medium text-[11px] transition-all cursor-pointer whitespace-nowrap ${
              activeTab === 'json'
                ? 'bg-[var(--bg-card)] text-[#cc785c] font-semibold shadow-subtle'
                : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
            }`}
          >
            JSON
          </button>
        </div>
      </div>

      {/* ── Scrollable Tab Body ──────────────────────────────── */}
      <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-3.5 text-xs">
        {/* TAB 0: OFFICIAL SAC-ISRO OUTPUT DOCUMENT BULLETIN */}
        {activeTab === 'document' && (
          <div className="h-full flex flex-col space-y-3">
            {activeReportId ? (
              <div className="flex-1 flex flex-col bg-[var(--bg-card)] rounded-2xl border border-[var(--border-subtle)] overflow-hidden shadow-claude min-h-[580px]">
                {/* Document Sub-toolbar */}
                <div className="px-3.5 py-2 bg-[var(--bg-panel)] border-b border-[var(--border-subtle)] flex items-center justify-between gap-2 shrink-0">
                  <div className="flex items-center gap-2 overflow-hidden">
                    <span className="w-2 h-2 rounded-full bg-[#cc785c] animate-pulse shrink-0" />
                    <span className="font-semibold text-xs text-[var(--text-main)] truncate">
                      SAC-ISRO Rapid Surveillance Bulletin
                    </span>
                    <Badge variant="neutral" className="text-[9px] hidden sm:inline font-mono">
                      A4 OFFICIAL
                    </Badge>
                  </div>
                  <div className="flex items-center gap-1.5 shrink-0">
                    <a
                      href={SatQueryAPI.getReportViewUrl(activeReportId)}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-2 py-1 text-[11px] font-medium text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-[var(--bg-surface)] rounded-lg transition-colors flex items-center gap-1 cursor-pointer border border-[var(--border-subtle)]"
                      title="Open in new window"
                    >
                      <ExternalLink className="w-3 h-3 text-[#cc785c]" />
                      <span className="hidden sm:inline">New Tab</span>
                    </a>
                    <button
                      type="button"
                      onClick={() => {
                        const link = document.createElement('a');
                        link.href = SatQueryAPI.getReportDownloadUrl(activeReportId);
                        link.download = `SatQuery_Bulletin_${activeReportId}.pdf`;
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                      }}
                      className="px-2 py-1 text-[11px] font-medium bg-[#cc785c] hover:bg-[#b8674d] text-white rounded-lg transition-colors flex items-center gap-1 cursor-pointer shadow-sm"
                      title="Download Official PDF"
                    >
                      <FileDown className="w-3 h-3 text-white" />
                      <span>Download PDF</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => result && openPdfModal(result, targetImage)}
                      className="p-1.5 text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-[var(--bg-surface)] rounded-lg transition-colors cursor-pointer border border-[var(--border-subtle)]"
                      title="Open Full Dossier Modal"
                    >
                      <Maximize2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>

                {/* PDF Viewer Embed with full browser toolbar (Fit Width) */}
                <div className="flex-1 relative min-h-[580px] bg-neutral-900 overflow-hidden flex flex-col">
                  <iframe
                    src={`${SatQueryAPI.getReportViewUrl(activeReportId)}#view=FitH&toolbar=1`}
                    className="w-full h-full border-0 bg-white"
                    title="SAC-ISRO Official Intelligence Bulletin"
                  />
                </div>
              </div>
            ) : isGeneratingReport ? (
              <div className="p-10 rounded-2xl border border-[var(--border-subtle)] bg-[var(--bg-card)] text-center space-y-3">
                <div className="w-12 h-12 rounded-full border-2 border-[#cc785c]/30 border-t-[#cc785c] animate-spin mx-auto flex items-center justify-center">
                  <Satellite className="w-5 h-5 text-[#cc785c]" />
                </div>
                <h4 className="font-semibold text-xs text-[var(--text-main)]">
                  Compiling SAC-ISRO Rapid Surveillance Bulletin...
                </h4>
                <p className="text-[11px] text-[var(--text-dim)] max-w-xs mx-auto font-mono">
                  Formatting satellite surveillance evidence, sensor metrics, and key findings into the official publication document.
                </p>
              </div>
            ) : result ? (
              <div className="p-8 rounded-2xl border border-[var(--border-subtle)] bg-[var(--bg-card)] text-center space-y-3">
                <FileText className="w-10 h-10 text-[#cc785c] mx-auto" />
                <h4 className="font-semibold text-xs text-[var(--text-main)]">
                  Official SAC-ISRO Intelligence Bulletin Ready
                </h4>
                <p className="text-[11px] text-[var(--text-dim)] max-w-xs mx-auto">
                  Click below to compile the official high-resolution satellite surveillance bulletin document.
                </p>
                <button
                  type="button"
                  onClick={() => generateReportForCurrentResult(result, activeImages)}
                  className="px-4 py-2 rounded-xl bg-[#cc785c] hover:bg-[#b8674d] text-white font-medium text-xs shadow-sm transition-colors cursor-pointer inline-flex items-center gap-2"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Generate Output Document</span>
                </button>
              </div>
            ) : (
              <div className="p-8 rounded-2xl border border-dashed border-[var(--border-subtle)] text-center text-[var(--text-muted)] space-y-2">
                <Satellite className="w-8 h-8 text-[var(--text-dim)] mx-auto" />
                <p className="font-medium text-xs text-[var(--text-main)]">
                  No analysis document compiled yet
                </p>
                <p className="text-[11px] text-[var(--text-dim)] max-w-xs mx-auto">
                  Submit a query or load a sample mission to generate and inspect the official SAC-ISRO output document.
                </p>
              </div>
            )}
          </div>
        )}

        {/* TAB 1: INTERACTIVE CANVAS STUDIO */}
        {activeTab === 'canvas' && (
          <div className="space-y-3">
            {primaryThumb ? (
              <CartographicIntelligenceViewer
                image1Url={primaryThumb}
                image2Url={secondaryThumb || undefined}
                maskUrl={result?.vqa_grounding?.overlay_url || spatial?.mask_url || (spatial as any)?.vqa_grounding?.overlay_url}
                boxes={boxes}
                label1={targetImage?.filename || 'baseline_t1.tif'}
                label2={secondaryImage?.filename}
                modality1={targetImage?.modality || 'OPTICAL'}
                modality2={secondaryImage ? secondaryImage.modality : undefined}
                taskType={result?.interpreted_query?.task_type || trace?.task_identified || spatial?.type || (result as any)?.task_type}
                clusters={clusters}
                changedAreaHectares={spatial?.changed_area_hectares}
                changedAreaPercent={spatial?.changed_area_percent}
                confidence={trace?.confidence_score}
                descriptionNode={
                  result?.text_response ? (
                    <div className="space-y-2 font-sans">
                      <div className="flex items-center justify-between text-[11px] font-mono pb-1.5 border-b border-[var(--border-subtle)]">
                        <span className="text-[#cc785c] font-semibold flex items-center gap-1.5">
                          <Sparkles className="w-3.5 h-3.5 text-[#cc785c]" />
                          {trace?.task_identified === 'SINGLE_VQA' || (result as any)?.task_type === 'SINGLE_VQA'
                            ? 'RS-VLM Biophysical Synthesis'
                            : 'AI Analytical Findings'}
                        </span>
                        {trace?.confidence_score && (
                          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[#cc785c]/10 text-[#cc785c] border border-[#cc785c]/25 font-semibold">
                            {(trace.confidence_score * 100).toFixed(0)}% Confidence
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-[var(--text-main)] leading-relaxed pt-0.5">
                        <MarkdownContent content={result.text_response} compact={true} />
                      </div>
                    </div>
                  ) : undefined
                }
                bounds={targetImage?.bounds_latlon || null}
                fileId={targetImage?.file_id || null}
                onOpenPdf={result ? () => openPdfModal(result, targetImage) : undefined}
              />
            ) : (
              <div className="p-8 rounded-2xl border border-dashed border-[var(--border-subtle)] text-center text-[var(--text-muted)] space-y-2">
                <Satellite className="w-8 h-8 text-[var(--text-dim)] mx-auto" />
                <p className="font-medium text-xs text-[var(--text-main)]">
                  No active imagery loaded
                </p>
                <p className="text-[11px] text-[var(--text-dim)] max-w-xs mx-auto">
                  Attach a GeoTIFF raster from the input bar or launch a sample mission to inspect the interactive canvas.
                </p>
              </div>
            )}
          </div>
        )}

        {/* TAB 2: SENSOR TELEMETRY & SCIENCE */}
        {activeTab === 'telemetry' && (
          <>
            {/* Active Image Specifications */}
            {targetImage ? (
              <div className="rounded-2xl p-4 bg-[var(--bg-card)] border border-[var(--border-subtle)] shadow-subtle space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-medium text-[var(--text-muted)]">
                    Primary Satellite Raster
                  </span>
                  <Badge variant="neutral">{targetImage.modality}</Badge>
                </div>

                <div className="flex items-center gap-3">
                  {targetImage.thumbnail_url || targetImage.filename ? (
                    <img
                      src={SatQueryAPI.getRasterPreviewUrl(targetImage.thumbnail_url || targetImage.filename || 'cartosat_t1.tif')}
                      alt={targetImage.filename}
                      className="w-12 h-12 rounded-xl object-cover border border-[var(--border-subtle)] bg-black"
                      onError={(e) => {
                        (e.currentTarget as HTMLImageElement).src = SatQueryAPI.getRasterPreviewUrl('cartosat_t1.tif');
                      }}
                    />
                  ) : (
                    <div className="w-12 h-12 rounded-xl bg-[var(--bg-surface)] border border-[var(--border-subtle)] flex items-center justify-center text-[var(--text-muted)]">
                      <Satellite className="w-6 h-6" />
                    </div>
                  )}
                  <div className="overflow-hidden">
                    <span className="font-semibold text-[var(--text-main)] block truncate text-xs">
                      {targetImage.filename}
                    </span>
                    <span className="text-[10px] font-mono text-[var(--text-dim)] block">
                      ID: {targetImage.file_id}
                    </span>
                  </div>
                </div>

                <div className="divide-y divide-[var(--border-subtle)] text-[11px] font-mono pt-1">
                  <div className="py-1.5 flex justify-between">
                    <span className="text-[var(--text-muted)]">CRS / Projection</span>
                    <span className="text-[var(--text-main)] font-semibold">{targetImage.crs}</span>
                  </div>
                  <div className="py-1.5 flex justify-between">
                    <span className="text-[var(--text-muted)]">Pixel Resolution</span>
                    <span className="text-[var(--text-main)]">{targetImage.width} × {targetImage.height} px</span>
                  </div>
                  <div className="py-1.5 flex justify-between">
                    <span className="text-[var(--text-muted)]">Spectral Channels</span>
                    <span className="text-[var(--text-main)]">{targetImage.band_count} Bands</span>
                  </div>
                  <div className="py-1.5 flex justify-between">
                    <span className="text-[var(--text-muted)]">File Size</span>
                    <span className="text-[var(--text-main)]">
                      {(targetImage.file_size_bytes / (1024 * 1024)).toFixed(1)} MB
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-4 rounded-xl border border-dashed border-[var(--border-subtle)] text-center text-[var(--text-dim)] text-xs">
                No active imagery loaded
              </div>
            )}

            {/* Geographic Coverage Range */}
            {targetImage?.bounds_latlon && (
              <div className="rounded-2xl p-4 bg-[var(--bg-card)] border border-[var(--border-subtle)] shadow-subtle space-y-2.5">
                <div className="flex items-center gap-2 text-[var(--text-main)]">
                  <MapPin className="w-4 h-4 text-[#cc785c]" />
                  <span className="text-xs font-semibold">Geographic Coverage (WGS84)</span>
                </div>

                <div className="grid grid-cols-2 gap-2 text-[11px] font-mono bg-[var(--bg-surface)] p-3 rounded-xl border border-[var(--border-subtle)]">
                  <div>
                    <span className="text-[var(--text-dim)] text-[9px] block uppercase tracking-wider">LATITUDE</span>
                    <span className="text-[var(--text-main)] font-medium">
                      {targetImage.bounds_latlon.min_lat.toFixed(3)}° to {targetImage.bounds_latlon.max_lat.toFixed(3)}°
                    </span>
                  </div>
                  <div>
                    <span className="text-[var(--text-dim)] text-[9px] block uppercase tracking-wider">LONGITUDE</span>
                    <span className="text-[var(--text-main)] font-medium">
                      {targetImage.bounds_latlon.min_lon.toFixed(3)}° to {targetImage.bounds_latlon.max_lon.toFixed(3)}°
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Agent Execution Metadata */}
            {trace && (
              <div className="rounded-2xl p-4 bg-[var(--bg-card)] border border-[var(--border-subtle)] shadow-subtle space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-[var(--text-main)]">
                    Model Classification & Confidence
                  </span>
                  <Badge variant="neutral">{trace.task_identified}</Badge>
                </div>

                <div className="divide-y divide-[var(--border-subtle)] text-[11px] font-mono">
                  <div className="py-1.5 flex justify-between">
                    <span className="text-[var(--text-muted)]">Total Pipeline Latency</span>
                    <span className="text-[var(--text-main)] font-semibold">{trace.total_execution_time_ms.toFixed(0)} ms</span>
                  </div>
                  <div className="py-1.5 flex justify-between">
                    <span className="text-[var(--text-muted)]">Model Match Confidence</span>
                    <span className="text-[#cc785c] font-semibold">
                      {(trace.confidence_score * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="py-1.5 flex justify-between">
                    <span className="text-[var(--text-muted)]">Trace ID</span>
                    <span className="text-[var(--text-main)] font-mono">{trace.trace_id}</span>
                  </div>
                </div>

                <div>
                  <span className="text-[10px] text-[var(--text-dim)] uppercase tracking-wider block mb-1.5 font-mono">
                    Specialist Tool Pipeline
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {trace.selected_tools.map((t) => (
                      <span
                        key={t}
                        className="px-2.5 py-1 rounded-lg bg-[var(--bg-surface)] text-[var(--text-main)] border border-[var(--border-subtle)] font-mono text-[10.5px]"
                      >
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Calibrated Confidence Decomposition */}
            {(result?.confidence_decomposition || trace?.confidence_decomposition) && (
              <div className="rounded-2xl p-4 bg-[var(--bg-card)] border border-[var(--border-subtle)] shadow-subtle space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-[var(--text-main)]">
                    Calibrated Confidence
                  </span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-600 border border-emerald-500/20 font-bold">
                    {(((result?.confidence_decomposition || trace?.confidence_decomposition)?.overall_confidence || 0) * 100).toFixed(0)}% OVERALL
                  </span>
                </div>
                <div className="space-y-2 text-[11px] font-mono">
                  {/* Model */}
                  <div>
                    <div className="flex justify-between text-[var(--text-muted)] mb-1">
                      <span>Model Confidence (40%)</span>
                      <span className="font-semibold text-[var(--text-main)]">
                        {(((result?.confidence_decomposition || trace?.confidence_decomposition)?.model_confidence || 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="w-full bg-[var(--bg-surface)] h-1.5 rounded-full overflow-hidden">
                      <div
                        className="h-1.5 bg-[#cc785c] rounded-full"
                        style={{ width: `${((result?.confidence_decomposition || trace?.confidence_decomposition)?.model_confidence || 0) * 100}%` }}
                      />
                    </div>
                  </div>
                  {/* Spatial */}
                  <div>
                    <div className="flex justify-between text-[var(--text-muted)] mb-1">
                      <span>Spatial Agreement (25%)</span>
                      <span className="font-semibold text-[var(--text-main)]">
                        {(((result?.confidence_decomposition || trace?.confidence_decomposition)?.spatial_agreement || 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="w-full bg-[var(--bg-surface)] h-1.5 rounded-full overflow-hidden">
                      <div
                        className="h-1.5 bg-emerald-500 rounded-full"
                        style={{ width: `${((result?.confidence_decomposition || trace?.confidence_decomposition)?.spatial_agreement || 0) * 100}%` }}
                      />
                    </div>
                  </div>
                  {/* Quality */}
                  <div>
                    <div className="flex justify-between text-[var(--text-muted)] mb-1">
                      <span>Input Quality (20%)</span>
                      <span className="font-semibold text-[var(--text-main)]">
                        {(((result?.confidence_decomposition || trace?.confidence_decomposition)?.input_quality || 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="w-full bg-[var(--bg-surface)] h-1.5 rounded-full overflow-hidden">
                      <div
                        className="h-1.5 bg-sky-500 rounded-full"
                        style={{ width: `${((result?.confidence_decomposition || trace?.confidence_decomposition)?.input_quality || 0) * 100}%` }}
                      />
                    </div>
                  </div>
                  {/* Cross modal */}
                  <div>
                    <div className="flex justify-between text-[var(--text-muted)] mb-1">
                      <span>Cross-Modal Consensus (15%)</span>
                      <span className="font-semibold text-[var(--text-main)]">
                        {(((result?.confidence_decomposition || trace?.confidence_decomposition)?.cross_modal_agreement || 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="w-full bg-[var(--bg-surface)] h-1.5 rounded-full overflow-hidden">
                      <div
                        className="h-1.5 bg-purple-500 rounded-full"
                        style={{ width: `${((result?.confidence_decomposition || trace?.confidence_decomposition)?.cross_modal_agreement || 0) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Input Gate Validation & Remediation Advice */}
            {trace?.input_validation && (
              <div className="rounded-2xl p-4 bg-[var(--bg-card)] border border-[var(--border-subtle)] shadow-subtle space-y-2.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="w-4 h-4 text-emerald-500" />
                    <span className="text-xs font-semibold text-[var(--text-main)]">
                      Input Intelligence Gate
                    </span>
                  </div>
                  <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full ${
                    trace.input_validation.is_valid
                      ? 'bg-emerald-500/10 text-emerald-600 border border-emerald-500/20'
                      : 'bg-rose-500/10 text-rose-600 border border-rose-500/20'
                  }`}>
                    {trace.input_validation.is_valid ? 'PASSED' : 'FLAGGED'}
                  </span>
                </div>
                <div className="divide-y divide-[var(--border-subtle)] text-[11px] font-mono">
                  <div className="py-1 flex justify-between">
                    <span className="text-[var(--text-muted)]">CRS Compatible</span>
                    <span className="text-[var(--text-main)] font-semibold">{trace.input_validation.crs_compatible ? 'Yes (EPSG aligned)' : 'Auto-reprojected'}</span>
                  </div>
                  {trace.input_validation.input_quality_score !== undefined && (
                    <div className="py-1 flex justify-between">
                      <span className="text-[var(--text-muted)]">Quality Index</span>
                      <span className="text-[#cc785c] font-semibold">{(trace.input_validation.input_quality_score * 100).toFixed(0)}%</span>
                    </div>
                  )}
                </div>
                {trace.input_validation.remediation_advice && trace.input_validation.remediation_advice.length > 0 && (
                  <div className="pt-2 border-t border-[var(--border-subtle)] space-y-1">
                    <span className="text-[10px] text-amber-600 dark:text-amber-400 font-semibold uppercase tracking-wider block">
                      Remediation Advice
                    </span>
                    <ul className="list-disc list-inside text-[11px] text-[var(--text-muted)] space-y-0.5">
                      {trace.input_validation.remediation_advice.map((adv, aIdx) => (
                        <li key={aIdx}>{adv}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </>
        )}

        {/* TAB 3: AUDIT STEPS */}
        {activeTab === 'audit' && (
          <div className="rounded-2xl p-4 bg-[var(--bg-card)] border border-[var(--border-subtle)] shadow-subtle space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-[var(--border-subtle)]">
              <span className="text-xs font-semibold text-[var(--text-main)]">
                Execution Audit Trail
              </span>
              <span className="text-[10px] font-mono text-[var(--text-dim)]">
                {trace?.execution_steps?.length || 0} Steps
              </span>
            </div>

            {trace?.execution_steps && trace.execution_steps.length > 0 ? (
              <div className="space-y-1.5">
                {trace.execution_steps.map((step, idx) => (
                  <div
                    key={idx}
                    className="p-2.5 rounded-xl bg-[var(--bg-surface)] border border-[var(--border-subtle)] flex items-center justify-between text-[11px] font-mono"
                  >
                    <div className="flex items-center gap-2 truncate">
                      <CheckCircle2 className="w-3.5 h-3.5 text-[#cc785c] shrink-0" />
                      <span className="truncate text-[var(--text-main)] font-medium">
                        {step.step_name}
                      </span>
                    </div>
                    {step.duration_ms > 0 && (
                      <span className="text-[10px] text-[var(--text-dim)] shrink-0 font-mono">
                        +{step.duration_ms.toFixed(0)}ms
                      </span>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-[var(--text-dim)] text-xs py-6 text-center">
                No execution steps recorded yet
              </p>
            )}
          </div>
        )}

        {/* TAB 4: RAW JSON */}
        {activeTab === 'json' && (
          <div className="space-y-2">
            <div className="flex justify-end">
              <button
                type="button"
                onClick={handleCopyJson}
                className="px-3 py-1.5 rounded-lg text-[11px] font-mono bg-[var(--bg-surface)] hover:bg-[var(--bg-card)] text-[var(--text-main)] border border-[var(--border-subtle)] flex items-center gap-1.5 transition-colors cursor-pointer"
              >
                {copiedJson ? <Check className="w-3 h-3 text-[#cc785c]" /> : <Copy className="w-3 h-3" />}
                <span>{copiedJson ? 'Copied' : 'Copy JSON'}</span>
              </button>
            </div>

            <pre className="p-3.5 bg-[var(--bg-card)] rounded-2xl text-[10.5px] font-mono text-[var(--text-main)] overflow-x-auto max-h-[480px] border border-[var(--border-subtle)] leading-relaxed shadow-subtle">
              {result ? JSON.stringify(result, null, 2) : '// No active result payload'}
            </pre>
          </div>
        )}
      </div>

      {/* ── Footer Actions ───────── */}
      {result && (
        <div className="p-3.5 border-t border-[var(--border-subtle)] bg-[var(--bg-panel)] flex items-center gap-2.5 shrink-0">
          <button
            type="button"
            onClick={() => openPdfModal(result, targetImage)}
            className="flex-1 py-2 px-3.5 rounded-xl bg-[#cc785c] hover:bg-[#b8674d] text-white font-medium text-xs transition-colors flex items-center justify-center gap-1.5 shadow-sm cursor-pointer"
            title="Preview Executive Briefing Dossier"
          >
            <FileText className="w-3.5 h-3.5 text-white" />
            <span>Executive Briefing</span>
          </button>
          <button
            type="button"
            onClick={handleDownloadPdf}
            disabled={isGeneratingPdf}
            className="py-2 px-3.5 rounded-xl bg-[var(--bg-card)] hover:bg-[var(--bg-surface)] text-[var(--text-main)] font-medium text-xs transition-colors flex items-center justify-center gap-1.5 border border-[var(--border-subtle)] shadow-subtle disabled:opacity-50 cursor-pointer"
            title="Download PDF directly"
          >
            <FileDown className="w-3.5 h-3.5 text-[#cc785c]" />
            <span>{isGeneratingPdf ? 'Exporting...' : 'PDF'}</span>
          </button>
        </div>
      )}
    </aside>
  );
};
