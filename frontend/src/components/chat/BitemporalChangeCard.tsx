import React, { useState } from 'react';
import {
  FileText,
  Layers,
  Calendar,
  MapPin,
  TrendingUp,
  Maximize2,
  X,
} from 'lucide-react';
import { SatQueryLogo } from '../ui/SatQueryLogo';
import { SatQueryAPI } from '../../services/api';

const resolvePreview = (url?: string, fallback: string = 'cartosat_t1.tif'): string => {
  if (!url) return SatQueryAPI.getRasterPreviewUrl(fallback);
  return SatQueryAPI.getRasterPreviewUrl(url);
};

const handleImgError = (e: React.SyntheticEvent<HTMLImageElement>, fallback: string = 'cartosat_t1.tif') => {
  const target = e.currentTarget;
  if (!target.dataset.fallbackApplied) {
    target.dataset.fallbackApplied = 'true';
    target.src = SatQueryAPI.getRasterPreviewUrl(fallback);
  }
};

export interface BitemporalCardData {
  analysis_id: string;
  date: string;
  area_of_interest: string;
  task: string;
  t1_url: string;
  t2_url: string;
  mask_binary_url: string;
  overlay_t2_url: string;
  zoom_t1_url: string;
  zoom_t2_url: string;
  model_optical_url: string;
  model_sar_url: string;
  model_changeformer_url: string;
  input_details: {
    image1: string;
    date1: string;
    image2: string;
    date2: string;
    format: string;
    resolution: string;
    crs: string;
    area_of_interest: string;
  };
  legend: Array<{
    label: string;
    color: string;
    outline?: boolean;
  }>;
  quantitative: {
    original_built_up_km2: string;
    new_built_up_km2: string;
    removed_built_up_km2: string;
    net_change_km2: string;
    percentage_increase: string;
    total_changed_km2: string;
    high_confidence_pct: string;
    uncertain_change_km2: string;
  };
  model_wise: Array<{
    name: string;
    detected_change: string;
    color: string;
    url: string;
  }>;
  insights: string[];
  executive_summary?: {
    situation: string;
    infrastructure_impact: string;
    zoning_recommendations: string;
  };
}

interface BitemporalChangeCardProps {
  data: BitemporalCardData;
  onOpenPdf?: () => void;
}

export const BitemporalChangeCard: React.FC<BitemporalChangeCardProps> = ({
  data,
  onOpenPdf,
}) => {
  const [lightboxImg, setLightboxImg] = useState<{ url: string; title: string } | null>(null);

  return (
    <div className="w-full my-4 rounded-xl overflow-hidden border border-[var(--border-subtle)] bg-[var(--bg-surface)] shadow-lg animate-fadeIn text-[var(--text-main)] font-sans">
      {/* ── HEADER BANNER ── */}
      <div className="bg-[#0f172a] text-white px-5 py-4 border-b border-slate-700 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-red-600/20 border border-red-500/40 flex items-center justify-center text-red-400">
            <SatQueryLogo size={20} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-base tracking-tight text-white">SatQuery AI</span>
              <span className="bg-red-500/20 text-red-300 text-[10px] font-mono px-2 py-0.5 rounded border border-red-500/30 uppercase tracking-wider font-semibold">
                Bitemporal Change
              </span>
            </div>
            <p className="text-slate-300 text-xs font-medium">
              Bi-temporal Change Analysis — Detect • Localize • Quantify • Explain
            </p>
          </div>
        </div>

        {onOpenPdf && (
          <button
            type="button"
            onClick={onOpenPdf}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-red-600 hover:bg-red-500 text-white rounded-lg text-xs font-medium transition-colors shadow-sm cursor-pointer"
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Executive Briefing PDF</span>
          </button>
        )}
      </div>

      {/* ── METADATA STATUS STRIP ── */}
      <div className="bg-slate-900/90 text-slate-300 px-5 py-2 text-[11px] font-mono border-b border-slate-800 flex flex-wrap items-center gap-y-1 gap-x-5">
        <div>
          <span className="text-slate-400">Analysis ID:</span>{' '}
          <span className="text-white font-semibold">{data.analysis_id}</span>
        </div>
        <div>
          <span className="text-slate-400">Date:</span>{' '}
          <span className="text-white">{data.date}</span>
        </div>
        <div className="flex items-center gap-1">
          <MapPin className="w-3 h-3 text-red-400" />
          <span className="text-slate-400">AOI:</span>{' '}
          <span className="text-white font-medium">{data.area_of_interest}</span>
        </div>
        <div className="flex items-center gap-1">
          <Layers className="w-3 h-3 text-emerald-400" />
          <span className="text-slate-400">Task:</span>{' '}
          <span className="text-white font-medium">{data.task}</span>
        </div>
      </div>

      {/* ── MAIN CONTENT (2-COLUMN LAYOUT) ── */}
      <div className="p-4 sm:p-5 grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left Column: 2x2 High-Resolution Imagery Grid (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {/* 1. Image 1 (T1) */}
            <div className="group relative rounded-lg overflow-hidden border border-slate-700 bg-slate-950 aspect-square">
              <img
                src={resolvePreview(data.t1_url, 'cartosat_t1.tif')}
                alt="Image 1 (T1) Baseline"
                className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
                onError={(e) => handleImgError(e, 'cartosat_t1.tif')}
              />
              <button
                type="button"
                onClick={() => setLightboxImg({ url: resolvePreview(data.t1_url, 'cartosat_t1.tif'), title: 'Image 1 (T1) Baseline' })}
                className="absolute top-2 right-2 p-1.5 rounded-md bg-black/60 text-white/80 hover:text-white hover:bg-black/90 opacity-0 group-hover:opacity-100 transition-opacity"
                title="Expand view"
              >
                <Maximize2 className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* 2. Image 2 (T2) */}
            <div className="group relative rounded-lg overflow-hidden border border-slate-700 bg-slate-950 aspect-square">
              <img
                src={resolvePreview(data.t2_url, 'cartosat_t2.tif')}
                alt="Image 2 (T2) Current"
                className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
                onError={(e) => handleImgError(e, 'cartosat_t2.tif')}
              />
              <button
                type="button"
                onClick={() => setLightboxImg({ url: resolvePreview(data.t2_url, 'cartosat_t2.tif'), title: 'Image 2 (T2) Current' })}
                className="absolute top-2 right-2 p-1.5 rounded-md bg-black/60 text-white/80 hover:text-white hover:bg-black/90 opacity-0 group-hover:opacity-100 transition-opacity"
                title="Expand view"
              >
                <Maximize2 className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* 3. Change Mask (Binary) */}
            <div className="group relative rounded-lg overflow-hidden border border-slate-700 bg-slate-950 aspect-square">
              <img
                src={resolvePreview(data.mask_binary_url, 'cartosat_t2.tif')}
                alt="Change Mask (Binary Output)"
                className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
                onError={(e) => handleImgError(e, 'cartosat_t2.tif')}
              />
              <button
                type="button"
                onClick={() => setLightboxImg({ url: resolvePreview(data.mask_binary_url, 'cartosat_t2.tif'), title: 'Change Mask (Binary Output)' })}
                className="absolute top-2 right-2 p-1.5 rounded-md bg-black/60 text-white/80 hover:text-white hover:bg-black/90 opacity-0 group-hover:opacity-100 transition-opacity"
                title="Expand view"
              >
                <Maximize2 className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* 4. Change Overlay on T2 */}
            <div className="group relative rounded-lg overflow-hidden border border-slate-700 bg-slate-950 aspect-square">
              <img
                src={resolvePreview(data.overlay_t2_url, 'cartosat_t2.tif')}
                alt="Change Overlay on T2"
                className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
                onError={(e) => handleImgError(e, 'cartosat_t2.tif')}
              />
              <button
                type="button"
                onClick={() => setLightboxImg({ url: resolvePreview(data.overlay_t2_url, 'cartosat_t2.tif'), title: 'Change Overlay on T2' })}
                className="absolute top-2 right-2 p-1.5 rounded-md bg-black/60 text-white/80 hover:text-white hover:bg-black/90 opacity-0 group-hover:opacity-100 transition-opacity"
                title="Expand view"
              >
                <Maximize2 className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Details, Legend, and Quantitative Results (5 Cols) */}
        <div className="lg:col-span-5 space-y-4 flex flex-col justify-between">
          {/* Section: Input Details */}
          <div className="p-3.5 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-app)]/60 text-xs">
            <h4 className="font-bold text-xs uppercase tracking-wider text-[var(--text-muted)] mb-2.5 flex items-center gap-1.5">
              <Calendar className="w-3.5 h-3.5 text-red-500" />
              Input Details
            </h4>
            <div className="grid grid-cols-2 gap-y-1.5 gap-x-2 text-[11px]">
              <div>
                <span className="text-[var(--text-dim)]">Image 1:</span>{' '}
                <span className="font-semibold text-[var(--text-main)]">{data.input_details.image1}</span>
              </div>
              <div>
                <span className="text-[var(--text-dim)]">Date 1:</span>{' '}
                <span className="font-mono text-[var(--text-main)]">{data.input_details.date1}</span>
              </div>
              <div>
                <span className="text-[var(--text-dim)]">Image 2:</span>{' '}
                <span className="font-semibold text-[var(--text-main)]">{data.input_details.image2}</span>
              </div>
              <div>
                <span className="text-[var(--text-dim)]">Date 2:</span>{' '}
                <span className="font-mono text-[var(--text-main)]">{data.input_details.date2}</span>
              </div>
              <div>
                <span className="text-[var(--text-dim)]">Format:</span>{' '}
                <span className="font-mono text-[var(--text-main)]">{data.input_details.format}</span>
              </div>
              <div>
                <span className="text-[var(--text-dim)]">Resolution:</span>{' '}
                <span className="font-mono text-[var(--text-main)]">{data.input_details.resolution}</span>
              </div>
              <div>
                <span className="text-[var(--text-dim)]">CRS:</span>{' '}
                <span className="font-mono text-[var(--text-main)]">{data.input_details.crs}</span>
              </div>
              <div>
                <span className="text-[var(--text-dim)]">AOI Area:</span>{' '}
                <span className="font-mono font-semibold text-[var(--text-main)]">
                  {data.input_details.area_of_interest}
                </span>
              </div>
            </div>
          </div>

          {/* Section: Legend */}
          <div className="p-3.5 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-app)]/60 text-xs">
            <h4 className="font-bold text-xs uppercase tracking-wider text-[var(--text-muted)] mb-2.5">
              Legend
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px]">
              {data.legend.map((item, idx) => (
                <div key={idx} className="flex items-center gap-2">
                  <span
                    className="w-3.5 h-3.5 rounded-sm flex-shrink-0"
                    style={{
                      backgroundColor: item.outline ? 'transparent' : item.color,
                      border: item.outline ? `2px solid ${item.color}` : '1px solid rgba(0,0,0,0.1)',
                    }}
                  />
                  <span className="text-[var(--text-main)] text-[11px] font-medium leading-tight">
                    {item.label}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Section: Quantitative Results */}
          <div className="p-3.5 rounded-lg border border-red-500/20 bg-red-500/5 text-xs space-y-2">
            <div className="flex items-center justify-between pb-1 border-b border-red-500/10">
              <h4 className="font-bold text-xs uppercase tracking-wider text-red-600 dark:text-red-400 flex items-center gap-1.5">
                <TrendingUp className="w-3.5 h-3.5" />
                Quantitative Results
              </h4>
              <span className="font-mono text-[10px] bg-red-500/20 text-red-700 dark:text-red-300 font-bold px-2 py-0.5 rounded">
                Net: {data.quantitative.net_change_km2} ({data.quantitative.percentage_increase})
              </span>
            </div>

            <div className="grid grid-cols-2 gap-y-1.5 gap-x-3 text-[11px]">
              <div className="flex justify-between">
                <span className="text-[var(--text-muted)]">Original Built-up:</span>
                <span className="font-mono font-medium">{data.quantitative.original_built_up_km2}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-red-600 dark:text-red-400 font-medium">New Built-up:</span>
                <span className="font-mono font-bold text-red-600 dark:text-red-400">
                  {data.quantitative.new_built_up_km2}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-blue-600 dark:text-blue-400 font-medium">Removed Built-up:</span>
                <span className="font-mono font-medium">{data.quantitative.removed_built_up_km2}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[var(--text-muted)]">Total Changed:</span>
                <span className="font-mono font-semibold">{data.quantitative.total_changed_km2}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[var(--text-muted)]">High Confidence:</span>
                <span className="font-mono text-emerald-600 dark:text-emerald-400 font-medium">
                  {data.quantitative.high_confidence_pct}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-[var(--text-muted)]">Uncertain Area:</span>
                <span className="font-mono text-amber-600 dark:text-amber-400 font-medium">
                  {data.quantitative.uncertain_change_km2}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── LOWER SECTION: ZOOMED VIEW & MODEL-WISE COMPARISON & INSIGHTS ── */}
      <div className="px-5 pb-5 pt-1 border-t border-[var(--border-subtle)] space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-5 pt-3">
          {/* Zoomed-in View (Selected Area) (4 Cols) */}
          <div className="md:col-span-4 space-y-2">
            <h4 className="font-bold text-xs uppercase tracking-wider text-[var(--text-muted)]">
              Zoomed-in View (Selected Area)
            </h4>
            <div className="grid grid-cols-2 gap-2">
              <div className="group relative rounded-lg overflow-hidden border border-slate-700 bg-slate-950 aspect-square">
                <img
                  src={resolvePreview(data.zoom_t1_url, 'cartosat_t1.tif')}
                  alt="Zoomed View T1"
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                  onError={(e) => handleImgError(e, 'cartosat_t1.tif')}
                />
                <span className="absolute bottom-1 left-1 bg-black/70 text-white font-mono text-[9px] px-1.5 py-0.5 rounded">
                  Zoomed T1
                </span>
              </div>
              <div className="group relative rounded-lg overflow-hidden border border-slate-700 bg-slate-950 aspect-square">
                <img
                  src={resolvePreview(data.zoom_t2_url, 'cartosat_t2.tif')}
                  alt="Zoomed View T2"
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                  onError={(e) => handleImgError(e, 'cartosat_t2.tif')}
                />
                <span className="absolute bottom-1 left-1 bg-black/70 text-white font-mono text-[9px] px-1.5 py-0.5 rounded">
                  Zoomed T2
                </span>
              </div>
            </div>
          </div>

          {/* Model-wise Change Detection (8 Cols) */}
          <div className="md:col-span-8 space-y-2">
            <h4 className="font-bold text-xs uppercase tracking-wider text-[var(--text-muted)]">
              Model-wise Change Detection
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {data.model_wise.map((m, idx) => (
                <div
                  key={idx}
                  className="rounded-lg overflow-hidden border border-slate-700 bg-slate-950 flex flex-col justify-between"
                >
                  <div className="relative aspect-[4/3] bg-black">
                    <img
                      src={resolvePreview(m.url, 'cartosat_t2.tif')}
                      alt={m.name}
                      className="w-full h-full object-cover"
                      onError={(e) => handleImgError(e, 'cartosat_t2.tif')}
                    />
                  </div>
                  <div className="p-2 bg-slate-900 border-t border-slate-800 text-[11px] flex items-center justify-between">
                    <span className="text-slate-200 font-medium">{m.name}</span>
                    <span className="font-mono text-[10px] text-slate-400">{m.detected_change}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Key Insights (Numbered Badges 1-4) */}
        <div className="p-4 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-app)]/60 text-xs space-y-2.5">
          <h4 className="font-bold text-xs uppercase tracking-wider text-[var(--text-muted)]">
            Key Insights
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-[12px]">
            {data.insights.map((insight, idx) => (
              <div key={idx} className="flex items-start gap-2.5">
                <span className="flex-shrink-0 w-5 h-5 rounded-full bg-red-600 text-white font-bold text-[11px] flex items-center justify-center">
                  {idx + 1}
                </span>
                <span className="text-[var(--text-main)] leading-relaxed">{insight}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── SECTION 5: EXECUTIVE STRATEGIC SUMMARY & URBAN PLANNING DIRECTIVES ── */}
        {data.executive_summary && (
          <div className="p-4 rounded-lg border border-slate-700 bg-slate-900/80 space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <h4 className="font-bold text-xs uppercase tracking-wider text-white flex items-center gap-2">
                <span className="w-4 h-4 rounded bg-red-600/30 text-red-400 flex items-center justify-center text-[10px] font-bold">5</span>
                Executive Strategic Summary &amp; Urban Planning Directives
              </h4>
              <span className="text-[10px] font-mono text-emerald-400 font-semibold px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 uppercase">
                Zoning Intelligence
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
              <div className="p-3 rounded bg-slate-950/60 border border-slate-800 space-y-1.5">
                <span className="text-red-400 font-bold uppercase text-[10px] tracking-wider block">
                  Expansion Overview
                </span>
                <p className="text-slate-300 leading-relaxed text-[11px]">
                  {data.executive_summary.situation}
                </p>
              </div>

              <div className="p-3 rounded bg-slate-950/60 border border-slate-800 space-y-1.5">
                <span className="text-blue-400 font-bold uppercase text-[10px] tracking-wider block">
                  Infrastructure Impact
                </span>
                <p className="text-slate-300 leading-relaxed text-[11px]">
                  {data.executive_summary.infrastructure_impact}
                </p>
              </div>

              <div className="p-3 rounded bg-slate-950/60 border border-slate-800 space-y-1.5">
                <span className="text-emerald-400 font-bold uppercase text-[10px] tracking-wider block">
                  Zoning &amp; Field Directives
                </span>
                <p className="text-slate-300 leading-relaxed text-[11px]">
                  {data.executive_summary.zoning_recommendations}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── LIGHTBOX MODAL ── */}
      {lightboxImg && (
        <div
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4"
          onClick={() => setLightboxImg(null)}
        >
          <div
            className="relative max-w-4xl max-h-[90vh] bg-slate-950 rounded-xl overflow-hidden border border-slate-700 shadow-2xl p-2"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-3 py-2 text-white border-b border-slate-800">
              <span className="font-medium text-xs font-mono">{lightboxImg.title}</span>
              <button
                type="button"
                onClick={() => setLightboxImg(null)}
                className="p-1 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <img
              src={resolvePreview(lightboxImg.url, 'cartosat_t1.tif')}
              alt={lightboxImg.title}
              className="max-h-[80vh] w-auto mx-auto object-contain mt-2"
              onError={(e) => handleImgError(e, 'cartosat_t1.tif')}
            />
          </div>
        </div>
      )}
    </div>
  );
};
