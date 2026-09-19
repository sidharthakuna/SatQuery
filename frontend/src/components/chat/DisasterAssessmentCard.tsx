import React, { useState } from 'react';
import {
  FileText,
  Waves,
  MapPin,
  Maximize2,
  X,
  CheckCircle2,
  Download,
} from 'lucide-react';
import { SatQueryLogo } from '../ui/SatQueryLogo';
import { SatQueryAPI } from '../../services/api';

const resolvePreview = (url?: string, fallback: string = 'flood_t1.tif'): string => {
  if (!url) return SatQueryAPI.getRasterPreviewUrl(fallback);
  return SatQueryAPI.getRasterPreviewUrl(url);
};

const handleImgError = (e: React.SyntheticEvent<HTMLImageElement>, fallback: string = 'flood_t1.tif') => {
  const target = e.currentTarget;
  if (!target.dataset.fallbackApplied) {
    target.dataset.fallbackApplied = 'true';
    target.src = SatQueryAPI.getRasterPreviewUrl(fallback);
  }
};

export interface DisasterCardData {
  analysis_id: string;
  date: string;
  area: string;
  task: string;
  status: string;
  badge: string;
  t1_url: string;
  t2_url: string;
  t1_label: string;
  t2_label: string;
  preprocessing_checks: Array<{
    check: string;
    img1: string;
    img2: string;
    status: string;
    valid: boolean;
  }>;
  area_info: {
    aoi_map_url: string;
    area: string;
    center: string;
    bbox: string;
  };
  model_results: {
    optical_url: string;
    optical_desc: string;
    sar_url: string;
    sar_desc: string;
    changeformer_url: string;
    changeformer_desc: string;
    fusion_url: string;
    fusion_desc: string;
    legend: Array<{
      label: string;
      color: string;
      outline?: boolean;
    }>;
  };
  zoomed_view: {
    t1_url: string;
    t2_url: string;
    mask_url: string;
    overlay_url: string;
  };
  quantitative: Array<{
    metric: string;
    value: string;
    bold?: boolean;
  }>;
  insights: string[];
  impacted_areas: Array<{
    id: number;
    location: string;
    area_km2: string;
    impact_type: string;
    danger_level?: string;
    confidence: string;
  }>;
  executive_summary?: {
    situation: string;
    danger_zones: string;
    safe_zones: string;
    action_protocols: string[];
  };
}

interface DisasterAssessmentCardProps {
  data: DisasterCardData;
  onOpenPdf?: () => void;
  onDownloadGeoJson?: () => void;
}

export const DisasterAssessmentCard: React.FC<DisasterAssessmentCardProps> = ({
  data,
  onOpenPdf,
  onDownloadGeoJson,
}) => {
  const [lightboxImg, setLightboxImg] = useState<{ url: string; title: string } | null>(null);

  return (
    <div className="w-full my-4 rounded-xl overflow-hidden border border-[var(--border-subtle)] bg-[var(--bg-surface)] shadow-xl animate-fadeIn text-[var(--text-main)] font-sans">
      {/* ── HEADER BANNER ── */}
      <div className="bg-[#0b192c] text-white px-5 py-4 border-b border-blue-900/40 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-blue-600/20 border border-blue-400/40 flex items-center justify-center text-blue-400 shadow-sm">
            <Waves className="w-5 h-5 text-blue-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-base tracking-tight text-white">SatQuery AI</span>
              <span className="bg-blue-500/20 text-blue-300 text-[10px] font-mono px-2 py-0.5 rounded border border-blue-400/30 uppercase tracking-wider font-semibold">
                {data.badge || 'Natural Disaster Analysis'}
              </span>
              <span className="bg-emerald-500/20 text-emerald-300 text-[10px] font-mono px-2 py-0.5 rounded border border-emerald-400/30 font-semibold flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                {data.status || 'Completed'}
              </span>
            </div>
            <p className="text-blue-200/90 text-xs font-medium">
              Natural Disaster Analysis — Flood Impact Assessment
            </p>
          </div>
        </div>

        {onOpenPdf && (
          <button
            type="button"
            onClick={onOpenPdf}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-xs font-medium transition-colors shadow-sm cursor-pointer"
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Executive Briefing Dossier</span>
          </button>
        )}
      </div>

      {/* ── METADATA STRIP ── */}
      <div className="bg-slate-900 text-slate-300 px-5 py-2 text-[11px] font-mono border-b border-slate-800 flex flex-wrap items-center gap-y-1 gap-x-5">
        <div>
          <span className="text-slate-400">Analysis ID:</span>{' '}
          <span className="text-white font-semibold">{data.analysis_id}</span>
        </div>
        <div>
          <span className="text-slate-400">Date:</span>{' '}
          <span className="text-white">{data.date}</span>
        </div>
        <div className="flex items-center gap-1">
          <MapPin className="w-3 h-3 text-blue-400" />
          <span className="text-slate-400">Area:</span>{' '}
          <span className="text-white font-medium">{data.area}</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-slate-400">Task:</span>{' '}
          <span className="text-white font-medium">{data.task}</span>
        </div>
      </div>

      <div className="p-5 space-y-6">
        {/* ── SECTION 1: INPUT IMAGES (BI-TEMPORAL) ── */}
        <div className="space-y-2.5">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-2">
              <span className="w-4 h-4 rounded bg-blue-600/20 text-blue-500 flex items-center justify-center text-[10px] font-bold">1</span>
              Input Images (Bi-temporal)
            </h3>
            <span className="text-[11px] text-[var(--text-dim)] font-mono">Sentinel-2 MSI (10m Resolution)</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* T1 */}
            <div className="space-y-1.5">
              <div className="group relative rounded-lg overflow-hidden border border-slate-700 bg-slate-950 aspect-[4/3]">
                <img
                  src={resolvePreview(data.t1_url, 'flood_t1.tif')}
                  alt={data.t1_label}
                  className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
                  onError={(e) => handleImgError(e, 'flood_t1.tif')}
                />
                <button
                  type="button"
                  onClick={() => setLightboxImg({ url: resolvePreview(data.t1_url, 'flood_t1.tif'), title: data.t1_label })}
                  className="absolute top-2 right-2 p-1.5 rounded-md bg-black/60 text-white/80 hover:text-white hover:bg-black/90 opacity-0 group-hover:opacity-100 transition-opacity"
                  title="Expand view"
                >
                  <Maximize2 className="w-3.5 h-3.5" />
                </button>
              </div>
              <div className="text-[11px] font-medium text-[var(--text-main)] flex items-center justify-between">
                <span>{data.t1_label}</span>
                <span className="font-mono text-[10px] text-[var(--text-dim)]">Optical RGB</span>
              </div>
            </div>

            {/* T2 */}
            <div className="space-y-1.5">
              <div className="group relative rounded-lg overflow-hidden border border-slate-700 bg-slate-950 aspect-[4/3]">
                <img
                  src={resolvePreview(data.t2_url, 'flood_t2.tif')}
                  alt={data.t2_label}
                  className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
                  onError={(e) => handleImgError(e, 'flood_t2.tif')}
                />
                <button
                  type="button"
                  onClick={() => setLightboxImg({ url: resolvePreview(data.t2_url, 'flood_t2.tif'), title: data.t2_label })}
                  className="absolute top-2 right-2 p-1.5 rounded-md bg-black/60 text-white/80 hover:text-white hover:bg-black/90 opacity-0 group-hover:opacity-100 transition-opacity"
                  title="Expand view"
                >
                  <Maximize2 className="w-3.5 h-3.5" />
                </button>
              </div>
              <div className="text-[11px] font-medium text-[var(--text-main)] flex items-center justify-between">
                <span>{data.t2_label}</span>
                <span className="font-mono text-[10px] text-[var(--text-dim)]">Optical RGB</span>
              </div>
            </div>
          </div>
        </div>

        {/* ── SECTION 2: PREPROCESSING & INPUT CHECKS ── */}
        <div className="space-y-2.5">
          <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-2">
            <span className="w-4 h-4 rounded bg-blue-600/20 text-blue-500 flex items-center justify-center text-[10px] font-bold">2</span>
            Preprocessing & Input Checks
          </h3>

          <div className="overflow-x-auto rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-app)]/50">
            <table className="w-full text-left text-[11px]">
              <thead className="bg-[var(--bg-surface)] text-[var(--text-dim)] font-mono text-[10px] uppercase border-b border-[var(--border-subtle)]">
                <tr>
                  <th className="py-2 px-3 font-semibold">Check / Preprocessing Step</th>
                  <th className="py-2 px-3 font-semibold">Image 1 (T1)</th>
                  <th className="py-2 px-3 font-semibold">Image 2 (T2)</th>
                  <th className="py-2 px-3 font-semibold">Status / Result</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border-subtle)] text-[var(--text-main)]">
                {data.preprocessing_checks.map((row, idx) => (
                  <tr key={idx} className="hover:bg-[var(--bg-surface)]/80 transition-colors">
                    <td className="py-2 px-3 font-medium">{row.check}</td>
                    <td className="py-2 px-3 font-mono text-emerald-600 dark:text-emerald-400">
                      {row.img1}
                    </td>
                    <td className="py-2 px-3 font-mono text-emerald-600 dark:text-emerald-400">
                      {row.img2}
                    </td>
                    <td className="py-2 px-3">
                      <span className="inline-flex items-center gap-1 font-mono text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20 font-semibold">
                        <CheckCircle2 className="w-3 h-3 text-emerald-500" />
                        {row.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* ── SECTION 3: AREA INFORMATION ── */}
        <div className="space-y-2.5">
          <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-2">
            <span className="w-4 h-4 rounded bg-blue-600/20 text-blue-500 flex items-center justify-center text-[10px] font-bold">3</span>
            Area Information
          </h3>

          <div className="p-3.5 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-app)]/50 grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
            <div className="md:col-span-4 flex items-center justify-center">
              <div className="relative rounded-lg overflow-hidden border border-slate-700 bg-slate-950 w-full max-w-[200px] aspect-square">
                <img
                  src={resolvePreview(data.area_info?.aoi_map_url, 'flood_t1.tif')}
                  alt="AOI Map"
                  className="w-full h-full object-cover"
                  onError={(e) => handleImgError(e, 'flood_t1.tif')}
                />
                <span className="absolute bottom-1 right-1 bg-black/70 text-white font-mono text-[9px] px-1.5 py-0.5 rounded">
                  AOI Bounds
                </span>
              </div>
            </div>

            <div className="md:col-span-8 space-y-2 text-xs">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div className="p-2.5 rounded-md bg-[var(--bg-surface)] border border-[var(--border-subtle)]">
                  <span className="text-[var(--text-dim)] block text-[10px] uppercase font-mono">
                    Area of Interest
                  </span>
                  <span className="font-mono font-bold text-sm text-[var(--text-main)]">
                    {data.area_info.area}
                  </span>
                </div>
                <div className="p-2.5 rounded-md bg-[var(--bg-surface)] border border-[var(--border-subtle)]">
                  <span className="text-[var(--text-dim)] block text-[10px] uppercase font-mono">
                    Center Coordinates
                  </span>
                  <span className="font-mono font-bold text-xs text-[var(--text-main)]">
                    {data.area_info.center}
                  </span>
                </div>
              </div>
              <div className="p-2.5 rounded-md bg-[var(--bg-surface)] border border-[var(--border-subtle)]">
                <span className="text-[var(--text-dim)] block text-[10px] uppercase font-mono">
                  Bounding Box (Envelope)
                </span>
                <span className="font-mono text-xs text-[var(--text-main)]">
                  {data.area_info.bbox}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* ── SECTION 4: MODEL-WISE RESULTS ── */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-2">
              <span className="w-4 h-4 rounded bg-blue-600/20 text-blue-500 flex items-center justify-center text-[10px] font-bold">4</span>
              Model-Wise Results
            </h3>
            <span className="text-[11px] text-blue-500 font-medium font-mono">Multi-Sensor Validation</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {/* 4.1 Optical Model */}
            <div className="rounded-lg overflow-hidden border border-slate-700 bg-slate-950 flex flex-col justify-between">
              <div className="relative aspect-square">
                <img
                  src={resolvePreview(data.model_results?.optical_url, 'flood_t2.tif')}
                  alt="4.1 Optical Model"
                  className="w-full h-full object-cover"
                  onError={(e) => handleImgError(e, 'flood_t2.tif')}
                />
                <button
                  type="button"
                  onClick={() => setLightboxImg({ url: resolvePreview(data.model_results?.optical_url, 'flood_t2.tif'), title: '4.1 Optical Model (Water Segmentation)' })}
                  className="absolute top-1.5 right-1.5 p-1 rounded bg-black/60 text-white/80 hover:text-white"
                >
                  <Maximize2 className="w-3 h-3" />
                </button>
              </div>
              <div className="p-2.5 bg-slate-900 border-t border-slate-800 space-y-1">
                <h5 className="font-semibold text-xs text-white">4.1 Optical Model</h5>
                <p className="text-[10px] text-slate-300 leading-tight">
                  {data.model_results.optical_desc}
                </p>
              </div>
            </div>

            {/* 4.2 SAR Model */}
            <div className="rounded-lg overflow-hidden border border-slate-700 bg-slate-950 flex flex-col justify-between">
              <div className="relative aspect-square">
                <img
                  src={resolvePreview(data.model_results?.sar_url, 'risat_sar.tif')}
                  alt="4.2 SAR Model"
                  className="w-full h-full object-cover"
                  onError={(e) => handleImgError(e, 'risat_sar.tif')}
                />
                <button
                  type="button"
                  onClick={() => setLightboxImg({ url: resolvePreview(data.model_results?.sar_url, 'risat_sar.tif'), title: '4.2 SAR Model (Backscatter Analysis)' })}
                  className="absolute top-1.5 right-1.5 p-1 rounded bg-black/60 text-white/80 hover:text-white"
                >
                  <Maximize2 className="w-3 h-3" />
                </button>
              </div>
              <div className="p-2.5 bg-slate-900 border-t border-slate-800 space-y-1">
                <h5 className="font-semibold text-xs text-white">4.2 SAR Model</h5>
                <p className="text-[10px] text-slate-300 leading-tight">
                  {data.model_results.sar_desc}
                </p>
              </div>
            </div>

            {/* 4.3 Change Detection */}
            <div className="rounded-lg overflow-hidden border border-slate-700 bg-slate-950 flex flex-col justify-between">
              <div className="relative aspect-square">
                <img
                  src={resolvePreview(data.model_results?.changeformer_url, 'flood_t2.tif')}
                  alt="4.3 Change Detection"
                  className="w-full h-full object-cover"
                  onError={(e) => handleImgError(e, 'flood_t2.tif')}
                />
                <button
                  type="button"
                  onClick={() => setLightboxImg({ url: resolvePreview(data.model_results?.changeformer_url, 'flood_t2.tif'), title: '4.3 Change Detection (ChangeFormer)' })}
                  className="absolute top-1.5 right-1.5 p-1 rounded bg-black/60 text-white/80 hover:text-white"
                >
                  <Maximize2 className="w-3 h-3" />
                </button>
              </div>
              <div className="p-2.5 bg-slate-900 border-t border-slate-800 space-y-1">
                <h5 className="font-semibold text-xs text-white">4.3 Change Detection</h5>
                <p className="text-[10px] text-slate-300 leading-tight">
                  {data.model_results.changeformer_desc}
                </p>
              </div>
            </div>

            {/* 4.4 Multi-Model Fusion */}
            <div className="rounded-lg overflow-hidden border border-blue-600 bg-slate-950 flex flex-col justify-between shadow-md">
              <div className="relative aspect-square">
                <img
                  src={resolvePreview(data.model_results?.fusion_url, 'flood_t2.tif')}
                  alt="4.4 Multi-Model Fusion"
                  className="w-full h-full object-cover"
                  onError={(e) => handleImgError(e, 'flood_t2.tif')}
                />
                <button
                  type="button"
                  onClick={() => setLightboxImg({ url: resolvePreview(data.model_results?.fusion_url, 'flood_t2.tif'), title: '4.4 Multi-Model Fusion (Final Result)' })}
                  className="absolute top-1.5 right-1.5 p-1 rounded bg-black/60 text-white/80 hover:text-white"
                >
                  <Maximize2 className="w-3 h-3" />
                </button>
              </div>
              <div className="p-2.5 bg-blue-950/80 border-t border-blue-800/80 space-y-1">
                <div className="flex items-center justify-between">
                  <h5 className="font-semibold text-xs text-blue-200">4.4 Fusion Result</h5>
                  <span className="font-mono text-[9px] bg-blue-500/20 text-blue-300 px-1.5 py-0.2 rounded font-bold">
                    Primary
                  </span>
                </div>
                <p className="text-[10px] text-blue-100/80 leading-tight">
                  {data.model_results.fusion_desc}
                </p>
              </div>
            </div>
          </div>

          {/* Disaster Map Legend */}
          <div className="p-3 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-app)]/50">
            <h5 className="font-bold text-[10.5px] uppercase tracking-wider text-[var(--text-muted)] mb-2">
              Classification & Cartographic Legend
            </h5>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
              {data.model_results.legend.map((item, idx) => (
                <div key={idx} className="flex items-center gap-2">
                  <span
                    className="w-3 h-3 rounded-sm flex-shrink-0"
                    style={{
                      backgroundColor: item.outline ? 'transparent' : item.color,
                      border: item.outline ? `2px solid ${item.color}` : '1px solid rgba(0,0,0,0.1)',
                    }}
                  />
                  <span className="text-[var(--text-main)] text-[11px] font-medium leading-tight truncate">
                    {item.label}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ── SECTION 5: ZOOMED-IN VIEW ── */}
        <div className="space-y-2.5">
          <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-2">
            <span className="w-4 h-4 rounded bg-blue-600/20 text-blue-500 flex items-center justify-center text-[10px] font-bold">5</span>
            Zoomed-in View (Critical Inundation Hotspot)
          </h3>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
            <div className="group relative rounded-lg overflow-hidden border border-slate-700 bg-slate-950 aspect-square">
              <img
                src={resolvePreview(data.zoomed_view?.t1_url, 'flood_t1.tif')}
                alt="Before Flood (T1 Zoomed)"
                className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                onError={(e) => handleImgError(e, 'flood_t1.tif')}
              />
              <span className="absolute bottom-1 left-1 bg-black/75 text-white font-mono text-[9px] px-1.5 py-0.5 rounded">
                T1 Before Flood
              </span>
            </div>

            <div className="group relative rounded-lg overflow-hidden border border-slate-700 bg-slate-950 aspect-square">
              <img
                src={resolvePreview(data.zoomed_view?.t2_url, 'flood_t2.tif')}
                alt="After Flood (T2 Zoomed)"
                className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                onError={(e) => handleImgError(e, 'flood_t2.tif')}
              />
              <span className="absolute bottom-1 left-1 bg-black/75 text-white font-mono text-[9px] px-1.5 py-0.5 rounded">
                T2 After Flood
              </span>
            </div>

            <div className="group relative rounded-lg overflow-hidden border border-slate-700 bg-slate-950 aspect-square">
              <img
                src={resolvePreview(data.zoomed_view?.mask_url, 'flood_t2.tif')}
                alt="Change Mask Zoomed"
                className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                onError={(e) => handleImgError(e, 'flood_t2.tif')}
              />
              <span className="absolute bottom-1 left-1 bg-black/75 text-white font-mono text-[9px] px-1.5 py-0.5 rounded">
                Change Mask
              </span>
            </div>

            <div className="group relative rounded-lg overflow-hidden border border-slate-700 bg-slate-950 aspect-square">
              <img
                src={resolvePreview(data.zoomed_view?.overlay_url, 'flood_t2.tif')}
                alt="Overlay on T2 Zoomed"
                className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                onError={(e) => handleImgError(e, 'flood_t2.tif')}
              />
              <span className="absolute bottom-1 left-1 bg-blue-900/90 text-white font-mono text-[9px] px-1.5 py-0.5 rounded">
                Overlay on T2
              </span>
            </div>
          </div>
        </div>

        {/* ── SECTION 6: QUANTITATIVE ANALYSIS ── */}
        <div className="space-y-2.5">
          <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-2">
            <span className="w-4 h-4 rounded bg-blue-600/20 text-blue-500 flex items-center justify-center text-[10px] font-bold">6</span>
            Quantitative Analysis
          </h3>

          <div className="p-3.5 rounded-lg border border-blue-500/30 bg-blue-500/5 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            {data.quantitative.map((item, idx) => (
              <div
                key={idx}
                className={`p-2.5 rounded-md border ${
                  item.bold
                    ? 'border-blue-500/50 bg-blue-500/15'
                    : 'border-[var(--border-subtle)] bg-[var(--bg-surface)]'
                }`}
              >
                <span className="text-[var(--text-dim)] block text-[10px] uppercase font-mono truncate">
                  {item.metric}
                </span>
                <span
                  className={`font-mono block mt-1 ${
                    item.bold
                      ? 'text-blue-600 dark:text-blue-400 font-bold text-sm'
                      : 'text-[var(--text-main)] font-semibold text-xs'
                  }`}
                >
                  {item.value}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* ── SECTION 7: KEY INSIGHTS ── */}
        <div className="p-4 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-app)]/50 space-y-2.5">
          <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-2">
            <span className="w-4 h-4 rounded bg-blue-600/20 text-blue-500 flex items-center justify-center text-[10px] font-bold">7</span>
            Key Insights
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-[12px]">
            {data.insights.map((insight, idx) => (
              <div key={idx} className="flex items-start gap-2.5">
                <span className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-600 text-white font-bold text-[11px] flex items-center justify-center">
                  {idx + 1}
                </span>
                <span className="text-[var(--text-main)] leading-relaxed">{insight}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── SECTION 8: IMPACTED AREAS TABLE ── */}
        <div className="space-y-2.5">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-2">
              <span className="w-4 h-4 rounded bg-blue-600/20 text-blue-500 flex items-center justify-center text-[10px] font-bold">8</span>
              Impacted Areas (Coordinates, Danger Zones & Safe Corridors)
            </h3>
            <span className="text-[10px] font-mono text-slate-400">
              <span className="inline-block w-2 h-2 rounded-full bg-red-500 mr-1"></span>Red: Danger Zone &bull; <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 mr-1 ml-2"></span>Green: Safe Zone
            </span>
          </div>

          <div className="overflow-x-auto rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-app)]/50">
            <table className="w-full text-left text-[11px]">
              <thead className="bg-[var(--bg-surface)] text-[var(--text-dim)] font-mono text-[10px] uppercase border-b border-[var(--border-subtle)]">
                <tr>
                  <th className="py-2 px-3 font-semibold">#</th>
                  <th className="py-2 px-3 font-semibold">Location (Lat, Lon)</th>
                  <th className="py-2 px-3 font-semibold">Area (km²)</th>
                  <th className="py-2 px-3 font-semibold">Impact Type</th>
                  <th className="py-2 px-3 font-semibold">Hazard Classification</th>
                  <th className="py-2 px-3 font-semibold">Confidence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border-subtle)] text-[var(--text-main)]">
                {data.impacted_areas.map((loc) => {
                  const isRed = loc.danger_level?.includes('DANGER') || loc.danger_level?.includes('RED') || loc.impact_type.toLowerCase().includes('inundat') || loc.impact_type.toLowerCase().includes('submerg');
                  const isGreen = loc.danger_level?.includes('SAFE') || loc.danger_level?.includes('GREEN') || loc.impact_type.toLowerCase().includes('safe') || loc.impact_type.toLowerCase().includes('airfield');
                  const isAmber = !isRed && !isGreen;

                  return (
                    <tr key={loc.id} className="hover:bg-[var(--bg-surface)]/80 transition-colors">
                      <td className="py-2 px-3 font-mono font-bold text-blue-600 dark:text-blue-400">
                        {loc.id}
                      </td>
                      <td className="py-2 px-3 font-mono">{loc.location}</td>
                      <td className="py-2 px-3 font-mono font-semibold">{loc.area_km2} km²</td>
                      <td className="py-2 px-3">
                        <span className="font-medium">{loc.impact_type}</span>
                      </td>
                      <td className="py-2 px-3">
                        {loc.danger_level ? (
                          <span
                            className={`px-2 py-0.5 rounded text-[9.5px] font-bold tracking-wider font-mono border inline-block ${
                              isRed
                                ? 'bg-red-500/15 text-red-500 dark:text-red-400 border-red-500/40 shadow-[0_0_8px_rgba(239,68,68,0.2)]'
                                : isAmber
                                ? 'bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/40'
                                : 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/40 shadow-[0_0_8px_rgba(16,185,129,0.2)]'
                            }`}
                          >
                            {loc.danger_level}
                          </span>
                        ) : (
                          <span className={`px-2 py-0.5 rounded text-[10px] font-medium border ${isRed ? 'bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20' : isGreen ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20' : 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20'}`}>
                            {isRed ? 'DANGER ZONE (RED)' : isGreen ? 'SAFE ZONE (GREEN)' : 'MODERATE RISK'}
                          </span>
                        )}
                      </td>
                      <td className="py-2 px-3 font-mono text-emerald-600 dark:text-emerald-400 font-bold">
                        {loc.confidence}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* ── SECTION 9: EXECUTIVE STRATEGIC ASSESSMENT & FIELD RESCUE DIRECTIVES ── */}
        {data.executive_summary && (
          <div className="p-4 rounded-lg border border-red-900/30 bg-slate-900/80 space-y-3 shadow-inner">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <h3 className="text-xs font-bold uppercase tracking-wider text-white flex items-center gap-2">
                <span className="w-4 h-4 rounded bg-red-600/30 text-red-400 flex items-center justify-center text-[10px] font-bold">9</span>
                Executive Strategic Assessment &amp; Field Rescue Directives
              </h3>
              <span className="text-[10px] font-mono text-red-400 font-semibold px-2 py-0.5 rounded bg-red-500/10 border border-red-500/20 uppercase">
                Active Tactical Guidance
              </span>
            </div>

            {data.executive_summary.situation && (
              <p className="text-slate-300 text-xs leading-relaxed">
                <strong className="text-white">Situation Assessment: </strong>
                {data.executive_summary.situation}
              </p>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
              <div className="p-3 rounded bg-red-950/30 border border-red-500/30 space-y-1.5">
                <span className="text-red-400 font-bold uppercase text-[10px] tracking-wider block flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                  Critical Danger Zones (Red Alert)
                </span>
                <p className="text-slate-300 leading-relaxed text-[11px]">
                  {data.executive_summary.danger_zones}
                </p>
              </div>

              <div className="p-3 rounded bg-emerald-950/30 border border-emerald-500/30 space-y-1.5">
                <span className="text-emerald-400 font-bold uppercase text-[10px] tracking-wider block flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-emerald-500" />
                  Designated Safe Zones &amp; Evacuation Hubs (Green Safe Corridors)
                </span>
                <p className="text-slate-300 leading-relaxed text-[11px]">
                  {data.executive_summary.safe_zones}
                </p>
              </div>
            </div>

            {data.executive_summary.action_protocols && data.executive_summary.action_protocols.length > 0 && (
              <div className="pt-2 border-t border-slate-800/80 space-y-2">
                <span className="text-[10px] font-mono uppercase text-slate-400 font-semibold block">
                  Priority Rescue Protocols:
                </span>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-[11px]">
                  {data.executive_summary.action_protocols.map((protocol, idx) => (
                    <div key={idx} className="flex items-start gap-2 text-slate-300">
                      <span className="flex-shrink-0 w-4 h-4 rounded bg-blue-600/30 text-blue-400 border border-blue-500/30 font-bold text-[10px] flex items-center justify-center font-mono">
                        {idx + 1}
                      </span>
                      <span className="leading-snug">{protocol}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── DOWNLOAD OUTPUTS BAR ── */}
        <div className="pt-3 border-t border-[var(--border-subtle)] flex flex-wrap items-center justify-between gap-3">
          <span className="text-xs font-semibold text-[var(--text-muted)] flex items-center gap-1.5">
            <Download className="w-3.5 h-3.5 text-blue-500" />
            Download Calibrated Outputs:
          </span>
          <div className="flex flex-wrap items-center gap-2">
            <a
              href={data.model_results.fusion_url}
              download="flood_assessment_fusion.tif"
              className="px-2.5 py-1 rounded bg-[var(--bg-surface)] hover:bg-[var(--bg-app)] border border-[var(--border-subtle)] text-xs font-mono text-[var(--text-main)] transition-colors"
            >
              GeoTIFF (.tif)
            </a>
            {onDownloadGeoJson && (
              <button
                type="button"
                onClick={onDownloadGeoJson}
                className="px-2.5 py-1 rounded bg-[var(--bg-surface)] hover:bg-[var(--bg-app)] border border-[var(--border-subtle)] text-xs font-mono text-emerald-600 dark:text-emerald-400 transition-colors"
              >
                GeoJSON (.geojson)
              </button>
            )}
            {onOpenPdf && (
              <button
                type="button"
                onClick={onOpenPdf}
                className="px-3 py-1 rounded bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium transition-colors shadow-sm flex items-center gap-1 cursor-pointer"
              >
                <FileText className="w-3.5 h-3.5" />
                <span>Executive Dossier (PDF)</span>
              </button>
            )}
            <a
              href={data.model_results.fusion_url}
              download="flood_assessment_package.zip"
              className="px-2.5 py-1 rounded bg-[var(--bg-surface)] hover:bg-[var(--bg-app)] border border-[var(--border-subtle)] text-xs font-mono text-[var(--text-dim)] transition-colors"
            >
              ZIP Package
            </a>
          </div>
        </div>
      </div>

      {/* ── LIGHTBOX MODAL ── */}
      {lightboxImg && (
        <div
          className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex items-center justify-center p-4"
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
              src={resolvePreview(lightboxImg.url, 'flood_t1.tif')}
              alt={lightboxImg.title}
              className="max-h-[80vh] w-auto mx-auto object-contain mt-2"
              onError={(e) => handleImgError(e, 'flood_t1.tif')}
            />
          </div>
        </div>
      )}
    </div>
  );
};
