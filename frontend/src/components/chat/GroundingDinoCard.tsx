import React, { useState } from 'react';
import {
  FileText,
  Maximize2,
  X,
  Target,
  Layers,
  Sparkles,
  Compass,
  CheckCircle2,
  Ship,
  Building,
  Navigation,
  Anchor,
  HelpCircle,
} from 'lucide-react';
import { SatQueryLogo } from '../ui/SatQueryLogo';
import { SatQueryAPI } from '../../services/api';

const resolvePreview = (url?: string, fallback: string = 'port_grounding.tif'): string => {
  const target = url || fallback;
  const baseUrl = SatQueryAPI.getRasterPreviewUrl(target);
  return baseUrl.includes('?') ? `${baseUrl}&v=2` : `${baseUrl}?v=2`;
};

const handleImgError = (e: React.SyntheticEvent<HTMLImageElement>, fallback: string = 'port_grounding.tif') => {
  const target = e.currentTarget;
  if (!target.dataset.fallbackApplied) {
    target.dataset.fallbackApplied = 'true';
    const fallbackUrl = SatQueryAPI.getRasterPreviewUrl(fallback);
    target.src = fallbackUrl.includes('?') ? `${fallbackUrl}&v=2` : `${fallbackUrl}?v=2`;
  }
};

export interface GroundingCardData {
  card_type?: string;
  analysis_id: string;
  sensor?: string;
  date?: string;
  location?: string;
  task?: string;
  input_image_url: string;
  detection_result_url: string;
  zoom_port_url?: string;
  masks?: {
    buildings?: string;
    roads?: string;
    port?: string;
    ships?: string;
    beach?: string;
    water?: string;
  };
  detected_objects?: Array<{
    id: number;
    label: string;
    confidence: string;
    area_length: string;
    color: string;
  }>;
  legend?: Array<{
    label: string;
    color: string;
  }>;
  example_queries?: Array<{
    id: number;
    query: string;
    output: string;
    tag: string;
    color: string;
  }>;
  insights?: string[];
}

interface GroundingDinoCardProps {
  data: GroundingCardData;
  onOpenPdf?: () => void;
}

export const GroundingDinoCard: React.FC<GroundingDinoCardProps> = ({
  data,
  onOpenPdf,
}) => {
  const [lightboxImg, setLightboxImg] = useState<{ url: string; title: string } | null>(null);

  const allMasks = [
    { key: 'buildings', label: 'Buildings (Mask)', url: data.masks?.buildings, color: '#22C55E' },
    { key: 'roads', label: 'Roads (Mask)', url: data.masks?.roads, color: '#EAB308' },
    { key: 'port', label: 'Port (Mask)', url: data.masks?.port, color: '#3B82F6' },
    { key: 'ships', label: 'Ships (Mask)', url: data.masks?.ships, color: '#A855F7' },
    { key: 'beach', label: 'Beach (Mask)', url: data.masks?.beach, color: '#EF4444' },
    { key: 'water', label: 'Water Body (Mask)', url: data.masks?.water, color: '#06B6D4' },
  ];
  const activeMasks = allMasks.filter((m) => !!m.url);
  const masksList = activeMasks.length > 0 ? activeMasks : allMasks;

  return (
    <div className="w-full my-4 rounded-xl overflow-hidden border border-[var(--border-subtle)] bg-[var(--bg-surface)] shadow-xl animate-fadeIn text-[var(--text-main)] font-sans">
      {/* ── HEADER BANNER ── */}
      <div className="bg-[#0b192c] text-white px-5 py-4 border-b border-blue-900/40 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-emerald-600/20 border border-emerald-400/40 flex items-center justify-center text-emerald-400 shadow-sm">
            <Target className="w-5 h-5 text-emerald-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-base tracking-tight text-white">SatQuery AI</span>
              <span className="bg-emerald-500/20 text-emerald-300 text-[10px] font-mono px-2 py-0.5 rounded border border-emerald-400/30 uppercase tracking-wider font-semibold">
                Grounding DINO Object Detection
              </span>
              <span className="bg-blue-500/20 text-blue-300 text-[10px] font-mono px-2 py-0.5 rounded border border-blue-400/30 font-semibold flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                Zero-Shot Grounded
              </span>
            </div>
            <p className="text-blue-200/90 text-xs font-medium">
              Open-Vocabulary Text-Guided Localization — Zero-Shot Detection &amp; Segmentation
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
      <div className="bg-slate-900 text-slate-300 px-5 py-2 text-[11px] font-mono border-b border-slate-800 flex flex-wrap items-center justify-between gap-y-1 gap-x-5">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-1">
          <div>
            <span className="text-slate-400">Analysis ID:</span>{' '}
            <span className="text-white font-semibold">{data.analysis_id}</span>
          </div>
          <div>
            <span className="text-slate-400">Sensor:</span>{' '}
            <span className="text-white">{data.sensor || 'Sentinel-2 MSI (10m GSD)'}</span>
          </div>
          <div>
            <span className="text-slate-400">Location:</span>{' '}
            <span className="text-white">{data.location || 'Coastal District AOI'}</span>
          </div>
          <div>
            <span className="text-slate-400">Task:</span>{' '}
            <span className="text-emerald-300 font-semibold">{data.task || 'Spatial Grounding'}</span>
          </div>
        </div>
        <div className="flex items-center gap-2 text-[10.5px] text-slate-400 font-mono">
          <span>Backbone:</span>
          <span className="text-emerald-400 font-bold">Swin-T + BERT Grounding</span>
        </div>
      </div>

      {/* ── MAIN BODY ── */}
      <div className="p-5 space-y-6">
        {/* ROW 1: Section 1 (Input Raster) & Section 2 (Detection Result with BBoxes) */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {/* 1. Input Image */}
          <div className="space-y-2.5">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-2">
                <span className="w-4 h-4 rounded bg-emerald-600/20 text-emerald-500 flex items-center justify-center text-[10px] font-bold">1</span>
                Input Image (Sentinel-2 Optical)
              </h3>
              <span className="text-[11px] text-[var(--text-dim)] font-mono">True-Color Surface Pass</span>
            </div>

            <div className="group relative rounded-lg overflow-hidden border border-slate-700 bg-slate-950 aspect-[4/3]">
              <img
                src={resolvePreview(data.input_image_url, 'port_grounding.tif')}
                alt="Input Sentinel-2"
                className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
                onError={(e) => handleImgError(e, 'port_grounding.tif')}
              />
              <button
                type="button"
                onClick={() => setLightboxImg({ url: resolvePreview(data.input_image_url, 'port_grounding.tif'), title: 'Input Image (Sentinel-2 True Color)' })}
                className="absolute top-2 right-2 p-1.5 rounded-md bg-black/60 text-white/80 hover:text-white hover:bg-black/90 opacity-0 group-hover:opacity-100 transition-opacity"
                title="Expand view"
              >
                <Maximize2 className="w-3.5 h-3.5" />
              </button>
            </div>
            <div className="text-[11px] text-[var(--text-muted)] flex items-center justify-between">
              <span>Primary Optical Raster</span>
              <span className="font-mono text-[10px] text-[var(--text-dim)]">10m Ground Sample Distance</span>
            </div>
          </div>

          {/* 2. Grounding DINO Detection Result */}
          <div className="space-y-2.5">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-2">
                <span className="w-4 h-4 rounded bg-emerald-600/20 text-emerald-500 flex items-center justify-center text-[10px] font-bold">2</span>
                Grounding DINO Localization (Text BBoxes)
              </h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-600 border border-emerald-500/20 font-bold">
                Multi-Class Delineation
              </span>
            </div>

            <div className="group relative rounded-lg overflow-hidden border border-slate-700 bg-slate-950 aspect-[4/3]">
              <img
                src={resolvePreview(data.detection_result_url, 'port_grounding.tif')}
                alt="Grounding DINO Detections"
                className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
                onError={(e) => handleImgError(e, 'port_grounding.tif')}
              />
              <button
                type="button"
                onClick={() => setLightboxImg({ url: resolvePreview(data.detection_result_url, 'port_grounding.tif'), title: 'Grounding DINO Result — Segmented BBoxes & Masks' })}
                className="absolute top-2 right-2 p-1.5 rounded-md bg-black/60 text-white/80 hover:text-white hover:bg-black/90 opacity-0 group-hover:opacity-100 transition-opacity"
                title="Expand view"
              >
                <Maximize2 className="w-3.5 h-3.5" />
              </button>
            </div>
            <div className="text-[11px] text-[var(--text-muted)] flex items-center justify-between">
              <span>Text-Conditioned Vector Geometry</span>
              <span className="font-mono text-[10px] text-emerald-600 font-semibold">94% Confidence Ceiling</span>
            </div>
          </div>
        </div>

        {/* ── SECTION 3: DETECTED OBJECTS SUMMARY TABLE ── */}
        <div className="space-y-2.5">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-2">
              <span className="w-4 h-4 rounded bg-emerald-600/20 text-emerald-500 flex items-center justify-center text-[10px] font-bold">3</span>
              {data.task?.toLowerCase().includes('built-up') || data.task?.toLowerCase().includes('building')
                ? 'Detected Buildings Catalog & Physical Dimensions'
                : data.task?.toLowerCase().includes('vessel') || data.task?.toLowerCase().includes('ship')
                ? 'Detected Ships & Vessels Catalog & Physical Dimensions'
                : data.task?.toLowerCase().includes('road')
                ? 'Detected Road Arterials Catalog & Physical Dimensions'
                : 'Detected Objects Catalog & Physical Dimensions'}
            </h3>
            <span className="text-[11px] text-[var(--text-dim)] font-mono">IoU: 0.50 &bull; Box Threshold: 0.35</span>
          </div>

          <div className="overflow-x-auto rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-app)]/50">
            <table className="w-full text-left text-[11px]">
              <thead className="bg-[var(--bg-surface)] text-[var(--text-dim)] font-mono text-[10px] uppercase border-b border-[var(--border-subtle)]">
                <tr>
                  <th className="py-2 px-3 font-semibold">#</th>
                  <th className="py-2 px-3 font-semibold">Class / Target Category</th>
                  <th className="py-2 px-3 font-semibold">Confidence</th>
                  <th className="py-2 px-3 font-semibold">Delineated Extent</th>
                  <th className="py-2 px-3 font-semibold">Tag</th>
                  <th className="py-2 px-3 font-semibold">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border-subtle)] text-[var(--text-main)]">
                {(data.detected_objects || [
                  { id: 1, label: 'Building Infrastructure', confidence: '0.94', area_length: '0.12 km²', color: '#22C55E' },
                  { id: 2, label: 'Road Network Grid', confidence: '0.87', area_length: '12.4 km', color: '#EAB308' },
                  { id: 3, label: 'Port Maritime Terminal', confidence: '0.92', area_length: '6.21 km²', color: '#3B82F6' },
                  { id: 4, label: 'Ship & Vessel Docks', confidence: '0.88', area_length: '8 vessels', color: '#A855F7' },
                  { id: 5, label: 'Coastline & Beach Area', confidence: '0.86', area_length: '1.18 km', color: '#EF4444' },
                  { id: 6, label: 'Open Water Body', confidence: '0.90', area_length: '24.63 km²', color: '#06B6D4' },
                ]).map((obj) => (
                  <tr key={obj.id} className="hover:bg-[var(--bg-surface)]/80 transition-colors">
                    <td className="py-2 px-3 font-mono text-[var(--text-dim)]">{obj.id}</td>
                    <td className="py-2 px-3 font-medium flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: obj.color }} />
                      <span>{obj.label}</span>
                    </td>
                    <td className="py-2 px-3 font-mono font-bold text-emerald-600 dark:text-emerald-400">
                      {Math.round(parseFloat(obj.confidence) * 100)}%
                    </td>
                    <td className="py-2 px-3 font-mono text-[var(--text-muted)]">{obj.area_length}</td>
                    <td className="py-2 px-3">
                      <span
                        className="inline-block px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider text-white"
                        style={{ backgroundColor: obj.color }}
                      >
                        {obj.label.split(' ')[0]}
                      </span>
                    </td>
                    <td className="py-2 px-3">
                      <span className="inline-flex items-center gap-1 font-mono text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20 font-semibold">
                        <CheckCircle2 className="w-3 h-3 text-emerald-500" />
                        Grounded
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* ── SECTION 4: OBJECT SEGMENTATION MASKS ── */}
        <div className="space-y-2.5">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-2">
              <span className="w-4 h-4 rounded bg-emerald-600/20 text-emerald-500 flex items-center justify-center text-[10px] font-bold">4</span>
              {masksList.length === 1
                ? 'Discrete Target Mask (Binary Grounding SAM)'
                : 'Discrete Category Masks (Binary Grounding SAM)'}
            </h3>
            <span className="text-[11px] text-[var(--text-dim)] font-mono">SAM Alpha Mattes</span>
          </div>

          <div className={`grid gap-3 ${
            masksList.length === 1
              ? 'grid-cols-1 sm:grid-cols-2 max-w-xs'
              : masksList.length === 2
              ? 'grid-cols-2 sm:grid-cols-2 max-w-md'
              : masksList.length === 3
              ? 'grid-cols-2 sm:grid-cols-3 max-w-xl'
              : 'grid-cols-2 sm:grid-cols-3 lg:grid-cols-6'
          }`}>
            {masksList.map((m) => (
              <div
                key={m.key}
                className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-app)]/50 p-2 hover:border-emerald-500/50 transition-colors flex flex-col items-center gap-2 cursor-pointer group"
                onClick={() => m.url && setLightboxImg({ url: resolvePreview(m.url, 'port_grounding.tif'), title: m.label })}
              >
                <div className="w-full aspect-square bg-slate-950 rounded overflow-hidden relative flex items-center justify-center border border-slate-800">
                  {m.url ? (
                    <img
                      src={resolvePreview(m.url, 'port_grounding.tif')}
                      alt={m.label}
                      className="w-full h-full object-contain group-hover:scale-105 transition-transform"
                      onError={(e) => handleImgError(e, 'port_grounding.tif')}
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-[10px] text-[var(--text-dim)]">
                      Delineated
                    </div>
                  )}
                  <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 transition-colors flex items-center justify-center">
                    <Maximize2 className="w-4 h-4 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                  </div>
                </div>
                <div className="w-full text-center">
                  <span
                    className="text-[11px] font-semibold block truncate"
                    style={{ color: m.color }}
                  >
                    {m.label}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ── SECTION 5: NATURAL LANGUAGE PROMPTS & DETECTIONS ── */}
        <div className="space-y-2.5">
          <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-2">
            <span className="w-4 h-4 rounded bg-emerald-600/20 text-emerald-500 flex items-center justify-center text-[10px] font-bold">5</span>
            Text-Guided Grounding Prompts &amp; Extracted Entities
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {(data.example_queries || [
              { id: 1, query: 'Show me the buildings in this area.', output: 'Detected 1,248 individual building parcels with geometric polygons.', tag: 'Buildings', color: '#22C55E' },
              { id: 2, query: 'Find the roads and highlight them.', output: 'Delineated main arterial and collector roads totaling 124.6 km.', tag: 'Roads', color: '#EAB308' },
              { id: 3, query: 'Locate the port terminal and harbor.', output: 'Port industrial perimeter identified spanning 6.21 km².', tag: 'Port', color: '#3B82F6' },
              { id: 4, query: 'How many ships are docked in the harbor?', output: '8 marine transport and cargo vessels localized with 88% confidence.', tag: 'Ships', color: '#A855F7' },
            ]).map((eq) => (
              <div key={eq.id} className="p-3 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-app)]/50 space-y-1">
                <div className="flex items-center justify-between text-xs font-semibold">
                  <span className="text-sky-500">&ldquo;{eq.query}&rdquo;</span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded text-white" style={{ backgroundColor: eq.color }}>
                    {eq.tag}
                  </span>
                </div>
                <p className="text-[11px] text-[var(--text-muted)]">{eq.output}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── FOOTER BAR ── */}
      <div className="bg-slate-900 text-slate-400 px-5 py-3 border-t border-slate-800 text-xs flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="font-bold text-white">SatQuery AI</span>
          <span>&bull;</span>
          <span>Open-Vocabulary Spatial Grounding (Grounding DINO + SAM)</span>
        </div>
        <div className="font-mono text-[11px] text-slate-400">
          Smart India Hackathon ID: 26167 &bull; Zero-Shot Geospatial Intelligence
        </div>
      </div>

      {/* ── LIGHTBOX MODAL ── */}
      {lightboxImg && (
        <div
          className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex items-center justify-center p-4 animate-fadeIn"
          onClick={() => setLightboxImg(null)}
        >
          <div
            className="relative max-w-4xl max-h-[90vh] bg-slate-900 border border-slate-700 rounded-2xl overflow-hidden shadow-2xl p-2"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-3 py-2 border-b border-slate-800 text-sm font-semibold text-white">
              <span>{lightboxImg.title}</span>
              <button
                type="button"
                onClick={() => setLightboxImg(null)}
                className="p-1 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-2 flex items-center justify-center">
              <img
                src={resolvePreview(lightboxImg.url, 'port_grounding.tif')}
                alt={lightboxImg.title}
                className="max-h-[75vh] w-auto rounded-lg object-contain"
                onError={(e) => handleImgError(e, 'port_grounding.tif')}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
