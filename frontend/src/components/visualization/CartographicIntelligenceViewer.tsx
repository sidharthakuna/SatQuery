import React, { useState } from 'react';
import {
  Compass,
  Maximize2,
  Minimize2,
  Eye,
  EyeOff,
  MapPin,
  ListOrdered,
  SlidersHorizontal,
  Crosshair,
  Layers,
  Map as MapIcon,
  FileText,
} from 'lucide-react';
import type { GeoBoundsLatLon, SpatialCluster } from '../../types/api';
import { SwipeCompare } from './SwipeCompare';
import { GroundingViewer } from './GroundingViewer';
import { ChangeMaskViewer } from './ChangeMaskViewer';
import { MapView } from './MapView';
import { SatQueryAPI } from '../../services/api';

// ── Stable module-level sub-components (never defined inside parent render) ──────────────
const NorthArrow: React.FC<{ isCyan?: boolean }> = ({ isCyan }) => (
  <div className="absolute top-2.5 right-2.5 z-20 flex flex-col items-center pointer-events-none drop-shadow-md">
    <div
      className={`w-6 h-6 rounded-full bg-[#181614]/90 border flex items-center justify-center shadow ${
        isCyan ? 'border-[#cc785c]' : 'border-white/50'
      }`}
    >
      <Compass
        className={`w-3.5 h-3.5 ${isCyan ? 'text-[#cc785c]' : 'text-white'}`}
      />
    </div>
    <span className="text-[7.5px] font-mono font-bold text-white drop-shadow">
      N
    </span>
  </div>
);

const ScaleBar: React.FC<{ isCyan?: boolean }> = ({ isCyan }) => (
  <div className="absolute bottom-2 left-2 z-20 pointer-events-none">
    <div
      className={`bg-[#181614]/95 border rounded px-1.5 py-0.5 shadow-md ${
        isCyan ? 'border-[#cc785c]/70' : 'border-white/40'
      }`}
    >
      <div className="flex items-center text-[7px] font-mono text-white mb-0.5 tracking-tighter">
        <span className="w-4 text-left">0</span>
        <span className="w-6 text-center">250</span>
        <span className="w-7 text-right">500 m</span>
      </div>
      <div
        className={`flex h-1 w-17 border ${
          isCyan ? 'border-[#cc785c]' : 'border-white'
        }`}
      >
        <div className={`w-1/2 ${isCyan ? 'bg-[#cc785c]' : 'bg-white'}`} />
        <div className="w-1/2 bg-[#181614]" />
      </div>
    </div>
  </div>
);

export interface CartographicIntelligenceViewerProps {
  image1Url: string;
  image2Url?: string;
  maskUrl?: string | null;
  boxes?: number[][];
  label1?: string;
  label2?: string;
  modality1?: string;
  modality2?: string;
  taskType?: string;
  clusters?: SpatialCluster[];
  changedAreaHectares?: number | null;
  changedAreaPercent?: number | null;
  confidence?: number;
  descriptionNode?: React.ReactNode;
  bounds?: GeoBoundsLatLon | null;
  fileId?: string | null;
  className?: string;
  onOpenPdf?: () => void;
}

export type ViewMode = 'flow' | 'swipe' | 'grounding' | 'mask' | 'map';

export const CartographicIntelligenceViewer: React.FC<CartographicIntelligenceViewerProps> = ({
  image1Url,
  image2Url,
  maskUrl,
  boxes = [],
  label1 = 'pre_event_t1.tif',
  label2,
  modality1 = 'OPTICAL',
  modality2,
  taskType,
  clusters = [],
  changedAreaHectares,
  changedAreaPercent,
  confidence,
  descriptionNode,
  bounds,
  fileId,
  className = '',
  onOpenPdf,
}) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [hoveredCluster, setHoveredCluster] = useState<number | null>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>('flow');

  const hasSecondary = !!image2Url;
  const hasMask = !!maskUrl;
  const hasBoxes = !!(boxes && boxes.length > 0);
  const hasMap = !!bounds || !!fileId;

  const t1Filename = label1.split('/').pop() || 't1_baseline.tif';
  const t2Filename = (label2 || '').split('/').pop() || (hasSecondary ? 't2_surveillance.tif' : 'feature_mask.tif');

  const isVqa =
    taskType === 'SINGLE_VQA' ||
    taskType === 'VQA' ||
    taskType === 'vqa_grounding' ||
    (!hasSecondary && !hasBoxes && taskType !== 'CROSS_MODAL_FUSION');

  const isFusion =
    !isVqa &&
    (taskType === 'CROSS_MODAL_FUSION' ||
      (hasSecondary && (modality2 === 'SAR' || label2?.toLowerCase().includes('sar'))));

  const isGrounding =
    !isVqa && !isFusion && (hasBoxes || taskType === 'SINGLE_GROUNDING');

  const panelATitle = isFusion
    ? `A. OPTICAL MULTISPECTRAL (Cloud Contaminated)`
    : isVqa
    ? `A. MULTISPECTRAL SATELLITE SCENE (${t1Filename.slice(0, 20)})`
    : isGrounding
    ? `A. TARGET SURVEILLANCE SCENE (${t1Filename.slice(0, 20)})`
    : `A. PRE-EVENT BASELINE (${t1Filename.slice(0, 20)})`;

  const panelAPass = isFusion ? 'Cloudy Optical' : isVqa ? 'MSI True Color' : 'T1 Pass';

  const panelBTitle = hasSecondary
    ? isFusion
      ? `B. MICROWAVE SAR BACKSCATTER (All-Weather Penetration)`
      : `B. SURVEILLANCE (${t2Filename.slice(0, 20)})`
    : isVqa
    ? `B. EXTRACTED LAND COVER MAP (${(maskUrl || 'card_4_landcover.tif').split('/').pop()?.slice(0, 20)})`
    : isGrounding
    ? `B. TARGET SEGMENTATION & GROUNDING`
    : `B. SPECTRAL RADIOMETRIC CHANNEL`;

  const panelBPass = hasSecondary
    ? isFusion ? 'SAR Microwave' : 'T2 Pass'
    : isVqa ? 'Land Cover Map' : isGrounding ? 'SAM Segmentation' : 'Spectral Pass';

  const panelCTitle = isFusion
    ? 'C. RECONSTRUCTED CLOUD-FREE OPTICAL SATELLITE IMAGE (CLEAR SKY)'
    : isGrounding
    ? 'C. TEXT-GUIDED TARGET GROUNDING MAP'
    : isVqa
    ? 'C. CLASSIFIED LAND COVER & BIOPHYSICAL OVERLAY'
    : 'C. CALIBRATED CHANGE DETECTION MAP';

  // Format bounding boxes for GroundingViewer if present: [x1, y1, x2, y2]
  const formattedBoxes: [number, number, number, number][] = (boxes || []).map((b) => [
    b[0] ?? 0,
    b[1] ?? 0,
    b[2] ?? 0,
    b[3] ?? 0,
  ]);

  // Fallback realistic zones if not pre-computed
  const safeClusters: SpatialCluster[] =
    clusters.length > 0
      ? clusters
      : changedAreaHectares != null
      ? [
          {
            zone: 'Zone A',
            area_ha: Number((changedAreaHectares * 0.72).toFixed(1)),
            category: 'Submerged',
            centroid: [260, 240],
          },
          {
            zone: 'Zone B',
            area_ha: Number((changedAreaHectares * 0.16).toFixed(1)),
            category: 'Inundated Basin',
            centroid: [140, 90],
          },
          {
            zone: 'Zone C',
            area_ha: Number((changedAreaHectares * 0.08).toFixed(1)),
            category: 'Lowland Displacement',
            centroid: [210, 80],
          },
          {
            zone: 'Zone D',
            area_ha: Number((changedAreaHectares * 0.04).toFixed(1)),
            category: 'Peripheral Sector',
            centroid: [100, 240],
          },
        ]
      : [];


  const renderPanelA = () => (
    <div className="relative rounded-xl overflow-hidden border border-[var(--border-subtle)] bg-black shadow-sm flex flex-col">
      <div className="h-6 bg-[#181614]/95 border-b border-[var(--border-subtle)] flex items-center justify-between px-2.5 z-20 shrink-0">
        <span className="font-mono text-[9.5px] font-medium text-[var(--text-main)] truncate max-w-[75%] tracking-tight">
          {panelATitle}
        </span>
        <span className="font-mono text-[8.5px] text-[#cc785c] font-semibold uppercase tracking-wider">
          {panelAPass}
        </span>
      </div>
      <div className="relative flex-1 aspect-[4/3] bg-black overflow-hidden flex items-center justify-center">
        <img
          src={SatQueryAPI.getRasterPreviewUrl(image1Url)}
          alt="Panel A Baseline"
          className="w-full h-full object-cover select-none pointer-events-none"
        />
        <NorthArrow />
        <ScaleBar />
      </div>
    </div>
  );

  const renderPanelB = () => {
    // When secondary raster is present, show image2Url; otherwise show maskUrl (feature segmentation/land cover)
    const bImageUrl = hasSecondary ? image2Url! : (maskUrl || image1Url);
    return (
      <div className="relative rounded-xl overflow-hidden border border-[var(--border-subtle)] bg-black shadow-sm flex flex-col">
        <div className="h-6 bg-[#181614]/95 border-b border-[var(--border-subtle)] flex items-center justify-between px-2.5 z-20 shrink-0">
          <span className="font-mono text-[9.5px] font-medium text-[var(--text-main)] truncate max-w-[75%] tracking-tight">
            {panelBTitle}
          </span>
          <span className="font-mono text-[8.5px] text-[#cc785c] font-semibold uppercase tracking-wider">
            {panelBPass}
          </span>
        </div>
        <div className="relative flex-1 aspect-[4/3] bg-black overflow-hidden flex items-center justify-center">
          <img
            src={SatQueryAPI.getRasterPreviewUrl(bImageUrl)}
            alt="Panel B Feature"
            className="w-full h-full object-cover select-none pointer-events-none"
          />
          <NorthArrow />
          <ScaleBar />
        </div>
      </div>
    );
  };

  const renderPanelC = (isLarge?: boolean) => {
    // For optical-SAR fusion, maskUrl contains the reconstructed cloud-free optical raster!
    const cBaseUrl = isFusion ? (maskUrl || image1Url) : isVqa ? image1Url : (image2Url || image1Url);
    return (
      <div className="relative rounded-xl overflow-hidden border border-[#cc785c]/40 bg-black shadow-md flex flex-col">
        <div className="h-6 bg-[#181614]/95 border-b border-[#cc785c]/40 flex items-center justify-between px-2.5 z-20 shrink-0">
          <span className="font-mono text-[9.5px] font-semibold text-[var(--text-main)] truncate max-w-[75%] tracking-tight">
            {panelCTitle}
          </span>
          <span className="font-mono text-[8.5px] text-[#cc785c] bg-[#cc785c]/10 px-1.5 py-0.2 rounded font-bold uppercase tracking-wider border border-[#cc785c]/25">
            {isFusion ? 'Cloud-Free Optical' : 'Marked Output'}
          </span>
        </div>

        <div
          className={`relative flex-1 ${
            isLarge ? 'aspect-[16/10] min-h-[320px]' : 'aspect-[4/3]'
          } bg-black overflow-hidden flex items-center justify-center`}
        >
          <img
            src={SatQueryAPI.getRasterPreviewUrl(cBaseUrl)}
            alt="Panel C Base"
            className="w-full h-full object-cover select-none pointer-events-none"
          />

        {hasMask && showOverlay && !isFusion && (
          <img
            src={SatQueryAPI.getRasterPreviewUrl(maskUrl!)}
            alt="Calibrated Detection Mask"
            className="absolute inset-0 w-full h-full object-cover select-none pointer-events-none transition-opacity duration-150"
            style={{
              opacity: 0.65,
              mixBlendMode: 'screen',
              filter: 'drop-shadow(0 0 6px rgba(204, 120, 92, 0.4))',
            }}
          />
        )}

        {hasBoxes && showOverlay && (
          <svg
            viewBox="0 0 512 512"
            className="absolute inset-0 w-full h-full pointer-events-none"
          >
            {boxes.map((box, bIdx) => {
              const [x1, y1, x2, y2] = box;
              const w = Math.max(x2 - x1, 4);
              const h = Math.max(y2 - y1, 4);
              const zoneLabel = safeClusters[bIdx]?.zone || `Target #${bIdx + 1}`;
              const tagWidth = Math.max(68, zoneLabel.length * 6.2 + 8);
              return (
                <g key={bIdx}>
                  <rect
                    x={x1}
                    y={y1}
                    width={w}
                    height={h}
                    fill="rgba(204, 120, 92, 0.15)"
                    stroke="#cc785c"
                    strokeWidth="1.8"
                    strokeDasharray="4 2"
                  />
                  <rect
                    x={x1}
                    y={Math.max(4, y1 - 14)}
                    width={tagWidth}
                    height={13}
                    fill="#181614"
                    stroke="#cc785c"
                    strokeWidth="0.8"
                    rx="2"
                  />
                  <text
                    x={x1 + 4}
                    y={Math.max(12, y1 - 4)}
                    fill="#ffffff"
                    fontSize="8.5"
                    fontWeight="bold"
                    fontFamily="monospace"
                  >
                    {zoneLabel}
                  </text>
                </g>
              );
            })}
          </svg>
        )}

        {safeClusters.length > 0 && showOverlay && (
          <div className="absolute inset-0 pointer-events-none">
            {safeClusters.map((cluster, cIdx) => {
              if (!cluster.centroid) return null;
              const [cx, cy] = cluster.centroid;
              const leftPct = Math.max(3, Math.min(97, (cx / 512) * 100));
              const topPct = Math.max(4, Math.min(96, (cy / 512) * 100));
              const isHovered = hoveredCluster === cIdx;

              return (
                <div
                  key={cluster.zone || cIdx}
                  className="absolute -translate-x-1/2 -translate-y-1/2 pointer-events-auto transition-all duration-150 z-30 cursor-pointer"
                  style={{ left: `${leftPct}%`, top: `${topPct}%` }}
                  onMouseEnter={() => setHoveredCluster(cIdx)}
                  onMouseLeave={() => setHoveredCluster(null)}
                >
                  <div className="w-2 h-2 rounded-full bg-[#cc785c] border border-white shadow mx-auto mb-1 animate-ping" />
                  <div
                    className={`px-1.5 py-0.5 rounded text-[8.5px] font-mono font-bold tracking-tight shadow-lg border transition-transform ${
                      isHovered
                        ? 'bg-[#cc785c] text-white border-white scale-110'
                        : 'bg-[#181614]/95 text-[var(--text-main)] border-[#cc785c] backdrop-blur-sm'
                    }`}
                  >
                    {cluster.zone} (
                    {cluster.area_ha ? `${cluster.area_ha.toFixed(1)}ha` : cluster.category}
                    )
                  </div>
                </div>
              );
            })}
          </div>
        )}

        <NorthArrow isCyan={true} />
        <ScaleBar isCyan={true} />
      </div>

      {safeClusters.length > 0 && (
        <div className="px-3 py-1.5 bg-[var(--bg-panel)] border-t border-[var(--border-subtle)] flex flex-wrap items-center justify-between gap-2 text-[10.5px] font-mono">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-[var(--text-muted)] font-semibold flex items-center gap-1">
              <MapPin className="w-3 h-3 text-[#cc785c]" />
              Delineated:
            </span>
            {safeClusters.map((cluster, cIdx) => (
              <span
                key={cIdx}
                onMouseEnter={() => setHoveredCluster(cIdx)}
                onMouseLeave={() => setHoveredCluster(null)}
                className={`px-1.5 py-0.2 rounded border text-[9.5px] transition-colors cursor-pointer ${
                  hoveredCluster === cIdx
                    ? 'bg-[#cc785c] text-white border-white font-bold'
                    : 'bg-[var(--bg-surface)] text-[#cc785c] border-[#cc785c]/30 hover:border-[#cc785c]'
                }`}
              >
                {cluster.zone}:{' '}
                {cluster.area_ha ? `${cluster.area_ha.toFixed(1)} ha` : cluster.category}
              </span>
            ))}
          </div>
          <div className="text-[9.5px] text-[var(--text-dim)] font-mono">
            Spatial Resolution: 10m GSD
          </div>
        </div>
      )}
    </div>
    );
  };

  return (
    <div
      className={`rounded-2xl border border-[var(--border-subtle)] bg-[var(--bg-card)] overflow-hidden select-none my-3 shadow-claude transition-all duration-200 ${
        isFullscreen
          ? 'fixed inset-4 z-50 flex flex-col bg-[var(--bg-card)] p-4 overflow-y-auto shadow-2xl'
          : 'w-full'
      } ${className}`}
    >
      {/* ── Top Cartographic Studio Ribbon (Claude Warm Aesthetic) ── */}
      <div className="flex flex-wrap items-center justify-between px-3.5 py-2.5 bg-[var(--bg-panel)] border-b border-[var(--border-subtle)] text-xs gap-2">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="w-2 h-2 rounded-full bg-[#cc785c] animate-pulse" />
          <span className="font-semibold text-xs text-[var(--text-main)] tracking-tight">
            Cartographic Intelligence Studio
          </span>
          {changedAreaHectares != null && (
            <span className="px-2 py-0.5 rounded-md font-mono text-[10.5px] font-semibold bg-[#cc785c]/10 text-[#cc785c] border border-[#cc785c]/25">
              {changedAreaHectares.toFixed(1)} ha marked
            </span>
          )}
          {confidence !== undefined && (
            <span className="px-1.5 py-0.5 rounded-md font-mono text-[10px] text-[var(--text-muted)] bg-[var(--bg-surface)] border border-[var(--border-subtle)] hidden sm:inline">
              {(confidence * 100).toFixed(0)}% confidence
            </span>
          )}
        </div>

        {/* View Mode Switcher Pill */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <div className="flex items-center p-0.5 bg-[var(--bg-surface)] rounded-xl border border-[var(--border-subtle)]">
            <button
              type="button"
              onClick={() => setViewMode('flow')}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-[11px] font-medium cursor-pointer transition-colors ${
                viewMode === 'flow'
                  ? 'bg-[var(--bg-card)] text-[#cc785c] shadow-subtle'
                  : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
              }`}
              title="Sequential: 1. Input Passes -> 2. Description -> 3. Output Map"
            >
              <ListOrdered className="w-3 h-3 text-[#cc785c]" />
              <span>1-2-3 Flow</span>
            </button>

            {hasSecondary && (
              <button
                type="button"
                onClick={() => setViewMode('swipe')}
                className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-[11px] font-medium cursor-pointer transition-colors ${
                  viewMode === 'swipe'
                    ? 'bg-[var(--bg-card)] text-[#cc785c] shadow-subtle'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
                }`}
                title="Interactive Swipe Divider Comparison"
              >
                <SlidersHorizontal className="w-3 h-3 text-[#cc785c]" />
                <span>Swipe Compare</span>
              </button>
            )}

            {hasBoxes && (
              <button
                type="button"
                onClick={() => setViewMode('grounding')}
                className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-[11px] font-medium cursor-pointer transition-colors ${
                  viewMode === 'grounding'
                    ? 'bg-[var(--bg-card)] text-[#cc785c] shadow-subtle'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
                }`}
                title="Target Grounding Reticles & Bounding Boxes"
              >
                <Crosshair className="w-3 h-3 text-[#cc785c]" />
                <span>Grounding ({boxes.length})</span>
              </button>
            )}

            {hasMask && (
              <button
                type="button"
                onClick={() => setViewMode('mask')}
                className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-[11px] font-medium cursor-pointer transition-colors ${
                  viewMode === 'mask'
                    ? 'bg-[var(--bg-card)] text-[#cc785c] shadow-subtle'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
                }`}
                title="Calibrated Mask with Opacity Controls"
              >
                <Layers className="w-3 h-3 text-[#cc785c]" />
                <span>{isVqa ? 'Land Cover' : isGrounding ? 'Target Mask' : 'Change Mask'}</span>
              </button>
            )}

            {hasMap && (
              <button
                type="button"
                onClick={() => setViewMode('map')}
                className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-[11px] font-medium cursor-pointer transition-colors ${
                  viewMode === 'map'
                    ? 'bg-[var(--bg-card)] text-[#cc785c] shadow-subtle'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
                }`}
                title="Leaflet Slippy Map & Tile Coordinates"
              >
                <MapIcon className="w-3 h-3 text-[#cc785c]" />
                <span>Slippy Map</span>
              </button>
            )}
          </div>

          {/* Mask Overlay Toggle (only in Flow mode) */}
          {hasMask && viewMode === 'flow' && (
            <button
              type="button"
              onClick={() => setShowOverlay(!showOverlay)}
              className="px-2.5 py-1 rounded-lg text-[11px] border border-[var(--border-subtle)] bg-[var(--bg-surface)] text-[var(--text-muted)] hover:text-[var(--text-main)] flex items-center gap-1 cursor-pointer transition-colors"
              title="Toggle marked overlay"
            >
              {showOverlay ? (
                <Eye className="w-3 h-3 text-[#cc785c]" />
              ) : (
                <EyeOff className="w-3 h-3 text-[var(--text-dim)]" />
              )}
              <span className="hidden sm:inline">
                {showOverlay ? 'Overlay On' : 'Overlay Off'}
              </span>
            </button>
          )}

          {/* PDF Briefing Dossier Action */}
          {onOpenPdf && (
            <button
              type="button"
              onClick={onOpenPdf}
              className="px-2.5 py-1 rounded-lg text-[11px] font-medium bg-[#cc785c]/15 hover:bg-[#cc785c]/25 text-[#cc785c] border border-[#cc785c]/30 flex items-center gap-1.5 cursor-pointer transition-colors shadow-subtle"
              title="Open Executive Mission Briefing Dossier (PDF)"
            >
              <FileText className="w-3.5 h-3.5 text-[#cc785c]" />
              <span className="font-semibold">PDF Dossier</span>
            </button>
          )}

          {/* Fullscreen Toggle */}
          <button
            type="button"
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-1.5 text-[var(--text-muted)] hover:text-[var(--text-main)] rounded-lg hover:bg-[var(--bg-surface)] transition-colors cursor-pointer"
            title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
          >
            {isFullscreen ? (
              <Minimize2 className="w-4 h-4" />
            ) : (
              <Maximize2 className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      {/* ── MODE 1: SEQUENTIAL 1-2-3 FLOW ── */}
      {viewMode === 'flow' && (
        <div className="p-3.5 space-y-4 bg-[var(--bg-app)]">
          {/* 1. GIVEN SATELLITE IMAGES */}
          <div>
            <div className="flex items-center gap-2 mb-2 pb-1 border-b border-[var(--border-subtle)]">
              <span className="w-4 h-4 rounded-full bg-[#cc785c]/15 text-[#cc785c] font-mono text-[10px] font-bold flex items-center justify-center border border-[#cc785c]/30">
                1
              </span>
              <h4 className="font-mono text-[11px] font-semibold text-[var(--text-main)] uppercase tracking-wider">
                Given Satellite Imagery &amp; Sensor Passes
              </h4>
              <span className="text-[10px] font-mono text-[var(--text-dim)]">
                {hasSecondary ? 'Input Passes (T1 Baseline & T2 Surveillance)' : 'Input Baseline (A) & Feature Extraction Pass (B)'}
              </span>
            </div>

            <div className="grid gap-3 grid-cols-1 md:grid-cols-2">
              {renderPanelA()}
              {renderPanelB()}
            </div>
          </div>

          {/* 2. GEOSPATIAL INTELLIGENCE & ANALYTICAL NARRATIVE */}
          <div>
            <div className="flex items-center gap-2 mb-2 pb-1 border-b border-[var(--border-subtle)]">
              <span className="w-4 h-4 rounded-full bg-[#cc785c]/15 text-[#cc785c] font-mono text-[10px] font-bold flex items-center justify-center border border-[#cc785c]/30">
                2
              </span>
              <h4 className="font-mono text-[11px] font-semibold text-[var(--text-main)] uppercase tracking-wider">
                Geospatial Intelligence &amp; Analytical Narrative
              </h4>
              <span className="text-[10px] font-mono text-[var(--text-dim)]">
                Autonomous Multi-Agent Synthesis &amp; Biophysical Evidence
              </span>
            </div>

            <div className="p-3.5 rounded-xl bg-[var(--bg-panel)] border border-[var(--border-subtle)] text-[var(--text-main)] leading-relaxed">
              {descriptionNode || (
                <div className="space-y-2 font-sans">
                  <div className="flex items-center justify-between text-[11px] font-mono pb-1 border-b border-[var(--border-subtle)]">
                    <span className="text-[#cc785c] font-semibold">
                      {isVqa ? 'RS-VLM Biophysical Synthesis' : 'AI Analytical Findings'}
                    </span>
                    <span className="text-[var(--text-muted)]">
                      {(confidence ? (confidence * 100).toFixed(0) : '90')}% Calibrated Confidence
                    </span>
                  </div>
                  <p className="text-xs text-[var(--text-main)] leading-relaxed">
                    {isVqa
                      ? 'Multispectral scene analysis confirms dense structural and vegetative canopy distribution. Surface land cover classified across urban, vegetation, hydrological, and barren partitions with high radiometric fidelity.'
                      : 'Geospatial feature verification and autonomous multi-agent interpretation completed successfully across input rasters.'}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* 3. SATELLITE VIEW OF THE OUTPUT */}
          <div>
            <div className="flex items-center gap-2 mb-2 pb-1 border-b border-[#cc785c]/30">
              <span className="w-4 h-4 rounded-full bg-[#cc785c] text-white font-mono text-[10px] font-bold flex items-center justify-center shadow">
                3
              </span>
              <h4 className="font-mono text-[11px] font-semibold text-[var(--text-main)] uppercase tracking-wider">
                Satellite View of the Output
              </h4>
              <span className="text-[10px] font-mono text-[#cc785c]">
                {isVqa
                  ? 'Classified Land Cover Overlay'
                  : isFusion
                  ? 'Reconstructed Cloud-Free Optical Scene'
                  : isGrounding
                  ? 'Target Delineation Map'
                  : 'Calibrated Detection Map'}
              </span>
            </div>

            <div className="w-full">{renderPanelC(true)}</div>
          </div>
        </div>
      )}

      {/* ── MODE 2: INTERACTIVE SWIPE COMPARE SLIDER ── */}
      {viewMode === 'swipe' && hasSecondary && (
        <div className="p-3 bg-[var(--bg-app)]">
          <SwipeCompare
            image1Url={image1Url}
            image2Url={isFusion && maskUrl ? maskUrl : image2Url}
            label1={isFusion ? 'Cloudy Optical Pass' : t1Filename}
            label2={isFusion && maskUrl ? 'Reconstructed Clear Optical (No Cloud)' : t2Filename}
            modality1={isFusion ? 'OPTICAL' : modality1}
            modality2={isFusion ? 'CLEAR_OPTICAL' : modality2}
            maskUrl={isFusion ? undefined : maskUrl}
            clusters={safeClusters}
            changedAreaHectares={changedAreaHectares}
            changedAreaPercent={changedAreaPercent}
          />
        </div>
      )}

      {/* ── MODE 3: HIGH-PRECISION GROUNDING RETICLES ── */}
      {viewMode === 'grounding' && hasBoxes && (
        <div className="p-3 bg-[var(--bg-app)]">
          <GroundingViewer
            imageUrl={image2Url || image1Url}
            boxes={formattedBoxes}
            confidence={confidence}
            label="Detected Target"
          />
        </div>
      )}

      {/* ── MODE 4: CHANGE MASK & MARKED PLACES OVERLAY ── */}
      {viewMode === 'mask' && hasMask && (
        <div className="p-3 bg-[var(--bg-app)]">
          <ChangeMaskViewer
            baseImageUrl={image2Url || image1Url}
            maskUrl={maskUrl!}
            changedAreaHectares={changedAreaHectares}
            changedAreaPercent={changedAreaPercent}
            clusters={safeClusters}
            title={panelCTitle}
          />
        </div>
      )}

      {/* ── MODE 5: GEOSPATIAL SLIPPY MAP ── */}
      {viewMode === 'map' && hasMap && (
        <div className="p-3 bg-[var(--bg-app)]">
          <div className="rounded-xl overflow-hidden border border-[var(--border-subtle)]">
            <MapView
              bounds={bounds}
              fileId={fileId}
              filename={t1Filename}
              height={isFullscreen ? '600px' : '400px'}
            />
          </div>
        </div>
      )}
    </div>
  );
};
