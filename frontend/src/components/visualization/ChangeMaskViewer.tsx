import React, { useState } from 'react';
import { Layers, Sliders, Eye, EyeOff, Sparkles, MapPin } from 'lucide-react';
import { Badge } from '../ui/Badge';
import { SatQueryAPI } from '../../services/api';
import type { SpatialCluster } from '../../types/api';

interface ChangeMaskViewerProps {
  baseImageUrl: string;
  maskUrl: string;
  changedAreaHectares?: number | null;
  changedAreaPercent?: number | null;
  clusters?: SpatialCluster[];
  title?: string;
  badgeLabel?: string;
}

export const ChangeMaskViewer: React.FC<ChangeMaskViewerProps> = ({
  baseImageUrl,
  maskUrl,
  changedAreaHectares,
  changedAreaPercent,
  clusters = [],
  title = 'Marked Places & Detection Overlay',
  badgeLabel = 'Neural Evidence',
}) => {
  const [opacity, setOpacity] = useState<number>(0.55);
  const [showMask, setShowMask] = useState<boolean>(true);
  const [showPins, setShowPins] = useState<boolean>(true);
  const [hoveredClusterIndex, setHoveredClusterIndex] = useState<number | null>(null);
  const [isPeeking, setIsPeeking] = useState<boolean>(false);
  const [blendMode, setBlendMode] = useState<'normal' | 'screen' | 'overlay'>('screen');

  const isMaskVisible = showMask && !isPeeking;
  const safeClusters = clusters || [];

  return (
    <div className="w-full rounded-xl overflow-hidden border border-[var(--border-subtle)] bg-[var(--bg-card)] my-3 select-none shadow-subtle">
      {/* ── Top Header ────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between px-3.5 py-2 bg-[var(--bg-panel)] border-b border-[var(--border-subtle)] text-xs gap-2">
        <div className="flex items-center gap-2">
          <Layers className="w-3.5 h-3.5 text-[#0EA5E9]" />
          <span className="font-medium text-[var(--text-main)] text-[11px]">
            {title}
          </span>
          <Badge variant="neutral">{badgeLabel}</Badge>
          {safeClusters.length > 0 && (
            <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-[#0EA5E9]/10 text-[#0EA5E9] border border-[#0EA5E9]/25">
              {safeClusters.length} Places Marked
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {/* Pins Toggle */}
          {safeClusters.length > 0 && (
            <button
              type="button"
              onClick={() => setShowPins(!showPins)}
              className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-colors flex items-center gap-1 cursor-pointer ${
                showPins
                  ? 'bg-[var(--bg-surface)] text-[var(--text-main)] border-[#0EA5E9]/40 font-medium'
                  : 'bg-[var(--bg-card)] text-[var(--text-dim)] border-[var(--border-subtle)] opacity-60'
              }`}
              title="Show or hide place markers"
            >
              <MapPin className="w-2.5 h-2.5 text-[#0EA5E9]" />
              <span>Pins: {showPins ? 'ON' : 'OFF'}</span>
            </button>
          )}

          {/* Opacity Slider */}
          <div className="flex items-center gap-1.5 bg-[var(--bg-surface)] px-2 py-0.5 rounded-lg border border-[var(--border-subtle)]">
            <Sliders className="w-2.5 h-2.5 text-[var(--text-dim)]" />
            <input
              type="range"
              min={0.05}
              max={1}
              step={0.05}
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              className="w-14 h-1 bg-[var(--border-subtle)] rounded-lg appearance-none cursor-pointer accent-[#0EA5E9]"
              title="Mask Opacity"
            />
            <span className="text-[10px] font-mono text-[var(--text-main)] w-7">
              {Math.round(opacity * 100)}%
            </span>
          </div>

          {/* Blend Mode Toggle */}
          <button
            type="button"
            onClick={() => {
              if (blendMode === 'screen') setBlendMode('normal');
              else if (blendMode === 'normal') setBlendMode('overlay');
              else setBlendMode('screen');
            }}
            className="px-2 py-0.5 rounded text-[10px] font-mono border border-[var(--border-subtle)] bg-[var(--bg-surface)] text-[var(--text-muted)] hover:text-[var(--text-main)] flex items-center gap-1 cursor-pointer transition-colors"
            title="Toggle blend mode"
          >
            <Sparkles className="w-2.5 h-2.5 text-[#0EA5E9]" />
            <span>Blend: {blendMode.toUpperCase()}</span>
          </button>

          {/* Hold to Peek Pure Satellite Imagery */}
          <button
            type="button"
            onMouseDown={() => setIsPeeking(true)}
            onMouseUp={() => setIsPeeking(false)}
            onMouseLeave={() => setIsPeeking(false)}
            onTouchStart={() => setIsPeeking(true)}
            onTouchEnd={() => setIsPeeking(false)}
            className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-colors flex items-center gap-1 cursor-pointer ${
              isPeeking
                ? 'bg-[#0EA5E9]/20 text-[#0EA5E9] border-[#0EA5E9]/40 font-semibold'
                : 'bg-[var(--bg-surface)] text-[var(--text-muted)] hover:text-[var(--text-main)] border-[var(--border-subtle)]'
            }`}
            title="Hold to Peek"
          >
            <Eye className="w-2.5 h-2.5" />
            <span>{isPeeking ? 'Peeking...' : 'Hold Peek'}</span>
          </button>

          {/* Mask On/Off */}
          <button
            type="button"
            onClick={() => setShowMask(!showMask)}
            className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-colors flex items-center gap-1 cursor-pointer ${
              showMask
                ? 'bg-[var(--bg-surface)] text-[var(--text-main)] border-[var(--border-subtle)] font-medium'
                : 'bg-[var(--bg-card)] text-[var(--text-dim)] border-[var(--border-subtle)] opacity-60'
            }`}
          >
            {showMask ? <Eye className="w-2.5 h-2.5" /> : <EyeOff className="w-2.5 h-2.5" />}
            <span>{showMask ? 'MASK ON' : 'MASK OFF'}</span>
          </button>
        </div>
      </div>

      {/* ── Raster & Mask Container ─────────────────────────────── */}
      <div className="relative w-full aspect-square max-h-[460px] bg-black flex items-center justify-center overflow-hidden">
        <img
          src={SatQueryAPI.getRasterPreviewUrl(baseImageUrl || 'cartosat_t1.tif')}
          alt="Base Raster"
          className="w-full h-full object-contain pointer-events-none"
          onError={(e) => {
            (e.currentTarget as HTMLImageElement).src = SatQueryAPI.getRasterPreviewUrl('cartosat_t1.tif');
          }}
        />

        {isMaskVisible && (
          <img
            src={SatQueryAPI.getRasterPreviewUrl(maskUrl || 'flood_t2.tif')}
            alt="Change Mask"
            className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-150"
            style={{
              opacity,
              mixBlendMode: blendMode === 'screen' ? 'screen' : blendMode === 'overlay' ? 'overlay' : 'normal',
            }}
            onError={(e) => {
              (e.currentTarget as HTMLImageElement).src = SatQueryAPI.getRasterPreviewUrl('flood_t2.tif');
            }}
          />
        )}

        {/* Interactive Cluster / Place Callout Pins */}
        {showPins && safeClusters.length > 0 && (
          <div className="absolute inset-0 pointer-events-none">
            {safeClusters.map((cluster, idx) => {
              if (!cluster.centroid) return null;
              const [cx, cy] = cluster.centroid;
              const leftPct = Math.max(5, Math.min(95, (cx / 512) * 100));
              const topPct = Math.max(5, Math.min(95, (cy / 512) * 100));
              const isHovered = hoveredClusterIndex === idx;

              return (
                <div
                  key={cluster.zone || idx}
                  className="absolute -translate-x-1/2 -translate-y-1/2 pointer-events-auto transition-all duration-200 z-30 group cursor-pointer"
                  style={{ left: `${leftPct}%`, top: `${topPct}%` }}
                  onMouseEnter={() => setHoveredClusterIndex(idx)}
                  onMouseLeave={() => setHoveredClusterIndex(null)}
                >
                  <div className="relative flex items-center justify-center">
                    <span
                      className={`absolute w-7 h-7 rounded-full bg-[#0EA5E9] opacity-75 animate-ping ${
                        isHovered ? 'scale-150' : ''
                      }`}
                    />
                    <div
                      className={`w-5 h-5 rounded-full flex items-center justify-center shadow-lg border-2 border-white transition-transform ${
                        isHovered
                          ? 'bg-[#0EA5E9] scale-125 ring-4 ring-[#0EA5E9]/40'
                          : 'bg-[#0EA5E9]'
                      }`}
                    >
                      <MapPin className="w-2.5 h-2.5 text-white" />
                    </div>
                  </div>

                  <div
                    className={`mt-1 px-1.5 py-0.2 rounded text-[9px] font-mono font-bold whitespace-nowrap shadow border transition-all text-center ${
                      isHovered
                        ? 'bg-[var(--bg-card)] text-[#0EA5E9] border-[#0EA5E9]'
                        : 'bg-black/85 text-white border-white/30 backdrop-blur-sm'
                    }`}
                  >
                    {cluster.zone}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* ── Marked Places & Zonal Details List ──────────────────── */}
      {safeClusters.length > 0 && (
        <div className="p-3 bg-[var(--bg-panel)] border-t border-[var(--border-subtle)]">
          <div className="flex items-center justify-between mb-2">
            <span className="font-semibold text-xs text-[var(--text-main)] flex items-center gap-1.5">
              <MapPin className="w-3 h-3 text-[#0EA5E9]" />
              Marked Places & Delineated Sectors
            </span>
            <span className="text-[10px] font-mono text-[var(--text-dim)]">
              Hover place card to highlight on satellite image
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {safeClusters.map((cluster, idx) => {
              const isHovered = hoveredClusterIndex === idx;
              const [cx, cy] = cluster.centroid || [0, 0];

              return (
                <div
                  key={cluster.zone || idx}
                  onMouseEnter={() => setHoveredClusterIndex(idx)}
                  onMouseLeave={() => setHoveredClusterIndex(null)}
                  className={`p-2.5 rounded-xl border transition-all cursor-pointer ${
                    isHovered
                      ? 'bg-[#0EA5E9]/10 border-[#0EA5E9] shadow-sm'
                      : 'bg-[var(--bg-surface)] border-[var(--border-subtle)] hover:border-[#0EA5E9]/40'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-bold text-xs text-[var(--text-main)] flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-[#0EA5E9]" />
                      {cluster.zone}
                    </span>
                    {cluster.area_ha !== undefined && cluster.area_ha !== null && (
                      <span className="text-[10px] font-mono font-semibold px-1.5 py-0.2 rounded bg-[var(--bg-card)] border border-[var(--border-subtle)] text-[#0EA5E9]">
                        {cluster.area_ha} ha
                      </span>
                    )}
                  </div>

                  <p className="text-[11px] text-[var(--text-muted)] leading-snug mb-1">
                    {cluster.category}
                  </p>

                  <div className="flex items-center justify-between text-[10px] font-mono text-[var(--text-dim)] border-t border-[var(--border-subtle)]/60 pt-1">
                    <span>Centroid: ({cx}, {cy})</span>
                    {cluster.pixel_count && <span>{cluster.pixel_count.toLocaleString()} px</span>}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Quantitative Readouts Bar */}
      {(changedAreaHectares !== undefined && changedAreaHectares !== null) ||
      (changedAreaPercent !== undefined && changedAreaPercent !== null) ? (
        <div className="p-3 bg-[var(--bg-panel)] border-t border-[var(--border-subtle)] grid grid-cols-2 gap-2 text-xs">
          <div className="p-2.5 rounded-xl bg-[var(--bg-card)] border border-[var(--border-subtle)] shadow-subtle">
            <span className="block text-[10px] text-[var(--text-dim)] uppercase font-mono tracking-wider">
              Marked Extent
            </span>
            <span className="text-sm font-semibold font-mono text-[var(--text-main)]">
              {changedAreaHectares !== undefined && changedAreaHectares !== null
                ? `${changedAreaHectares.toFixed(1)} ha`
                : 'Computed in mask'}
            </span>
          </div>

          <div className="p-2.5 rounded-xl bg-[var(--bg-card)] border border-[var(--border-subtle)] shadow-subtle">
            <span className="block text-[10px] text-[var(--text-dim)] uppercase font-mono tracking-wider">
              Scene Share
            </span>
            <span className="text-sm font-semibold font-mono text-[var(--text-main)]">
              {changedAreaPercent !== undefined && changedAreaPercent !== null
                ? `${changedAreaPercent.toFixed(1)}%`
                : 'Computed in mask'}
            </span>
          </div>
        </div>
      ) : null}
    </div>
  );
};
