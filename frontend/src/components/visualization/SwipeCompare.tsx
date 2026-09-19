import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  Columns,
  ArrowLeftRight,
  Layers,
  Eye,
  Sliders,
  Sparkles,
  MapPin,
  Crosshair,
} from 'lucide-react';
import { Badge } from '../ui/Badge';
import type { SpatialCluster } from '../../types/api';
import { SatQueryAPI } from '../../services/api';

export interface SwipeCompareProps {
  image1Url: string;
  image2Url: string;
  label1?: string;
  label2?: string;
  modality1?: string;
  modality2?: string;
  maskUrl?: string | null;
  clusters?: SpatialCluster[];
  changedAreaHectares?: number | null;
  changedAreaPercent?: number | null;
  taskTitle?: string;
}

export const SwipeCompare: React.FC<SwipeCompareProps> = ({
  image1Url,
  image2Url,
  label1 = 'Scene 1',
  label2 = 'Scene 2',
  modality1 = 'OPTICAL',
  modality2 = 'SAR',
  maskUrl,
  clusters = [],
  changedAreaHectares,
  changedAreaPercent,
}) => {
  const img1 = SatQueryAPI.getRasterPreviewUrl(image1Url);
  const img2 = SatQueryAPI.getRasterPreviewUrl(image2Url);
  const mUrl = maskUrl ? SatQueryAPI.getRasterPreviewUrl(maskUrl) : null;
  const [sliderPos, setSliderPos] = useState<number>(50);
  // Default to 'overlay' if a mask is present so the marked places are immediately visible to the user!
  const [mode, setMode] = useState<'slider' | 'overlay' | 'side-by-side'>(() =>
    maskUrl ? 'overlay' : 'slider'
  );
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [showMaskInSlider, setShowMaskInSlider] = useState<boolean>(true);
  const [showPins, setShowPins] = useState<boolean>(true);
  const [hoveredClusterIndex, setHoveredClusterIndex] = useState<number | null>(null);
  const [opacity, setOpacity] = useState<number>(0.55);
  const [isPeeking, setIsPeeking] = useState<boolean>(false);
  const [blendMode, setBlendMode] = useState<'screen' | 'normal' | 'overlay'>('screen');

  const containerRef = useRef<HTMLDivElement | null>(null);

  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPos(pct);
  }, []);

  useEffect(() => {
    if (!isDragging) return;

    const handleWindowMouseMove = (e: MouseEvent) => {
      handleMove(e.clientX);
    };

    const handleWindowMouseUp = () => {
      setIsDragging(false);
    };

    const handleWindowTouchMove = (e: TouchEvent) => {
      if (e.touches.length > 0) {
        handleMove(e.touches[0].clientX);
      }
    };

    const handleWindowTouchEnd = () => {
      setIsDragging(false);
    };

    window.addEventListener('mousemove', handleWindowMouseMove);
    window.addEventListener('mouseup', handleWindowMouseUp);
    window.addEventListener('touchmove', handleWindowTouchMove);
    window.addEventListener('touchend', handleWindowTouchEnd);

    return () => {
      window.removeEventListener('mousemove', handleWindowMouseMove);
      window.removeEventListener('mouseup', handleWindowMouseUp);
      window.removeEventListener('touchmove', handleWindowTouchMove);
      window.removeEventListener('touchend', handleWindowTouchEnd);
    };
  }, [isDragging, handleMove]);

  const hasMask = !!maskUrl;
  const isMaskActive = hasMask && !isPeeking;
  const safeClusters = clusters || [];

  return (
    <div className="w-full rounded-xl overflow-hidden border border-[var(--border-subtle)] bg-[var(--bg-card)] my-3 select-none shadow-subtle">
      {/* ── Top Header Ribbon ────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between px-3.5 py-2.5 bg-[var(--bg-panel)] border-b border-[var(--border-subtle)] text-xs gap-2">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-semibold text-[var(--text-main)] text-[11px] flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-[#cc785c]" />
            Satellite Inspection Studio
          </span>
          <div className="flex items-center gap-1.5 font-mono">
            <Badge variant="neutral">{modality1}</Badge>
            <span className="text-[var(--text-dim)] text-[10px]">vs</span>
            <Badge variant="neutral">{modality2}</Badge>
          </div>
          {safeClusters.length > 0 && (
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#cc785c]/10 text-[#cc785c] border border-[#cc785c]/25 font-medium">
              {safeClusters.length} Marked Place{safeClusters.length === 1 ? '' : 's'}
            </span>
          )}
        </div>

        {/* Mode Selector Tabs (Marked Places | Swipe | Side by Side) */}
        <div className="flex items-center gap-1 p-0.5 bg-[var(--bg-surface)] rounded-lg border border-[var(--border-subtle)]">
          {hasMask && (
            <button
              type="button"
              onClick={() => setMode('overlay')}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-[10.5px] font-mono transition-all cursor-pointer ${
                mode === 'overlay'
                  ? 'bg-[var(--bg-card)] text-[#cc785c] font-semibold shadow-subtle border border-[var(--border-subtle)]'
                  : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
              }`}
              title="View marked places with neural detection mask overlay"
            >
              <Crosshair className="w-3 h-3 text-[#cc785c]" />
              <span>Marked Places</span>
            </button>
          )}

          <button
            type="button"
            onClick={() => setMode('slider')}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-[10.5px] font-mono transition-all cursor-pointer ${
              mode === 'slider'
                ? 'bg-[var(--bg-card)] text-[#cc785c] font-semibold shadow-subtle border border-[var(--border-subtle)]'
                : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
            }`}
            title="Draggable dual-image comparator swipe"
          >
            <ArrowLeftRight className="w-3 h-3 text-[#cc785c]" />
            <span>Swipe</span>
          </button>

          <button
            type="button"
            onClick={() => setMode('side-by-side')}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-[10.5px] font-mono transition-all cursor-pointer ${
              mode === 'side-by-side'
                ? 'bg-[var(--bg-card)] text-[#cc785c] font-semibold shadow-subtle border border-[var(--border-subtle)]'
                : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
            }`}
            title="View both scenes side-by-side"
          >
            <Columns className="w-3 h-3 text-[#cc785c]" />
            <span>Side by Side</span>
          </button>
        </div>
      </div>

      {/* ── Sub-Toolbar for Visual Controls (Opacity, Peek, Pins) ─────── */}
      <div className="flex flex-wrap items-center justify-between px-3.5 py-1.5 bg-[var(--bg-panel)]/80 border-b border-[var(--border-subtle)] text-[10.5px] gap-2">
        <div className="flex items-center gap-2 flex-wrap">
          {/* Toggle Pins */}
          {safeClusters.length > 0 && (
            <button
              type="button"
              onClick={() => setShowPins(!showPins)}
              className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-colors flex items-center gap-1 cursor-pointer ${
                showPins
                  ? 'bg-[var(--bg-surface)] text-[var(--text-main)] border-[#cc785c]/40 font-medium'
                  : 'bg-[var(--bg-card)] text-[var(--text-dim)] border-[var(--border-subtle)] opacity-60'
              }`}
              title="Toggle interactive pins marking each detected place on the satellite image"
            >
              <MapPin className="w-2.5 h-2.5 text-[#cc785c]" />
              <span>Pins: {showPins ? 'ON' : 'OFF'}</span>
            </button>
          )}

          {/* If in Swipe mode and mask exists, allow toggling mask on top of Scene 1 */}
          {hasMask && mode === 'slider' && (
            <button
              type="button"
              onClick={() => setShowMaskInSlider(!showMaskInSlider)}
              className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-colors flex items-center gap-1 cursor-pointer ${
                showMaskInSlider
                  ? 'bg-[var(--bg-surface)] text-[var(--text-main)] border-[#cc785c]/40 font-medium'
                  : 'bg-[var(--bg-card)] text-[var(--text-dim)] border-[var(--border-subtle)] opacity-60'
              }`}
              title="Show or hide marked places overlay on Scene 1 while swiping"
            >
              <Layers className="w-2.5 h-2.5 text-[#cc785c]" />
              <span>Marked Overlay: {showMaskInSlider ? 'ON' : 'OFF'}</span>
            </button>
          )}

          {/* Opacity Slider in Overlay Mode */}
          {hasMask && (mode === 'overlay' || (mode === 'slider' && showMaskInSlider)) && (
            <div className="flex items-center gap-1.5 bg-[var(--bg-surface)] px-2 py-0.5 rounded border border-[var(--border-subtle)]">
              <Sliders className="w-2.5 h-2.5 text-[var(--text-dim)]" />
              <input
                type="range"
                min={0.1}
                max={1}
                step={0.05}
                value={opacity}
                onChange={(e) => setOpacity(parseFloat(e.target.value))}
                className="w-14 h-1 bg-[var(--border-subtle)] rounded-lg appearance-none cursor-pointer accent-[#cc785c]"
                title="Mask Opacity"
              />
              <span className="text-[10px] font-mono text-[var(--text-main)] w-7">
                {Math.round(opacity * 100)}%
              </span>
            </div>
          )}

          {/* Blend Mode Toggle */}
          {hasMask && mode === 'overlay' && (
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
              <Sparkles className="w-2.5 h-2.5 text-[#cc785c]" />
              <span>Blend: {blendMode.toUpperCase()}</span>
            </button>
          )}

          {/* Hold to Peek Pure Satellite Imagery */}
          {hasMask && (
            <button
              type="button"
              onMouseDown={() => setIsPeeking(true)}
              onMouseUp={() => setIsPeeking(false)}
              onMouseLeave={() => setIsPeeking(false)}
              onTouchStart={() => setIsPeeking(true)}
              onTouchEnd={() => setIsPeeking(false)}
              className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-colors flex items-center gap-1 cursor-pointer ${
                isPeeking
                  ? 'bg-[#cc785c]/20 text-[#cc785c] border-[#cc785c]/40 font-semibold'
                  : 'bg-[var(--bg-surface)] text-[var(--text-muted)] hover:text-[var(--text-main)] border-[var(--border-subtle)]'
              }`}
              title="Hold to temporarily hide the overlay and see the raw satellite pixels"
            >
              <Eye className="w-2.5 h-2.5" />
              <span>{isPeeking ? 'Peeking...' : 'Hold Peek'}</span>
            </button>
          )}
        </div>

        <div className="text-[10px] font-mono text-[var(--text-dim)] hidden sm:block">
          {mode === 'overlay'
            ? 'Marked places & neural evidence overlay active'
            : mode === 'slider'
            ? 'Drag divider to compare Scene 1 vs Scene 2'
            : 'Dual-sensor aligned view'}
        </div>
      </div>

      {/* ── Main Canvas View Area ─────────────────────────────────── */}

      {/* MODE 1: MARKED PLACES OVERLAY VIEW */}
      {mode === 'overlay' && (
        <div className="relative w-full aspect-square max-h-[460px] bg-black overflow-hidden flex items-center justify-center">
          {/* Base Satellite Image (Scene 1) */}
          <img
            src={img1}
            alt={label1}
            className="w-full h-full object-contain pointer-events-none"
          />

          {/* Marked Places Mask Overlay */}
          {isMaskActive && mUrl && (
            <img
              src={mUrl}
              alt="Neural Marked Places Mask"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-150"
              style={{
                opacity,
                mixBlendMode: blendMode === 'screen' ? 'screen' : blendMode === 'overlay' ? 'overlay' : 'normal',
              }}
            />
          )}

          {/* Interactive Cluster / Place Callout Pins */}
          {showPins && safeClusters.length > 0 && (
            <div className="absolute inset-0 pointer-events-none">
              {safeClusters.map((cluster, idx) => {
                if (!cluster.centroid) return null;
                const [cx, cy] = cluster.centroid;
                // Normalize 512x512 coordinates to percentage
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
                    {/* Pulsing Pin Marker */}
                    <div className="relative flex items-center justify-center">
                      <span
                        className={`absolute w-7 h-7 rounded-full bg-[#cc785c] opacity-75 animate-ping ${
                          isHovered ? 'scale-150' : ''
                        }`}
                      />
                      <div
                        className={`w-5 h-5 rounded-full flex items-center justify-center shadow-lg border-2 border-white transition-transform ${
                          isHovered
                            ? 'bg-[#cc785c] scale-125 ring-4 ring-[#cc785c]/40'
                            : 'bg-[#cc785c]'
                        }`}
                      >
                        <MapPin className="w-2.5 h-2.5 text-white" />
                      </div>
                    </div>

                    {/* Pin Label Badge */}
                    <div
                      className={`mt-1.5 px-2 py-0.5 rounded-md text-[9.5px] font-mono font-bold whitespace-nowrap shadow-md border transition-all text-center ${
                        isHovered
                          ? 'bg-[var(--bg-card)] text-[#cc785c] border-[#cc785c] scale-105'
                          : 'bg-black/85 text-white border-white/30 backdrop-blur-sm'
                      }`}
                    >
                      {cluster.zone}
                      {cluster.area_ha ? ` (${cluster.area_ha} ha)` : ''}
                    </div>

                    {/* Rich Tooltip on Hover */}
                    {isHovered && (
                      <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 w-52 p-2.5 rounded-xl bg-[var(--bg-card)]/95 backdrop-blur-md border border-[var(--border-subtle)] shadow-xl text-left z-40 animate-fadeIn">
                        <div className="flex items-center justify-between mb-1 pb-1 border-b border-[var(--border-subtle)]">
                          <span className="font-bold text-xs text-[var(--text-main)]">
                            {cluster.zone}
                          </span>
                          {cluster.area_ha && (
                            <Badge variant="warning">{cluster.area_ha} ha</Badge>
                          )}
                        </div>
                        <p className="text-[11px] text-[var(--text-main)] font-medium mb-1 leading-snug">
                          {cluster.category}
                        </p>
                        <div className="text-[10px] font-mono text-[var(--text-dim)]">
                          Centroid: ({cx}, {cy})
                          {cluster.pixel_count ? ` • ${cluster.pixel_count.toLocaleString()} px` : ''}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* Floating Scene Label */}
          <div className="absolute top-3 left-3 z-20 pointer-events-none">
            <span className="px-2.5 py-1 rounded-lg bg-[var(--bg-panel)]/90 backdrop-blur-sm border border-[var(--border-subtle)] text-[var(--text-main)] text-[10.5px] font-mono shadow-subtle flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-[#cc785c] animate-pulse" />
              {label1} • Marked Places Overlay
            </span>
          </div>
        </div>
      )}

      {/* MODE 2: SWIPE COMPARISON VIEW */}
      {mode === 'slider' && (
        <div
          ref={containerRef}
          onMouseDown={(e) => {
            setIsDragging(true);
            handleMove(e.clientX);
          }}
          onTouchStart={(e) => {
            if (e.touches.length > 0) {
              setIsDragging(true);
              handleMove(e.touches[0].clientX);
            }
          }}
          className="relative w-full aspect-square max-h-[460px] bg-black overflow-hidden cursor-ew-resize"
        >
          {/* Underneath Image 2 (SAR / Post) */}
          <img
            src={img2}
            alt={label2}
            className="absolute inset-0 w-full h-full object-contain pointer-events-none"
          />

          {/* Clipped Top Image 1 (Optical / Pre) */}
          <div
            className="absolute inset-0 overflow-hidden pointer-events-none"
            style={{ clipPath: `polygon(0 0, ${sliderPos}% 0, ${sliderPos}% 100%, 0 100%)` }}
          >
            <img
              src={img1}
              alt={label1}
              className="absolute inset-0 w-full h-full object-contain pointer-events-none"
            />

            {/* Optional Marked Overlay inside the Optical Swipe Slice */}
            {hasMask && showMaskInSlider && isMaskActive && mUrl && (
              <img
                src={mUrl}
                alt="Marked Places Mask"
                className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                style={{
                  opacity,
                  mixBlendMode: blendMode === 'screen' ? 'screen' : 'normal',
                }}
              />
            )}
          </div>

          {/* Interactive Cluster Pins on Slider Canvas */}
          {showPins && safeClusters.length > 0 && (
            <div className="absolute inset-0 pointer-events-none z-20">
              {safeClusters.map((cluster, idx) => {
                if (!cluster.centroid) return null;
                const [cx, cy] = cluster.centroid;
                const leftPct = Math.max(5, Math.min(95, (cx / 512) * 100));
                const topPct = Math.max(5, Math.min(95, (cy / 512) * 100));
                const isHovered = hoveredClusterIndex === idx;

                return (
                  <div
                    key={cluster.zone || idx}
                    className="absolute -translate-x-1/2 -translate-y-1/2 pointer-events-auto cursor-pointer"
                    style={{ left: `${leftPct}%`, top: `${topPct}%` }}
                    onMouseEnter={() => setHoveredClusterIndex(idx)}
                    onMouseLeave={() => setHoveredClusterIndex(null)}
                  >
                    <div
                      className={`w-4 h-4 rounded-full flex items-center justify-center shadow-md border-2 border-white transition-transform ${
                        isHovered ? 'bg-[#cc785c] scale-125 ring-2 ring-[#cc785c]' : 'bg-[#cc785c]'
                      }`}
                    >
                      <MapPin className="w-2 h-2 text-white" />
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Divider Line & Handle */}
          <div
            className="absolute top-0 bottom-0 pointer-events-none z-30"
            style={{
              left: `${sliderPos}%`,
              width: '2px',
              backgroundColor: '#cc785c',
              boxShadow: '0 0 10px rgba(0, 0, 0, 0.6)',
            }}
          >
            <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-7 h-7 rounded-full bg-[var(--bg-card)] border-2 border-[#cc785c] shadow-lg flex items-center justify-center font-mono font-bold text-[11px] text-[#cc785c]">
              ⇄
            </div>
          </div>

          {/* Clean Floating Labels */}
          <div className="absolute top-3 left-3 z-20 pointer-events-none">
            <span className="px-2.5 py-1 rounded-lg bg-[var(--bg-panel)]/90 backdrop-blur-sm border border-[var(--border-subtle)] text-[var(--text-main)] text-[10.5px] font-mono shadow-subtle">
              {label1} {hasMask && showMaskInSlider ? '(with Marked Overlay)' : ''}
            </span>
          </div>
          <div className="absolute top-3 right-3 z-20 pointer-events-none">
            <span className="px-2.5 py-1 rounded-lg bg-[var(--bg-panel)]/90 backdrop-blur-sm border border-[var(--border-subtle)] text-[var(--text-main)] text-[10.5px] font-mono shadow-subtle">
              {label2}
            </span>
          </div>
        </div>
      )}

      {/* MODE 3: SIDE BY SIDE VIEW */}
      {mode === 'side-by-side' && (
        <div className="grid grid-cols-1 sm:grid-cols-2 divide-y sm:divide-y-0 sm:divide-x divide-[var(--border-subtle)] bg-black max-h-[440px]">
          <div className="relative aspect-square flex items-center justify-center p-2">
            <img src={img1} alt={label1} className="w-full h-full object-contain" />
            {hasMask && isMaskActive && mUrl && (
              <img
                src={mUrl}
                alt="Mask Overlay"
                className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                style={{ opacity: 0.6, mixBlendMode: 'screen' }}
              />
            )}
            <div className="absolute top-3 left-3">
              <span className="px-2.5 py-1 rounded-lg bg-[var(--bg-panel)]/90 backdrop-blur-sm border border-[var(--border-subtle)] text-[var(--text-main)] text-[10.5px] font-mono shadow-subtle">
                {label1}
              </span>
            </div>
          </div>
          <div className="relative aspect-square flex items-center justify-center p-2">
            <img src={img2} alt={label2} className="w-full h-full object-contain" />
            <div className="absolute top-3 right-3">
              <span className="px-2.5 py-1 rounded-lg bg-[var(--bg-panel)]/90 backdrop-blur-sm border border-[var(--border-subtle)] text-[var(--text-main)] text-[10.5px] font-mono shadow-subtle">
                {label2}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* ── Detailed Marked Places Inventory (Answers "What are the marked places") ── */}
      {safeClusters.length > 0 && (
        <div className="p-3 bg-[var(--bg-panel)] border-t border-[var(--border-subtle)]">
          <div className="flex items-center justify-between mb-2.5">
            <div className="flex items-center gap-1.5">
              <MapPin className="w-3.5 h-3.5 text-[#cc785c]" />
              <span className="font-semibold text-xs text-[var(--text-main)]">
                Marked Places & Delineated Sectors
              </span>
              <span className="text-[10.5px] font-mono text-[var(--text-dim)]">
                ({safeClusters.length} identified)
              </span>
            </div>
            <span className="text-[10px] font-mono text-[var(--text-muted)]">
              Hover card to highlight place on satellite image
            </span>
          </div>

          {/* Cards for each marked place */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {safeClusters.map((cluster, idx) => {
              const isHovered = hoveredClusterIndex === idx;
              const [cx, cy] = cluster.centroid || [0, 0];

              return (
                <div
                  key={cluster.zone || idx}
                  onMouseEnter={() => {
                    setHoveredClusterIndex(idx);
                    if (mode !== 'overlay') setMode('overlay');
                  }}
                  onMouseLeave={() => setHoveredClusterIndex(null)}
                  className={`p-2.5 rounded-xl border transition-all cursor-pointer ${
                    isHovered
                      ? 'bg-[#cc785c]/10 border-[#cc785c] shadow-sm'
                      : 'bg-[var(--bg-surface)] border-[var(--border-subtle)] hover:border-[#cc785c]/40'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-bold text-xs text-[var(--text-main)] flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-[#cc785c]" />
                      {cluster.zone}
                    </span>
                    {cluster.area_ha !== undefined && cluster.area_ha !== null && (
                      <span className="text-[10.5px] font-mono font-semibold px-1.5 py-0.2 rounded bg-[var(--bg-card)] border border-[var(--border-subtle)] text-[#cc785c]">
                        {cluster.area_ha} ha
                      </span>
                    )}
                  </div>

                  <p className="text-[11.5px] text-[var(--text-muted)] leading-snug mb-1.5">
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

      {/* Quantitative Summary Bar */}
      {(changedAreaHectares !== undefined && changedAreaHectares !== null) ||
      (changedAreaPercent !== undefined && changedAreaPercent !== null) ? (
        <div className="px-3.5 py-2 bg-[var(--bg-surface)] border-t border-[var(--border-subtle)] flex items-center justify-between text-xs">
          <div className="flex items-center gap-4 font-mono text-[11px]">
            {changedAreaHectares !== undefined && changedAreaHectares !== null && (
              <span>
                <strong className="text-[var(--text-main)] font-semibold">Changed Extent:</strong>{' '}
                <span className="text-[#cc785c]">{changedAreaHectares.toFixed(1)} ha</span>
              </span>
            )}
            {changedAreaPercent !== undefined && changedAreaPercent !== null && (
              <span>
                <strong className="text-[var(--text-main)] font-semibold">Scene Delta:</strong>{' '}
                <span className="text-[#cc785c]">{changedAreaPercent.toFixed(1)}%</span>
              </span>
            )}
          </div>
          <span className="text-[10.5px] font-mono text-[var(--text-dim)] hidden sm:inline">
            Calibrated against EO ground sampling distance
          </span>
        </div>
      ) : null}
    </div>
  );
};
