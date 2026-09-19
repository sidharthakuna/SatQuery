import React, { useState, useRef } from 'react';
import {
  Crosshair,
  Maximize2,
  Minimize2,
  Eye,
  EyeOff,
  Sliders,
  Sparkles,
  Tag,
  Square,
  Scan,
} from 'lucide-react';
import { Badge } from '../ui/Badge';
import { SatQueryAPI } from '../../services/api';

interface GroundingViewerProps {
  imageUrl: string;
  boxes?: [number, number, number, number][] | null;
  confidence?: number;
  label?: string;
  imgWidth?: number;
  imgHeight?: number;
}

type BoxRenderStyle = 'reticle' | 'outline' | 'shaded';
type TagVisibility = 'hover' | 'always' | 'none';

interface ColorOption {
  id: string;
  name: string;
  stroke: string;
  fill: string;
  glow: string;
}

const COLOR_PALETTES: Record<string, ColorOption> = {
  terracotta: {
    id: 'terracotta',
    name: 'Terracotta',
    stroke: '#cc785c',
    fill: 'rgba(204, 120, 92, 0.14)',
    glow: 'rgba(204, 120, 92, 0.45)',
  },
  amber: {
    id: 'amber',
    name: 'Amber',
    stroke: '#d97706',
    fill: 'rgba(217, 119, 6, 0.14)',
    glow: 'rgba(217, 119, 6, 0.45)',
  },
  emerald: {
    id: 'emerald',
    name: 'Emerald',
    stroke: '#059669',
    fill: 'rgba(5, 150, 105, 0.14)',
    glow: 'rgba(5, 150, 105, 0.45)',
  },
  white: {
    id: 'white',
    name: 'Monochrome',
    stroke: '#e4e4e7',
    fill: 'rgba(228, 228, 231, 0.12)',
    glow: 'rgba(255, 255, 255, 0.35)',
  },
};

export const GroundingViewer: React.FC<GroundingViewerProps> = ({
  imageUrl,
  boxes = [],
  confidence,
  label = 'Target',
  imgWidth = 512,
  imgHeight = 512,
}) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [showBoxes, setShowBoxes] = useState(true);
  const [isPeeking, setIsPeeking] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [renderStyle, setRenderStyle] = useState<BoxRenderStyle>('reticle');
  const [tagVisibility, setTagVisibility] = useState<TagVisibility>('hover');
  const [activeColor, setActiveColor] = useState<string>('terracotta');
  const [fillOpacity, setFillOpacity] = useState<number>(0); // 0% default: zero obstruction!
  const [strokeOpacity, setStrokeOpacity] = useState<number>(0.92);
  const [showSettings, setShowSettings] = useState(false);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const safeBoxes = boxes || [];
  const palette = COLOR_PALETTES[activeColor] || COLOR_PALETTES.terracotta;

  // Compute L-bracket paths for unobtrusive corner reticles
  const getCornerReticlePath = (x1: number, y1: number, x2: number, y2: number) => {
    const w = x2 - x1;
    const h = y2 - y1;
    const arm = Math.max(4, Math.min(w * 0.28, h * 0.28, 16));

    return [
      // Top-Left
      `M ${x1} ${y1 + arm} L ${x1} ${y1} L ${x1 + arm} ${y1}`,
      // Top-Right
      `M ${x2 - arm} ${y1} L ${x2} ${y1} L ${x2} ${y1 + arm}`,
      // Bottom-Right
      `M ${x2} ${y2 - arm} L ${x2} ${y2} L ${x2 - arm} ${y2}`,
      // Bottom-Left
      `M ${x1 + arm} ${y2} L ${x1} ${y2} L ${x1} ${y2 - arm}`,
    ].join(' ');
  };

  const isLayerVisible = showBoxes && !isPeeking;

  return (
    <div
      ref={containerRef}
      className={`relative rounded-xl overflow-hidden border border-[var(--border-subtle)] bg-[var(--bg-card)] my-3 select-none transition-all shadow-subtle ${
        isFullscreen ? 'fixed inset-4 z-50 flex flex-col bg-black' : 'w-full'
      }`}
    >
      {/* ── Top Header Toolbar ─────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between px-3.5 py-2 bg-[var(--bg-panel)] border-b border-[var(--border-subtle)] text-xs gap-2">
        <div className="flex items-center gap-2">
          <Crosshair className="w-3.5 h-3.5 text-[#cc785c]" />
          <span className="font-medium text-[var(--text-main)] font-mono text-[11px]">
            Grounding: {safeBoxes.length} Object{safeBoxes.length === 1 ? '' : 's'}
          </span>
          {confidence !== undefined && (
            <Badge variant="success">{(confidence * 100).toFixed(0)}% Match</Badge>
          )}
        </div>

        {/* Action Controls Ribbon */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {/* Quick Style Switcher (Reticle / Outline / Shaded) */}
          <div className="flex items-center p-0.5 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)]">
            <button
              type="button"
              onClick={() => setRenderStyle('reticle')}
              className={`px-2 py-0.5 rounded text-[10px] font-mono flex items-center gap-1 transition-all cursor-pointer ${
                renderStyle === 'reticle'
                  ? 'bg-[var(--bg-card)] text-[#cc785c] border border-[var(--border-subtle)] shadow-subtle font-semibold'
                  : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
              }`}
              title="Corner Reticles: Unobtrusive L-brackets that leave the satellite imagery 100% visible"
            >
              <Scan className="w-3 h-3" />
              <span>Reticles</span>
            </button>
            <button
              type="button"
              onClick={() => setRenderStyle('outline')}
              className={`px-2 py-0.5 rounded text-[10px] font-mono flex items-center gap-1 transition-all cursor-pointer ${
                renderStyle === 'outline'
                  ? 'bg-[var(--bg-card)] text-[var(--text-main)] border border-[var(--border-subtle)] shadow-subtle font-semibold'
                  : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
              }`}
              title="Thin Outline: Clean 1px border without interior obstruction"
            >
              <Square className="w-3 h-3" />
              <span>Outline</span>
            </button>
            <button
              type="button"
              onClick={() => setRenderStyle('shaded')}
              className={`px-2 py-0.5 rounded text-[10px] font-mono flex items-center gap-1 transition-all cursor-pointer ${
                renderStyle === 'shaded'
                  ? 'bg-[var(--bg-card)] text-[var(--text-main)] border border-[var(--border-subtle)] shadow-subtle font-semibold'
                  : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
              }`}
              title="Tinted Box: Semi-transparent tinted highlight"
            >
              <Sparkles className="w-3 h-3" />
              <span>Tinted</span>
            </button>
          </div>

          {/* Tag Visibility Toggle */}
          <button
            type="button"
            onClick={() => {
              if (tagVisibility === 'hover') setTagVisibility('always');
              else if (tagVisibility === 'always') setTagVisibility('none');
              else setTagVisibility('hover');
            }}
            className="px-2 py-0.5 rounded text-[10px] font-mono border border-[var(--border-subtle)] bg-[var(--bg-surface)] text-[var(--text-muted)] hover:text-[var(--text-main)] flex items-center gap-1 cursor-pointer transition-colors"
            title={`Tags: ${tagVisibility === 'hover' ? 'Shown on hover only (unobtrusive)' : tagVisibility === 'always' ? 'Always visible' : 'Hidden completely'}`}
          >
            <Tag className="w-2.5 h-2.5 text-[var(--text-dim)]" />
            <span>Tags: {tagVisibility === 'hover' ? 'Hover' : tagVisibility === 'always' ? 'On' : 'Off'}</span>
          </button>

          {/* Quick Peek (Hold or Click to view pure satellite imagery) */}
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
            title="Hold to Peek: Temporarily hides all boxes to inspect the raw satellite pixels"
          >
            <Eye className="w-2.5 h-2.5" />
            <span>{isPeeking ? 'Peeking...' : 'Hold Peek'}</span>
          </button>

          {/* Opacity & Filter Settings Popover Toggle */}
          <button
            type="button"
            onClick={() => setShowSettings(!showSettings)}
            className={`p-1 rounded text-[var(--text-muted)] hover:text-[var(--text-main)] border transition-colors cursor-pointer ${
              showSettings
                ? 'bg-[#cc785c]/15 border-[#cc785c]/40 text-[#cc785c]'
                : 'bg-[var(--bg-surface)] border-[var(--border-subtle)]'
            }`}
            title="Adjust box opacity and colors"
          >
            <Sliders className="w-3.5 h-3.5" />
          </button>

          {/* Toggle Boxes On/Off */}
          <button
            type="button"
            onClick={() => setShowBoxes(!showBoxes)}
            className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-colors flex items-center gap-1 cursor-pointer ${
              showBoxes
                ? 'bg-[var(--bg-surface)] text-[var(--text-main)] border-[var(--border-subtle)]'
                : 'bg-[var(--bg-card)] text-[var(--text-dim)] border-[var(--border-subtle)] opacity-60'
            }`}
          >
            {showBoxes ? <Eye className="w-2.5 h-2.5" /> : <EyeOff className="w-2.5 h-2.5" />}
            <span>{showBoxes ? 'Visible' : 'Hidden'}</span>
          </button>

          {/* Fullscreen Toggle */}
          <button
            type="button"
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-1 text-[var(--text-muted)] hover:text-[var(--text-main)] rounded transition-colors cursor-pointer"
            title={isFullscreen ? 'Exit Fullscreen' : 'Toggle Fullscreen'}
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* ── Slide-Down Transparency & Color Tuning Panel ───────────────── */}
      {showSettings && (
        <div className="px-3.5 py-2.5 bg-[var(--bg-panel)] border-b border-[var(--border-subtle)] flex flex-wrap items-center justify-between gap-3 text-xs animate-fadeIn">
          {/* Fill Opacity Slider */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono text-[var(--text-muted)]">Fill Opacity:</span>
            <input
              type="range"
              min={0}
              max={0.4}
              step={0.02}
              value={fillOpacity}
              onChange={(e) => setFillOpacity(parseFloat(e.target.value))}
              className="w-20 h-1 bg-[var(--border-subtle)] rounded-lg appearance-none cursor-pointer accent-[#cc785c]"
              title="Box Interior Fill Opacity (0% leaves satellite imagery 100% crystal clear)"
            />
            <span className="text-[10px] font-mono text-[var(--text-main)] w-8">
              {Math.round(fillOpacity * 100)}%
            </span>
          </div>

          {/* Stroke Opacity Slider */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono text-[var(--text-muted)]">Line Visibility:</span>
            <input
              type="range"
              min={0.2}
              max={1}
              step={0.05}
              value={strokeOpacity}
              onChange={(e) => setStrokeOpacity(parseFloat(e.target.value))}
              className="w-20 h-1 bg-[var(--border-subtle)] rounded-lg appearance-none cursor-pointer accent-[#cc785c]"
              title="Box Border Opacity"
            />
            <span className="text-[10px] font-mono text-[var(--text-main)] w-8">
              {Math.round(strokeOpacity * 100)}%
            </span>
          </div>

          {/* Color Palette Selector */}
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] font-mono text-[var(--text-muted)]">Color:</span>
            {Object.entries(COLOR_PALETTES).map(([key, opt]) => (
              <button
                key={key}
                type="button"
                onClick={() => setActiveColor(key)}
                className={`w-4 h-4 rounded-full border transition-transform cursor-pointer ${
                  activeColor === key ? 'scale-125 ring-2 ring-[#cc785c]/60' : 'opacity-70 hover:opacity-100'
                }`}
                style={{ backgroundColor: opt.stroke, borderColor: 'var(--border-subtle)' }}
                title={`${opt.name} Tactical Overlay`}
              />
            ))}
          </div>
        </div>
      )}

      {/* ── Main Satellite Raster & Clean SVG Overlay Surface ───────────── */}
      <div className="relative w-full aspect-square max-h-[480px] flex items-center justify-center bg-black overflow-hidden group">
        <img
          src={SatQueryAPI.getRasterPreviewUrl(imageUrl || 'cartosat_t1.tif')}
          alt="Satellite Scene"
          className="w-full h-full object-contain"
          onError={(e) => {
            (e.currentTarget as HTMLImageElement).src = SatQueryAPI.getRasterPreviewUrl('cartosat_t1.tif');
          }}
        />

        {isLayerVisible && (
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            viewBox={`0 0 ${imgWidth} ${imgHeight}`}
            preserveAspectRatio="xMidYMid meet"
          >
            <defs>
              <filter id="subtleGlow" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="0" stdDeviation="1.5" floodColor={palette.stroke} floodOpacity="0.6" />
              </filter>
            </defs>

            {safeBoxes.map((box, i) => {
              const [x1, y1, x2, y2] = box;
              const w = Math.max(x2 - x1, 1);
              const h = Math.max(y2 - y1, 1);
              const isHovered = hoveredIndex === i;

              // Opacity logic:
              // - In 'reticle' or 'outline' mode: default fill is transparent (0%) so satellite image is completely unobstructed!
              // - In 'shaded' mode: gentle translucent fill
              const currentFillOpacity =
                renderStyle === 'shaded'
                  ? Math.max(fillOpacity, isHovered ? 0.25 : 0.08)
                  : isHovered
                  ? Math.max(fillOpacity, 0.12)
                  : fillOpacity;

              const boxFill =
                currentFillOpacity > 0
                  ? palette.fill.replace('0.14', currentFillOpacity.toFixed(2))
                  : 'none';

              const boxStroke = isHovered ? '#ffffff' : palette.stroke;
              const boxStrokeWidth = isHovered ? 1.8 : 1.25;
              const boxStrokeOpacity = isHovered ? 1 : strokeOpacity;

              // Tag display logic: only show if 'always' or on hover
              const shouldShowTag =
                tagVisibility === 'always' || (tagVisibility === 'hover' && isHovered);

              return (
                <g
                  key={i}
                  className="pointer-events-auto cursor-pointer"
                  onMouseEnter={() => setHoveredIndex(i)}
                  onMouseLeave={() => setHoveredIndex(null)}
                >
                  {/* Invisible hit-box area to facilitate easy hovering */}
                  <rect
                    x={x1}
                    y={y1}
                    width={w}
                    height={h}
                    fill={boxFill}
                    className="transition-colors duration-100"
                  />

                  {/* 1. Corner Reticles Style (Unobtrusive & Clean) */}
                  {renderStyle === 'reticle' && (
                    <path
                      d={getCornerReticlePath(x1, y1, x2, y2)}
                      fill="none"
                      stroke={boxStroke}
                      strokeWidth={boxStrokeWidth}
                      strokeOpacity={boxStrokeOpacity}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      vectorEffect="non-scaling-stroke"
                      filter={isHovered ? 'url(#subtleGlow)' : undefined}
                    />
                  )}

                  {/* 2. Precision Outline or Shaded Style */}
                  {renderStyle !== 'reticle' && (
                    <rect
                      x={x1}
                      y={y1}
                      width={w}
                      height={h}
                      fill="none"
                      stroke={boxStroke}
                      strokeWidth={boxStrokeWidth}
                      strokeOpacity={boxStrokeOpacity}
                      vectorEffect="non-scaling-stroke"
                      filter={isHovered ? 'url(#subtleGlow)' : undefined}
                    />
                  )}

                  {/* 3. Small Center Crosshair on Hover to pinpoint target */}
                  {isHovered && (
                    <g opacity={0.8}>
                      <line
                        x1={x1 + w / 2 - 4}
                        y1={y1 + h / 2}
                        x2={x1 + w / 2 + 4}
                        y2={y1 + h / 2}
                        stroke="#ffffff"
                        strokeWidth="1"
                        vectorEffect="non-scaling-stroke"
                      />
                      <line
                        x1={x1 + w / 2}
                        y1={y1 + h / 2 - 4}
                        x2={x1 + w / 2}
                        y2={y1 + h / 2 + 4}
                        stroke="#ffffff"
                        strokeWidth="1"
                        vectorEffect="non-scaling-stroke"
                      />
                    </g>
                  )}

                  {/* 4. Sleek Non-Obtrusive Tag */}
                  {shouldShowTag && (
                    <g transform={`translate(${x1}, ${Math.max(y1 - 16, 0)})`}>
                      <rect
                        x="0"
                        y="0"
                        width={Math.max(w * 0.8, 56)}
                        height="13"
                        rx="2"
                        fill="rgba(10, 10, 12, 0.85)"
                        stroke={boxStroke}
                        strokeOpacity={0.6}
                        strokeWidth="0.75"
                      />
                      <text
                        x="4"
                        y="9.5"
                        fill="#f4f4f5"
                        fontSize="8.5"
                        fontFamily="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"
                        fontWeight="600"
                        letterSpacing="0.02em"
                      >
                        {label} #{i + 1}
                      </text>
                    </g>
                  )}
                </g>
              );
            })}
          </svg>
        )}
      </div>

      {/* ── Marked Targets & Places Inventory ─────────────────────────── */}
      {safeBoxes.length > 0 && (
        <div className="p-3 bg-[var(--bg-panel)] border-t border-[var(--border-subtle)]">
          <div className="flex items-center justify-between mb-2">
            <span className="font-semibold text-xs text-[var(--text-main)] flex items-center gap-1.5">
              <Crosshair className="w-3 h-3 text-[#cc785c]" />
              Marked Places & Detected Objects ({safeBoxes.length})
            </span>
            <span className="text-[10px] font-mono text-[var(--text-dim)]">
              Hover target card to highlight reticle on satellite image
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
            {safeBoxes.map((box, idx) => {
              const isHovered = hoveredIndex === idx;
              const [x1, y1, x2, y2] = box;
              const bw = Math.round(x2 - x1);
              const bh = Math.round(y2 - y1);
              const cx = Math.round((x1 + x2) / 2);
              const cy = Math.round((y1 + y2) / 2);

              return (
                <div
                  key={idx}
                  onMouseEnter={() => setHoveredIndex(idx)}
                  onMouseLeave={() => setHoveredIndex(null)}
                  className={`p-2 rounded-xl border transition-all cursor-pointer ${
                    isHovered
                      ? 'bg-[#cc785c]/10 border-[#cc785c] shadow-sm'
                      : 'bg-[var(--bg-surface)] border-[var(--border-subtle)] hover:border-[#cc785c]/40'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-bold text-xs text-[var(--text-main)] flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-[#cc785c]" />
                      {label} #{idx + 1}
                    </span>
                    <span className="text-[9.5px] font-mono px-1 py-0.2 rounded bg-[var(--bg-card)] border border-[var(--border-subtle)] text-[var(--text-muted)]">
                      {bw}×{bh}px
                    </span>
                  </div>

                  <div className="text-[10px] font-mono text-[var(--text-dim)] flex items-center justify-between">
                    <span>Centroid: ({cx}, {cy})</span>
                    <span>[{Math.round(x1)}, {Math.round(y1)}]</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Dynamic Target Telemetry Footer ─────────────────────────────── */}
      <div className="px-3.5 py-1.5 bg-[var(--bg-panel)] border-t border-[var(--border-subtle)] text-[10px] font-mono text-[var(--text-muted)] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-[#cc785c] animate-pulse" />
          <span>
            {hoveredIndex !== null && safeBoxes[hoveredIndex]
              ? `Target #${hoveredIndex + 1} Selected: [${safeBoxes[hoveredIndex].map((n) => Math.round(n)).join(', ')}] • Size: ${Math.round(safeBoxes[hoveredIndex][2] - safeBoxes[hoveredIndex][0])}×${Math.round(safeBoxes[hoveredIndex][3] - safeBoxes[hoveredIndex][1])}px`
              : `${safeBoxes.length} targets detected • Mode: ${renderStyle.toUpperCase()} (unobstructed)`}
          </span>
        </div>
        <div className="hidden sm:flex items-center gap-2 text-[var(--text-dim)]">
          <span>Tip: Click &lsquo;Hold Peek&rsquo; or toggle Reticles for pure satellite view</span>
        </div>
      </div>
    </div>
  );
};
