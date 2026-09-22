import React, { useState } from 'react';
import {
  BarChart3,
  Compass,
  Activity,
  Layers,
  ChevronDown,
  ChevronUp,
  Info,
  ShieldCheck,
} from 'lucide-react';
import type { ChartDataPayload } from '../../types/api';

interface DynamicTelemetryChartProps {
  data: ChartDataPayload;
  className?: string;
  defaultExpanded?: boolean;
}

export const DynamicTelemetryChart: React.FC<DynamicTelemetryChartProps> = ({
  data,
  className = '',
  defaultExpanded = true,
}) => {
  const [activeTab, setActiveTab] = useState<'landcover' | 'quadrants' | 'histogram' | 'sensor'>('landcover');
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  const landCover = data.land_cover_chart || [];
  const quadrants = data.quadrant_chart || [];
  const histogram = data.spectral_histogram;
  const sensor = data.sensor_fidelity;

  const hasData = landCover.length > 0 || quadrants.length > 0 || Boolean(histogram);
  if (!hasData) return null;

  return (
    <div
      className={`rounded-2xl border border-[var(--border-subtle)] bg-[var(--bg-card)]/85 backdrop-blur-md overflow-hidden shadow-subtle transition-all duration-200 ${className}`}
    >
      {/* Header bar */}
      <div className="px-4 py-3 border-b border-[var(--border-subtle)] flex items-center justify-between bg-[var(--bg-surface)]/50">
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 rounded-lg bg-[#0EA5E9]/10 text-[#0EA5E9] border border-[#0EA5E9]/25">
            <BarChart3 className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-xs text-[var(--text-main)] tracking-tight">
                Dynamic Geospatial Telemetry
              </span>
              {sensor?.is_proxy && (
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-500 border border-amber-500/20 font-medium">
                  RGB Proxy Calibrated
                </span>
              )}
              {sensor && !sensor.is_proxy && (
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium">
                  Multi-Spectral Native
                </span>
              )}
            </div>
            <span className="text-[10.5px] text-[var(--text-muted)] block">
              Quantitative pixel analytics & spectral distributions
            </span>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1.5 rounded-lg text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-[var(--bg-app)] transition-colors cursor-pointer"
            title={isExpanded ? 'Collapse chart' : 'Expand chart'}
          >
            {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {isExpanded && (
        <div className="p-4 space-y-4">
          {/* Navigation Pill Tabs */}
          <div className="flex items-center gap-1.5 p-1 rounded-xl bg-[var(--bg-surface)] border border-[var(--border-subtle)] text-xs font-medium">
            {landCover.length > 0 && (
              <button
                type="button"
                onClick={() => setActiveTab('landcover')}
                className={`flex-1 py-1.5 px-2.5 rounded-lg transition-all cursor-pointer flex items-center justify-center gap-1.5 ${
                  activeTab === 'landcover'
                    ? 'bg-[var(--bg-card)] text-[#0EA5E9] font-semibold shadow-subtle'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                <span>Land Cover</span>
              </button>
            )}

            {quadrants.length > 0 && (
              <button
                type="button"
                onClick={() => setActiveTab('quadrants')}
                className={`flex-1 py-1.5 px-2.5 rounded-lg transition-all cursor-pointer flex items-center justify-center gap-1.5 ${
                  activeTab === 'quadrants'
                    ? 'bg-[var(--bg-card)] text-[#0EA5E9] font-semibold shadow-subtle'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
                }`}
              >
                <Compass className="w-3.5 h-3.5" />
                <span>Quadrants</span>
              </button>
            )}

            {histogram && (
              <button
                type="button"
                onClick={() => setActiveTab('histogram')}
                className={`flex-1 py-1.5 px-2.5 rounded-lg transition-all cursor-pointer flex items-center justify-center gap-1.5 ${
                  activeTab === 'histogram'
                    ? 'bg-[var(--bg-card)] text-[#0EA5E9] font-semibold shadow-subtle'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
                }`}
              >
                <Activity className="w-3.5 h-3.5" />
                <span>Histogram</span>
              </button>
            )}

            {sensor && (
              <button
                type="button"
                onClick={() => setActiveTab('sensor')}
                className={`flex-1 py-1.5 px-2.5 rounded-lg transition-all cursor-pointer flex items-center justify-center gap-1.5 ${
                  activeTab === 'sensor'
                    ? 'bg-[var(--bg-card)] text-[#0EA5E9] font-semibold shadow-subtle'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-main)]'
                }`}
              >
                <Info className="w-3.5 h-3.5" />
                <span>Sensor</span>
              </button>
            )}
          </div>

          {/* VIEW 1: LAND COVER DISTRIBUTION */}
          {activeTab === 'landcover' && landCover.length > 0 && (
            <div className="space-y-3.5 animate-fadeIn">
              {/* Stacked Progress Bar */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-[11px] font-mono text-[var(--text-muted)]">
                  <span>Partition Distribution</span>
                  <span>100% Total Scene Footprint</span>
                </div>
                <div className="h-3 w-full rounded-full overflow-hidden flex bg-[var(--bg-surface)] p-0.5 border border-[var(--border-subtle)]">
                  {landCover.map((item) => (
                    <div
                      key={item.key}
                      style={{
                        width: `${Math.max(item.pct, 0)}%`,
                        backgroundColor: item.color,
                      }}
                      className="h-full first:rounded-l-full last:rounded-r-full transition-all duration-300"
                      title={`${item.label}: ${item.pct}% (${item.area_ha} ha)`}
                    />
                  ))}
                </div>
              </div>

              {/* Category Breakdown Chips */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 pt-1">
                {landCover.map((item) => (
                  <div
                    key={item.key}
                    className="p-3 rounded-xl bg-[var(--bg-surface)]/70 border border-[var(--border-subtle)] space-y-1 hover:border-[var(--border-hover)] transition-colors"
                  >
                    <div className="flex items-center gap-1.5">
                      <span
                        className="w-2.5 h-2.5 rounded-full inline-block shrink-0"
                        style={{ backgroundColor: item.color }}
                      />
                      <span className="text-[11px] font-medium text-[var(--text-muted)] truncate">
                        {item.label}
                      </span>
                    </div>
                    <div className="flex items-baseline justify-between pt-0.5">
                      <span className="text-base font-bold text-[var(--text-main)] font-mono">
                        {item.pct}%
                      </span>
                      <span className="text-[10.5px] font-mono text-[var(--text-dim)]">
                        {item.area_ha} ha
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* VIEW 2: SPATIAL QUADRANT DISTRIBUTION */}
          {activeTab === 'quadrants' && quadrants.length > 0 && (
            <div className="space-y-3 animate-fadeIn">
              <div className="text-[11px] font-mono text-[var(--text-muted)] flex justify-between">
                <span>Quadrant Cross-Section</span>
                <span>Canopy vs Hydrology vs Built (%)</span>
              </div>

              <div className="space-y-2.5">
                {quadrants.map((q) => (
                  <div
                    key={q.quadrant}
                    className="p-3 rounded-xl bg-[var(--bg-surface)]/70 border border-[var(--border-subtle)] space-y-2"
                  >
                    <div className="flex items-center justify-between text-xs font-medium text-[var(--text-main)]">
                      <span className="font-semibold">{q.quadrant} Sector</span>
                      <div className="flex items-center gap-3 text-[10.5px] font-mono">
                        <span className="text-emerald-400">🌿 {q.veg_pct}%</span>
                        <span className="text-sky-400">💧 {q.water_pct}%</span>
                        <span className="text-amber-400">🏢 {q.built_pct}%</span>
                      </div>
                    </div>

                    {/* Multi-tier horizontal bars */}
                    <div className="space-y-1">
                      <div className="h-1.5 w-full bg-[var(--bg-app)] rounded-full overflow-hidden flex">
                        <div
                          style={{ width: `${Math.min(q.veg_pct, 100)}%` }}
                          className="h-full bg-emerald-500 rounded-full"
                        />
                      </div>
                      <div className="h-1.5 w-full bg-[var(--bg-app)] rounded-full overflow-hidden flex">
                        <div
                          style={{ width: `${Math.min(q.water_pct, 100)}%` }}
                          className="h-full bg-sky-500 rounded-full"
                        />
                      </div>
                      <div className="h-1.5 w-full bg-[var(--bg-app)] rounded-full overflow-hidden flex">
                        <div
                          style={{ width: `${Math.min(q.built_pct, 100)}%` }}
                          className="h-full bg-amber-500 rounded-full"
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* VIEW 3: SPECTRAL REFLECTANCE HISTOGRAM */}
          {activeTab === 'histogram' && histogram && (
            <div className="space-y-3 animate-fadeIn">
              <div className="flex items-center justify-between text-[11px] font-mono text-[var(--text-muted)]">
                <span>{histogram.metric_name} Distribution</span>
                <span>10 Radiometric Bins</span>
              </div>

              {/* Vector SVG Histogram */}
              <div className="p-3 rounded-xl bg-[var(--bg-surface)]/70 border border-[var(--border-subtle)] space-y-2">
                <div className="h-32 w-full flex items-end justify-between gap-1 pt-2">
                  {histogram.values.map((val, idx) => {
                    const maxVal = Math.max(...histogram.values, 1.0);
                    const heightPct = Math.round((val / maxVal) * 100);
                    return (
                      <div
                        key={histogram.bins[idx]}
                        className="flex-1 flex flex-col items-center gap-1 group h-full justify-end"
                        title={`Range: ${histogram.bins[idx]}\nFrequency: ${val}% of pixels`}
                      >
                        <span className="text-[9px] font-mono text-[var(--text-dim)] opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                          {val}%
                        </span>
                        <div
                          style={{ height: `${Math.max(heightPct, 4)}%` }}
                          className="w-full rounded-t bg-gradient-to-t from-[#0EA5E9]/50 to-[#0EA5E9] hover:from-sky-400 hover:to-emerald-400 transition-all duration-200"
                        />
                      </div>
                    );
                  })}
                </div>

                <div className="flex justify-between text-[9.5px] font-mono text-[var(--text-dim)] border-t border-[var(--border-subtle)] pt-1.5">
                  <span>-0.5 (Non-vegetated / Water)</span>
                  <span>0.0</span>
                  <span>+0.8 (Dense Biomass)</span>
                </div>
              </div>

              <div className="p-2.5 rounded-xl bg-[var(--bg-surface)] border border-[var(--border-subtle)] text-[11px] text-[var(--text-muted)] flex items-start gap-2">
                <ShieldCheck className="w-4 h-4 text-[#0EA5E9] shrink-0 mt-0.5" />
                <div>
                  <span className="font-semibold text-[var(--text-main)] block">
                    Calibration Protocol:
                  </span>
                  <p className="text-[10.5px] text-[var(--text-dim)]">
                    {histogram.calibration_method}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* VIEW 4: SENSOR & CALIBRATION SPECIFICATIONS */}
          {activeTab === 'sensor' && sensor && (
            <div className="space-y-3 animate-fadeIn text-xs font-mono">
              <div className="p-3 rounded-xl bg-[var(--bg-surface)]/70 border border-[var(--border-subtle)] divide-y divide-[var(--border-subtle)]">
                <div className="py-1.5 flex justify-between">
                  <span className="text-[var(--text-muted)]">Channels Detected</span>
                  <span className="text-[var(--text-main)] font-semibold">{sensor.band_count} Bands</span>
                </div>
                <div className="py-1.5 flex justify-between">
                  <span className="text-[var(--text-muted)]">Spatial Resolution</span>
                  <span className="text-[var(--text-main)]">{sensor.resolution_m} m Ground Sample Distance</span>
                </div>
                <div className="py-1.5 flex justify-between">
                  <span className="text-[var(--text-muted)]">Total Scene Area</span>
                  <span className="text-[var(--text-main)]">{sensor.total_area_ha} Hectares</span>
                </div>
                <div className="py-1.5 flex justify-between">
                  <span className="text-[var(--text-muted)]">Radiometric Pipeline</span>
                  <span className="text-[var(--text-main)] truncate max-w-[220px]">
                    {sensor.calibration_method}
                  </span>
                </div>
              </div>

              <div className="p-3 rounded-xl bg-[var(--bg-surface)] border border-[var(--border-subtle)] text-[11px] font-sans text-[var(--text-muted)] space-y-1">
                <div className="flex items-center gap-1.5 font-semibold text-[var(--text-main)]">
                  <Info className="w-3.5 h-3.5 text-[#0EA5E9]" />
                  <span>Scientific Rigor & Hallucination Prevention</span>
                </div>
                <p className="text-[10.5px] text-[var(--text-dim)] leading-relaxed">
                  Unlike large cloud LLMs that guess imagery contents without raw pixel access,
                  SatQuery executes genuine array mathematical calculations directly across raster pixels
                  with complete auditability.
                </p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
