import React, { useId } from 'react';

interface SatQueryLogoProps {
  className?: string;
  size?: number;
  isLoading?: boolean;
  color?: string;
  showSquircle?: boolean;
}

/**
 * SatQuery AI Satellite Emblem
 * Precision satellite insignia matching the user's design:
 * - Tilted orbital satellite bus with solar array wings and telemetry antenna
 * - Dynamic planetary orbit crescent arc underneath
 * - Smooth reactive hover micro-glow and telemetry ping animations
 */
export const SatQueryLogo: React.FC<SatQueryLogoProps> = ({
  className = '',
  size = 20,
  isLoading = false,
  color,
  showSquircle = false,
}) => {
  const rawId = useId();
  const id = rawId.replace(/[^a-zA-Z0-9]/g, '');

  const strokeColor = color || '#CC785C';
  const glowId = `sq-sat-glow-${id}`;
  const gradId = `sq-sat-grad-${id}`;

  const content = (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="w-full h-full overflow-visible"
      shapeRendering="geometricPrecision"
      stroke={strokeColor}
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <defs>
        <linearGradient id={gradId} x1="3" y1="3" x2="21" y2="21" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor={color || "#E5987D"} />
          <stop offset="50%" stopColor={color || "#CC785C"} />
          <stop offset="100%" stopColor={color || "#A85338"} />
        </linearGradient>
        <filter id={glowId} x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="0" stdDeviation="1.5" floodColor={color || "#CC785C"} floodOpacity="0.4" />
        </filter>
      </defs>

      {/* ── Active Antenna Ping Beacon (When Loading / Streaming) ── */}
      {isLoading && (
        <>
          <circle
            cx="19"
            cy="5"
            r="3"
            stroke={strokeColor}
            strokeWidth="0.75"
            fill="none"
            opacity="0.7"
            className="animate-ping origin-[19px_5px]"
          />
          <circle
            cx="19"
            cy="5"
            r="1"
            fill={strokeColor}
            stroke="none"
          />
        </>
      )}

      {/* ── Upper-Left Solar Wing Panel ── */}
      <path
        d="m13.5 6.5-3.148-3.148a1.205 1.205 0 0 0-1.704 0L6.352 5.648a1.205 1.205 0 0 0 0 1.704L9.5 10.5"
        stroke={`url(#${gradId})`}
        filter={`url(#${glowId})`}
      />

      {/* ── Telemetry Antenna Mast ── */}
      <path
        d="M16.5 7.5 19 5"
        stroke={strokeColor}
      />

      {/* ── Lower-Right Solar Wing Panel ── */}
      <path
        d="m17.5 10.5 3.148 3.148a1.205 1.205 0 0 1 0 1.704l-2.296 2.296a1.205 1.205 0 0 1-1.704 0L13.5 14.5"
        stroke={`url(#${gradId})`}
        filter={`url(#${glowId})`}
      />

      {/* ── Curved Planetary / Orbital Arc Underneath ── */}
      <path
        d="M9 21a6 6 0 0 0-6-6"
        stroke={strokeColor}
        className={isLoading ? 'animate-pulse' : ''}
      />

      {/* ── Central Satellite Bus / Avionics Body ── */}
      <path
        d="M9.352 10.648a1.205 1.205 0 0 0 0 1.704l2.296 2.296a1.205 1.205 0 0 0 1.704 0l4.296-4.296a1.205 1.205 0 0 0 0-1.704l-2.296-2.296a1.205 1.205 0 0 0-1.704 0z"
        stroke={`url(#${gradId})`}
        filter={`url(#${glowId})`}
      />
    </svg>
  );

  if (showSquircle) {
    return (
      <div
        className={`inline-flex items-center justify-center shrink-0 select-none rounded-xl bg-black/40 border border-[#CC785C]/30 p-1.5 transition-all duration-300 hover:border-[#CC785C]/60 hover:shadow-[0_0_12px_rgba(204,120,92,0.25)] ${
          isLoading ? 'animate-satquery-breathe' : 'hover:scale-105'
        } ${className}`}
        style={{ width: size, height: size }}
        title="SatQuery AI"
      >
        {content}
      </div>
    );
  }

  return (
    <div
      className={`inline-flex items-center justify-center shrink-0 select-none transition-all duration-300 ${
        isLoading
          ? 'animate-satquery-breathe'
          : 'hover:scale-110 hover:drop-shadow-[0_0_6px_rgba(204,120,92,0.5)] cursor-pointer'
      } ${className}`}
      style={{
        width: size,
        height: size,
        transformOrigin: 'center center',
      }}
      title="SatQuery AI"
    >
      {content}
    </div>
  );
};
