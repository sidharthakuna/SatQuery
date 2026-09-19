# SatQuery AI — Design System Specification
**Product:** SatQuery AI (ProjectS) — Vision-Language Assistant for Remote Sensing  
**Context:** ISRO / Space Applications Centre (SAC) — SIH Problem Statement ID 26167  
**Design Paradigm:** Aerospace-grade Defense Intelligence • Dark Glassmorphic • Precision Telemetry  
**Version:** `1.0.0`

---

## 📑 Table of Contents
1. [Design Philosophy & Visual Language](#1-design-philosophy--visual-language)
2. [Three-Layer Token Architecture](#2-three-layer-token-architecture)
3. [Primitive Tokens (Layer 1)](#3-primitive-tokens-layer-1)
4. [Semantic Tokens (Layer 2)](#4-semantic-tokens-layer-2)
5. [Component Tokens & Specs (Layer 3)](#5-component-tokens--specs-layer-3)
6. [Typography Scale & Font Pairings](#6-typography-scale--font-pairings)
7. [Iconography & Sensor Status System](#7-iconography--sensor-status-system)
8. [Complete CSS Variables (`tokens.css`)](#8-complete-css-variables-tokenscss)
9. [Tailwind CSS Configuration (`tailwind.config.ts`)](#9-tailwind-css-configuration-tailwindconfigts)

---

# 1. Design Philosophy & Visual Language

SatQuery AI is built for mission operators, geospatial researchers, and defense analysts. The visual language balances **high information density** with **low visual fatigue** during prolonged analysis of high-contrast satellite rasters:

* **Dark Void Base (`#020617` / `#0B0F19`):** Maximizes dynamic range when viewing 12-bit/16-bit high-contrast optical rasters and radar backscatter.
* **Sensor-Coded Spectral Accents:**
  * 🛰️ **Cyan Neon (`#00F2FE` / `#22D3EE`):** AI agent routing, active tools, and cross-attention fusion.
  * 🌿 **Optical Emerald (`#10B981` / `#34D399`):** Optical reflectance, vegetation vigor, and nominal validation.
  * 📡 **SAR Microwave Amber (`#F59E0B` / `#FBBF24`):** Radar backscatter ($\sigma^0$), cloud penetration, and microwave telemetry.
  * 🔮 **Change Detection Violet (`#A855F7` / `#C084FC`):** Bi-temporal shift, difference masks, and urban growth.
  * 🚨 **Critical Red (`#EF4444` / `#F87171`):** CRS mismatch, validation errors, and high-risk anomalies.
* **Telemetry Monospacing:** Sensor coordinates (Lat/Lon), EPSG codes, latencies, and bounding box vectors use `JetBrains Mono` or `Fira Code`.

---

# 2. Three-Layer Token Architecture

```
Layer 1: Primitive Tokens  ── (Raw color hex, font sizes, pixel spacing)
            │
            ▼
Layer 2: Semantic Tokens   ── (Purpose aliases: surface-primary, sensor-optical, text-muted)
            │
            ▼
Layer 3: Component Tokens  ── (slider-divider-bg, bbox-border, audit-step-badge)
```

---

# 3. Primitive Tokens (Layer 1)

### 3.1 Color Palette Primitives
```css
/* Neutral Dark / Space Ground */
--primitive-slate-950: #020617;
--primitive-slate-900: #0B0F19;
--primitive-slate-850: #111827;
--primitive-slate-800: #1E293B;
--primitive-slate-700: #334155;
--primitive-slate-600: #475569;
--primitive-slate-400: #94A3B8;
--primitive-slate-200: #E2E8F0;
--primitive-white:     #FFFFFF;

/* Cyan AI / Radar Core */
--primitive-cyan-400: #22D3EE;
--primitive-cyan-500: #06B6D4;
--primitive-cyan-600: #0891B2;
--primitive-cyan-glow:#00F2FE;

/* Emerald / Optical */
--primitive-emerald-400: #34D399;
--primitive-emerald-500: #10B981;
--primitive-emerald-600: #059669;

/* Amber / Microwave SAR */
--primitive-amber-400: #FBBF24;
--primitive-amber-500: #F59E0B;
--primitive-amber-600: #D97706;

/* Violet / Temporal Change */
--primitive-violet-400: #C084FC;
--primitive-violet-500: #A855F7;
--primitive-violet-600: #9333EA;

/* Rose / Error / Alert */
--primitive-rose-400: #FB7185;
--primitive-rose-500: #F43F5E;
--primitive-rose-600: #E11D48;
```

### 3.2 Elevation, Blur & Radii Primitives
* **Border Radii:**
  * `--radius-xs`: `4px` (Tags, badges, small buttons)
  * `--radius-sm`: `6px` (Inputs, dropdowns, telemetry readouts)
  * `--radius-md`: `10px` (Cards, panels, modal dialogs)
  * `--radius-lg`: `16px` (Main canvas containers)
  * `--radius-full`: `9999px` (Pill indicators, drag handles)
* **Backdrop Blur:**
  * `--blur-card`: `12px`
  * `--blur-nav`: `16px`
* **Glow Shadows:**
  * `--glow-cyan`: `0 0 20px rgba(6, 182, 212, 0.35)`
  * `--glow-violet`: `0 0 20px rgba(168, 85, 247, 0.35)`
  * `--glow-emerald`: `0 0 20px rgba(16, 185, 129, 0.35)`

---

# 4. Semantic Tokens (Layer 2)

| Semantic Token | Primitive Reference | Application / Purpose |
| :--- | :--- | :--- |
| `--surface-canvas` | `--primitive-slate-950` | App root background, satellite viewport background |
| `--surface-panel` | `--primitive-slate-900` | Sidebar panels, bottom console, audit drawer |
| `--surface-card` | `--primitive-slate-850` | Nested metric cards, tool cards, telemetry rows |
| `--surface-hover` | `--primitive-slate-800` | Hover states on interactive cards and list items |
| `--border-subtle` | `--primitive-slate-800` | Structural dividers, inactive card borders |
| `--border-active` | `--primitive-cyan-500` | Active input focus, active sensor slot outline |
| `--text-primary` | `--primitive-white` | Headings, primary answers, prompt text |
| `--text-secondary`| `--primitive-slate-400` | Labels, metadata descriptions, timestamps |
| `--text-mono` | `--primitive-cyan-400` | Coordinates, CRS tags, confidence numbers, latencies |
| `--sensor-optical`| `--primitive-emerald-400` | Optical chips, reflectance bands (B2, B3, B4, B8) |
| `--sensor-sar` | `--primitive-amber-400` | Radar chips, dB values, polarizations (VV, VH) |
| `--sensor-change` | `--primitive-violet-400` | Change masks, delta hectares, before/after comparison |
| `--status-online` | `--primitive-emerald-400` | Connected backend, healthy GPU/MOCK engine |
| `--status-error` | `--primitive-rose-500` | CRS mismatch, OOM warning, corrupted GeoTIFF |

---

# 5. Component Tokens & Specs (Layer 3)

### 5.1 Dual-Pane Swipe Compare Viewer
The center-stage interactive slider comparing $t_1$ vs. $t_2$ or Optical vs. SAR:
* `--slider-divider-width`: `2px`
* `--slider-divider-color`: `var(--primitive-cyan-400)`
* `--slider-divider-shadow`: `0 0 12px rgba(34, 211, 238, 0.8)`
* `--slider-handle-size`: `34px`
* `--slider-handle-bg`: `var(--primitive-cyan-500)`
* `--slider-handle-border`: `2px solid #FFFFFF`
* `--slider-handle-icon`: `⇄` (text in `#020617`, bold, 12px)

### 5.2 Sensor Slot Ingestion Cards
* **Slot 1 (Optical / $t_1$):**
  * Border: `1px solid var(--border-subtle)`
  * Active State: Border glows with `var(--sensor-optical)`
  * Badge: `bg-emerald-500/10 text-emerald-400 border border-emerald-500/30`
* **Slot 2 (SAR / $t_2$):**
  * Border: `1px solid var(--border-subtle)`
  * Active State: Border glows with `var(--sensor-sar)`
  * Badge: `bg-amber-500/10 text-amber-400 border border-amber-500/30`

### 5.3 Visual Grounding Bounding Box & Label
* **Bounding Box Border:** `2px solid var(--primitive-cyan-400)`
* **Fill Overlay:** `rgba(6, 182, 212, 0.12)`
* **Confidence Pill Tag:**
  * Font: `JetBrains Mono`, 10px, Medium
  * Background: `rgba(2, 6, 23, 0.85)` with `border: 1px solid var(--primitive-cyan-400)`
  * Text: `[Naval Vessel: 96.2%]`

### 5.4 Audit Trace Waterfall Timeline
* **Step Marker Dot:** `8px × 8px` rounded-full
  * Pending: `bg-slate-700`
  * Active: `bg-cyan-400 animate-ping`
  * Complete: `bg-emerald-400`
* **Latency Text:** `text-slate-400 font-mono text-xs` (e.g., `+142ms`)

---

# 6. Typography Scale & Font Pairings

| Role | Font Family | Size | Weight | Line Height | Usage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Display / Hero** | Inter / Geist Sans | 24px (`1.5rem`) | 700 (Bold) | 1.2 | Main Navigation Brand, Modal Headers |
| **Section Title** | Inter / Geist Sans | 16px (`1.0rem`) | 600 (Semibold) | 1.3 | Panel Titles, Metric Headers |
| **Body Primary** | Inter / Geist Sans | 14px (`0.875rem`)| 400 (Regular) | 1.5 | Natural Language Agent Responses |
| **Body Secondary**| Inter / Geist Sans | 12px (`0.75rem`) | 400 (Regular) | 1.4 | Helper text, input placeholders |
| **Telemetry Mono**| JetBrains Mono | 12px (`0.75rem`) | 500 (Medium) | 1.4 | Latencies, Coordinates, CRS, Confidence |
| **Micro Badge** | JetBrains Mono | 10px (`0.625rem`)| 600 (Semibold)| 1.0 | Modality pills (`16-BIT`, `SAR`, `UTM`) |

---

# 7. Iconography & Sensor Status System

* **Icon Library:** Lucide React (`lucide-react`)
* **Core Icon Mapping:**
  * 🛰️ `Satellite`: Mission Header, Satellite Rasters
  * 📡 `Radio`: SAR Microwave backscatter, RISAT sensor
  * 🌿 `Layers`: Multispectral Optical bands, Land Cover
  * ⇄ `Split`: Dual-pane swipe comparison
  * 🎯 `Crosshair`: Visual Grounding & Target Localization
  * ⚡ `Zap`: Agent state machine & fast routing
  * 📄 `FileDown`: One-click Executive Mission PDF generation
  * 🛡️ `ShieldCheck`: Input compatibility guard verification

---

# 8. Complete CSS Variables (`tokens.css`)

```css
:root {
  /* Primitive Colors */
  --color-slate-950: #020617;
  --color-slate-900: #0B0F19;
  --color-slate-850: #111827;
  --color-slate-800: #1E293B;
  --color-slate-700: #334155;
  --color-slate-400: #94A3B8;
  --color-white: #FFFFFF;

  --color-cyan-glow: #00F2FE;
  --color-cyan-500: #06B6D4;
  --color-cyan-400: #22D3EE;

  --color-emerald-400: #34D399;
  --color-emerald-500: #10B981;

  --color-amber-400: #FBBF24;
  --color-amber-500: #F59E0B;

  --color-violet-400: #C084FC;
  --color-violet-500: #A855F7;

  --color-rose-500: #F43F5E;

  /* Semantic Mappings */
  --bg-app: var(--color-slate-950);
  --bg-panel: var(--color-slate-900);
  --bg-card: var(--color-slate-850);
  --border-card: var(--color-slate-800);
  --border-glow: var(--color-cyan-500);

  --text-main: var(--color-white);
  --text-dim: var(--color-slate-400);
  --text-accent: var(--color-cyan-400);

  /* Component Shadows */
  --shadow-cyan-glow: 0 0 15px rgba(6, 182, 212, 0.35);
  --shadow-amber-glow: 0 0 15px rgba(245, 158, 11, 0.35);
  --shadow-violet-glow: 0 0 15px rgba(168, 85, 247, 0.35);
}
```

---

# 9. Tailwind CSS Configuration (`tailwind.config.ts`)

```typescript
import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        void: {
          950: "#020617",
          900: "#0B0F19",
          850: "#111827",
          800: "#1E293B",
        },
        telemetry: {
          cyan: "#22D3EE",
          emerald: "#34D399",
          amber: "#FBBF24",
          violet: "#C084FC",
          rose: "#F43F5E",
        },
      },
      fontFamily: {
        sans: ["Inter", "Geist", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
      boxShadow: {
        "cyan-glow": "0 0 20px -3px rgba(6, 182, 212, 0.4)",
        "amber-glow": "0 0 20px -3px rgba(245, 158, 11, 0.4)",
        "violet-glow": "0 0 20px -3px rgba(168, 85, 247, 0.4)",
      },
      animation: {
        pulseGlow: "pulseGlow 2.5s ease-in-out infinite",
      },
      keyframes: {
        pulseGlow: {
          "0%, 100%": { opacity: "0.8", filter: "drop-shadow(0 0 8px rgba(6,182,212,0.4))" },
          "50%": { opacity: "1", filter: "drop-shadow(0 0 16px rgba(6,182,212,0.8))" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
```

---
*Authored for SatQuery AI (ProjectS) | ISRO / Space Applications Centre — SIH 26167*
