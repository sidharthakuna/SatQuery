import React, { useState } from 'react';
import {
  Sparkles,
  FileText,
  Maximize2,
  Layers,
  Navigation,
  Info,
  X,
} from 'lucide-react';
import { useChat } from '../../context/ChatContext';

export interface VqaExampleItem {
  id: number;
  badgeColor: string;
  badgeTextColor: string;
  question: string;
  thumbnail: string;
  answerText: string;
  confidence: number;
  method: string;
  note: string;
  legendLabel: string;
  legendColor: string;
}

export const VQA_EXAMPLES: VqaExampleItem[] = [
  {
    id: 1,
    badgeColor: 'bg-blue-600',
    badgeTextColor: 'text-white',
    question: 'How many buildings are visible in this image?',
    thumbnail: '/api/v1/preview/card_1_buildings.tif',
    answerText:
      'Approximately 12,840 buildings are visible in the image. Most of them are concentrated in the central and coastal regions.\n\nThe image contains a mix of residential, commercial and industrial buildings.',
    confidence: 0.87,
    method: 'RS-VLM + Segmentation',
    note: 'Count is estimated and may vary for very small structures.',
    legendLabel: 'Detected Buildings',
    legendColor: '#ef4444',
  },
  {
    id: 2,
    badgeColor: 'bg-emerald-600',
    badgeTextColor: 'text-white',
    question: 'Show the roads in this area.',
    thumbnail: '/api/v1/preview/card_2_roads.tif',
    answerText:
      'Main roads and local roads are highlighted. Total road length: ≈ 124.6 km\n\nThe road network connects the coastal port facilities, commercial zones, and residential sectors.',
    confidence: 0.85,
    method: 'RS-VLM + Road Extraction',
    note: 'Total road length derived from 10m Sentinel-2 centerline vectorization.',
    legendLabel: 'Road Network',
    legendColor: '#eab308',
  },
  {
    id: 3,
    badgeColor: 'bg-orange-500',
    badgeTextColor: 'text-white',
    question: 'How many water bodies are present?',
    thumbnail: '/api/v1/preview/card_3_water.tif',
    answerText:
      '4 major water bodies detected (1 sea area, 2 lakes, 1 reservoir).\n\nSurface water occupies critical drainage and maritime transit zones across the surveyed scene.',
    confidence: 0.92,
    method: 'RS-VLM + Hydrological Segmentation',
    note: 'Water boundaries delineated via NDWI proxy and morphological connected components.',
    legendLabel: 'Water Bodies',
    legendColor: '#3b82f6',
  },
  {
    id: 4,
    badgeColor: 'bg-purple-600',
    badgeTextColor: 'text-white',
    question: 'What type of land cover is present in this image?',
    thumbnail: '/api/v1/preview/card_4_landcover.tif',
    answerText:
      'Urban: 46%\nVegetation: 38%\nWater: 8%\nOther (bare land, etc.): 8%\n\nThe landscape demonstrates a high-density coastal city with extensive urban sprawl and surrounding green space.',
    confidence: 0.88,
    method: 'RS-VLM + Land Cover Classification',
    note: 'Class shares calculated via 4-class multi-spectral classification.',
    legendLabel: 'Land Cover Classes',
    legendColor: '#8b5cf6',
  },
  {
    id: 5,
    badgeColor: 'bg-rose-500',
    badgeTextColor: 'text-white',
    question: 'Identify the port and its boundary.',
    thumbnail: '/api/v1/preview/card_5_port.tif',
    answerText:
      'Port area highlighted. Estimated area: 6.21 km²\n\nThe deepwater port infrastructure handles maritime container logistics with fortified breakwater barriers.',
    confidence: 0.90,
    method: 'RS-VLM + Boundary Delineation',
    note: 'Port area delineated via polygon envelope covering harbor berths and cargo logistics docks.',
    legendLabel: 'Port Boundary',
    legendColor: '#ef4444',
  },
  {
    id: 6,
    badgeColor: 'bg-indigo-600',
    badgeTextColor: 'text-white',
    question: 'How many ships are visible in the port?',
    thumbnail: '/api/v1/preview/card_6_ships.tif',
    answerText:
      '8 ships detected.\n\nAll vessels are positioned within the primary harbor basin and commercial shipping lanes.',
    confidence: 0.85,
    method: 'RS-VLM + Object Detection',
    note: 'Vessel detections produced by Grounding DINO with spatial non-maximum suppression.',
    legendLabel: 'Detected Ships',
    legendColor: '#22c55e',
  },
  {
    id: 7,
    badgeColor: 'bg-pink-600',
    badgeTextColor: 'text-white',
    question: 'Show the built-up area boundary.',
    thumbnail: '/api/v1/preview/card_7_builtup.tif',
    answerText:
      'Built-up area: 62.4 km² (shown in red).\n\nThe contiguous metropolitan envelope extends along the coastal corridor with high structural compactness and road connectivity.',
    confidence: 0.89,
    method: 'RS-VLM + Urban Extent Delineation',
    note: 'Built-up area delineated via high spatial frequency edge density and morphological closure.',
    legendLabel: 'Built-up Area',
    legendColor: '#ec4899',
  },
  {
    id: 8,
    badgeColor: 'bg-green-600',
    badgeTextColor: 'text-white',
    question: 'Are there any agricultural fields in this image?',
    thumbnail: '/api/v1/preview/card_8_agriculture.tif',
    answerText:
      'Yes. Agricultural fields are present in the northern region. Estimated area: 18.7 km²\n\nThese parcels exhibit active seasonal cultivation and irrigation networks.',
    confidence: 0.83,
    method: 'RS-VLM + Crop Parcel Segmentation',
    note: 'Agricultural parcels identified through regular geometric boundaries and active vegetative vigor.',
    legendLabel: 'Agricultural Fields',
    legendColor: '#16a34a',
  },
  {
    id: 9,
    badgeColor: 'bg-cyan-600',
    badgeTextColor: 'text-white',
    question: 'What changes are visible compared to a previous image?',
    thumbnail: '/api/v1/preview/card_9_change.tif',
    answerText:
      'New construction and road expansion detected in the eastern region.\n\nBi-temporal comparative analysis reveals emerging residential footprints and arterial road construction.',
    confidence: 0.81,
    method: 'RS-VLM + ChangeFormer',
    note: 'Change regions extracted by Siamese ChangeFormer network and validated against spectral delta.',
    legendLabel: 'New Construction & Road Expansion',
    legendColor: '#eab308',
  },
  {
    id: 10,
    badgeColor: 'bg-slate-700',
    badgeTextColor: 'text-white',
    question: 'Describe this image in short.',
    thumbnail: '/api/v1/preview/card_10_describe.tif',
    answerText:
      'A coastal city with a major port, dense urban areas, surrounding hills, vegetation and beaches along the eastern coast.',
    confidence: 0.90,
    method: 'RS-VLM Specialist',
    note: 'Scene description synthesized using multi-scale convolutional visual features and LoRA decoding.',
    legendLabel: 'Overview Scene',
    legendColor: '#0284c7',
  },
];

export const VqaStudioViewer: React.FC = () => {
  const { submitQuery } = useChat();
  const [selectedExample, setSelectedExample] = useState<VqaExampleItem>(VQA_EXAMPLES[0]);
  const [inputQuestion, setInputQuestion] = useState('');
  const [zoomModalUrl, setZoomModalUrl] = useState<string | null>(null);

  const handleSelectExample = (item: VqaExampleItem) => {
    setSelectedExample(item);
    setInputQuestion(item.question);
  };

  const handleAsk = () => {
    const q = inputQuestion.trim() || selectedExample.question;
    const matched = VQA_EXAMPLES.find(
      (e) => e.question.toLowerCase() === q.toLowerCase()
    );
    if (matched) {
      setSelectedExample(matched);
    }
    submitQuery(q);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-3 sm:p-6 space-y-6 font-sans select-none text-[var(--text-main)]">
      {/* ── TOP SECTION: 2-COLUMN DISPLAY (Input Satellite Image | Ask a Question VQA) ── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
        {/* Left Column: Input Satellite Image */}
        <div className="lg:col-span-5 flex flex-col rounded-2xl border border-[var(--border-subtle)] bg-[var(--bg-card)] shadow-subtle overflow-hidden">
          {/* Card Header */}
          <div className="px-4 py-3 border-b border-[var(--border-subtle)] bg-[var(--bg-panel)] flex items-center gap-2">
            <Layers className="w-4 h-4 text-sky-600 dark:text-sky-400" />
            <span className="font-semibold text-xs sm:text-sm tracking-tight text-[var(--text-main)]">
              Input Satellite Image
            </span>
          </div>

          {/* Satellite Image Frame with Cartographic Overlay */}
          <div className="relative flex-1 min-h-[320px] sm:min-h-[360px] bg-slate-950 overflow-hidden flex items-center justify-center group">
            <img
              src="/api/v1/preview/sentinel2_input.tif"
              alt="Sentinel-2 Satellite Scene"
              className="w-full h-full object-cover"
              onError={(e) => {
                (e.target as HTMLImageElement).src = '/api/v1/preview/port_grounding.tif';
              }}
            />

            {/* Top-Left Acquisition Badge */}
            <div className="absolute top-3 left-3 px-2.5 py-1.5 rounded-lg bg-black/60 backdrop-blur-md border border-white/20 text-white font-mono text-[10px] sm:text-[11px] leading-tight shadow-md">
              <div className="font-semibold text-sky-300">Sentinel-2 (True Color)</div>
              <div className="text-white/80">12 Jan 2026</div>
            </div>

            {/* Top-Right North Arrow */}
            <div className="absolute top-3 right-3 w-8 h-8 rounded-full bg-black/60 backdrop-blur-md border border-white/20 flex flex-col items-center justify-center text-white shadow-md">
              <Navigation className="w-3.5 h-3.5 text-rose-500 transform rotate-[-45deg]" />
              <span className="text-[8px] font-mono font-bold leading-none mt-0.5">N</span>
            </div>

            {/* Bottom-Left Metric Scale Bar */}
            <div className="absolute bottom-3 left-3 px-3 py-1 rounded-md bg-black/70 backdrop-blur-md border border-white/20 text-white font-mono text-[9.5px] flex items-center gap-3 shadow-md">
              <span>0</span>
              <span className="w-10 h-1 bg-white inline-block border-x border-white"></span>
              <span>2.5</span>
              <span className="w-10 h-1 bg-white/50 inline-block border-x border-white"></span>
              <span>5 km</span>
            </div>

            {/* Expand Hover Action */}
            <button
              type="button"
              onClick={() => setZoomModalUrl('/api/v1/preview/sentinel2_input.tif')}
              className="absolute bottom-3 right-3 p-1.5 rounded-lg bg-black/60 backdrop-blur-md border border-white/20 text-white opacity-0 group-hover:opacity-100 transition-opacity hover:bg-black/80 cursor-pointer"
              title="Expand satellite raster"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Right Column: Ask a Question (VQA) */}
        <div className="lg:col-span-7 flex flex-col rounded-2xl border border-[var(--border-subtle)] bg-[var(--bg-card)] shadow-subtle p-4 sm:p-5 space-y-4 justify-between">
          {/* Header & Question Search Bar */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sky-600 dark:text-sky-400 font-semibold text-xs sm:text-sm">
              <div className="w-6 h-6 rounded-lg bg-sky-500/15 flex items-center justify-center">
                <Sparkles className="w-3.5 h-3.5 text-sky-600 dark:text-sky-400" />
              </div>
              <span>Ask a Question (VQA)</span>
            </div>

            {/* Search Input Bar */}
            <div className="flex items-center gap-2">
              <div className="relative flex-1">
                <input
                  type="text"
                  value={inputQuestion}
                  onChange={(e) => setInputQuestion(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="e.g., How many buildings are visible in this image?"
                  className="w-full pl-3.5 pr-4 py-2.5 rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-surface)] text-xs sm:text-sm text-[var(--text-main)] placeholder-[var(--text-muted)] focus:outline-none focus:ring-2 focus:ring-sky-500/30 transition-all font-normal"
                />
              </div>
              <button
                type="button"
                onClick={handleAsk}
                className="px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-xs sm:text-sm font-semibold tracking-wide shadow-sm hover:shadow transition-all flex items-center gap-1.5 cursor-pointer shrink-0"
              >
                <span>Ask</span>
              </button>
            </div>
          </div>

          {/* Active Question Box with Green 'Q' */}
          <div className="flex items-start gap-3 p-3 rounded-xl bg-[var(--bg-panel)] border border-[var(--border-subtle)]/80">
            <div className="w-6 h-6 rounded-full bg-emerald-600 text-white flex items-center justify-center font-bold text-xs shrink-0 shadow-sm mt-0.5">
              Q
            </div>
            <div className="font-semibold text-xs sm:text-sm text-[var(--text-main)] leading-snug">
              {selectedExample.question}
            </div>
          </div>

          {/* Answer Section with Split Text + Visual Overlay */}
          <div className="space-y-2">
            <div className="flex items-center gap-1.5 text-sky-600 dark:text-sky-400 font-semibold text-xs tracking-wide">
              <FileText className="w-3.5 h-3.5" />
              <span>Answer</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-12 gap-3.5 items-start">
              {/* Text Narrative */}
              <div className="md:col-span-7 text-xs sm:text-[13px] leading-relaxed text-[var(--text-main)] font-normal space-y-2">
                <p className="whitespace-pre-line">
                  {selectedExample.answerText}
                </p>
              </div>

              {/* Visual Overlay Card */}
              <div className="md:col-span-5 flex flex-col items-center">
                <div
                  onClick={() => setZoomModalUrl(selectedExample.thumbnail)}
                  className="w-full relative group cursor-pointer rounded-xl overflow-hidden border border-[var(--border-subtle)] bg-slate-950 shadow-sm transition-transform hover:scale-[1.02]"
                  title="Click to expand high-resolution overlay"
                >
                  <img
                    src={selectedExample.thumbnail}
                    alt={selectedExample.legendLabel}
                    className="w-full h-36 object-cover"
                    onError={(e) => {
                      (e.target as HTMLImageElement).src = '/api/v1/preview/building_overlay.tif';
                    }}
                  />
                  <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center text-white text-xs font-medium gap-1.5 backdrop-blur-[1px]">
                    <Maximize2 className="w-3.5 h-3.5" />
                    <span>Expand</span>
                  </div>
                </div>

                {/* Legend Tag below overlay */}
                <div className="mt-2 inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md bg-[var(--bg-surface)] border border-[var(--border-subtle)] text-[10.5px] font-mono text-[var(--text-main)]">
                  <span
                    className="w-2.5 h-2.5 rounded-sm shrink-0"
                    style={{ backgroundColor: selectedExample.legendColor }}
                  />
                  <span>{selectedExample.legendLabel}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Additional Information Card (Light Blue Container matching screenshot) */}
          <div className="p-3.5 rounded-xl bg-sky-50 dark:bg-sky-950/20 border border-sky-100 dark:border-sky-900/40 space-y-2 text-xs">
            <div className="flex items-center gap-1.5 text-sky-800 dark:text-sky-300 font-semibold text-xs">
              <Info className="w-3.5 h-3.5 text-sky-600 dark:text-sky-400" />
              <span>Additional Information</span>
            </div>

            <div className="space-y-1 font-mono text-[11px] sm:text-[11.5px]">
              <div className="grid grid-cols-12 gap-2 text-[var(--text-main)]">
                <span className="col-span-3 text-[var(--text-muted)]">Method</span>
                <span className="col-span-9 font-medium">: {selectedExample.method}</span>
              </div>
              <div className="grid grid-cols-12 gap-2 text-[var(--text-main)]">
                <span className="col-span-3 text-[var(--text-muted)]">Confidence</span>
                <span className="col-span-9 font-medium text-emerald-600 dark:text-emerald-400">
                  : {(selectedExample.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <div className="grid grid-cols-12 gap-2 text-[var(--text-main)]">
                <span className="col-span-3 text-[var(--text-muted)]">Note</span>
                <span className="col-span-9 text-[var(--text-muted)] font-normal">
                  : {selectedExample.note}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── BOTTOM SECTION: 10 EXAMPLE QUESTIONS AND ANSWERS (5x2 Grid) ── */}
      <div className="space-y-3 pt-2">
        {/* Section Header */}
        <div className="flex items-center gap-2 text-sky-700 dark:text-sky-400 font-semibold text-xs sm:text-sm">
          <Layers className="w-4 h-4" />
          <span>Example Questions and Answers</span>
        </div>

        {/* 10 Cards Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
          {VQA_EXAMPLES.map((item) => {
            const isSelected = selectedExample.id === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => handleSelectExample(item)}
                className={`p-2.5 rounded-xl border text-left transition-all flex flex-col justify-between group cursor-pointer ${
                  isSelected
                    ? 'border-sky-500 ring-2 ring-sky-500/20 bg-[var(--bg-card)] shadow-md'
                    : 'border-[var(--border-subtle)] bg-[var(--bg-panel)] hover:bg-[var(--bg-surface)] hover:border-[var(--border-hover)]'
                }`}
              >
                {/* Card Top: Number badge + Question */}
                <div className="space-y-2 mb-2">
                  <div className="flex items-center gap-2">
                    <span
                      className={`w-5 h-5 rounded-full ${item.badgeColor} ${item.badgeTextColor} text-[10.5px] font-bold flex items-center justify-center shrink-0 shadow-xs`}
                    >
                      {item.id}
                    </span>
                    <span className="text-[11px] font-medium text-[var(--text-main)] line-clamp-2 leading-snug">
                      {item.question}
                    </span>
                  </div>

                  {/* Visual Thumbnail */}
                  <div className="w-full h-20 rounded-lg overflow-hidden border border-[var(--border-subtle)]/70 bg-slate-950 relative">
                    <img
                      src={item.thumbnail}
                      alt={item.question}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                      onError={(e) => {
                        (e.target as HTMLImageElement).src = '/api/v1/preview/building_overlay.tif';
                      }}
                    />
                  </div>
                </div>

                {/* Card Bottom: Summary + Confidence */}
                <div className="pt-1.5 border-t border-[var(--border-subtle)]/60 text-[10.5px] font-mono flex flex-col justify-between">
                  <div className="font-semibold text-[var(--text-main)] line-clamp-2 leading-tight mb-1">
                    {item.id === 1 && '= 12,840 buildings'}
                    {item.id === 2 && 'Main roads and local roads. Total: ≈ 124.6 km'}
                    {item.id === 3 && '4 major water bodies (1 sea, 2 lakes, 1 res.)'}
                    {item.id === 4 && 'Urban: 46% | Veg: 38% | Water: 8%'}
                    {item.id === 5 && 'Port highlighted. Area: 6.21 km²'}
                    {item.id === 6 && '8 ships detected.'}
                    {item.id === 7 && 'Built-up area: 62.4 km² (red).'}
                    {item.id === 8 && 'Agricultural fields: 18.7 km²'}
                    {item.id === 9 && 'New construction & road expansion'}
                    {item.id === 10 && 'Coastal city with major port & hills'}
                  </div>
                  <div className="text-[10px] text-[var(--text-muted)]">
                    Confidence: {(item.confidence * 100).toFixed(0)}%
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* ── FOOTER BAR: SatQuery AI Brand & Capabilities ── */}
      <div className="pt-4 border-t border-[var(--border-subtle)] flex flex-col md:flex-row items-center justify-between gap-3 text-xs text-[var(--text-muted)]">
        {/* Brand & Subtitle */}
        <div className="flex items-center gap-2">
          <span className="font-bold text-sm tracking-tight text-[var(--text-main)]">
            SatQuery <span className="text-sky-600 dark:text-sky-400 font-extrabold">AI</span>
          </span>
          <span className="text-[11px] text-[var(--text-muted)] hidden sm:inline">
            From Satellite Data to Real Answers.
          </span>
        </div>

        {/* 4 Feature Pills */}
        <div className="flex flex-wrap items-center gap-3 font-mono text-[10px] sm:text-[10.5px]">
          <span className="inline-flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-sky-500" />
            Remote Sensing VQA
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
            Object Detection
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-purple-500" />
            Spatial Understanding
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
            Real-world Insights
          </span>
        </div>

        {/* Right Tagline */}
        <div className="font-mono text-[10.5px] text-[var(--text-muted)] text-right">
          Ask. Analyze. Understand. A Safer Tomorrow.
        </div>
      </div>

      {/* ── HIGH-RESOLUTION ZOOM MODAL ── */}
      {zoomModalUrl && (
        <div
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4"
          onClick={() => setZoomModalUrl(null)}
        >
          <div
            className="relative max-w-4xl w-full bg-slate-950 rounded-2xl overflow-hidden border border-white/20 shadow-2xl p-2"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              type="button"
              onClick={() => setZoomModalUrl(null)}
              className="absolute top-4 right-4 p-2 rounded-full bg-black/60 text-white hover:bg-black/80 transition-colors z-10 cursor-pointer"
              title="Close"
            >
              <X className="w-5 h-5" />
            </button>
            <img
              src={zoomModalUrl}
              alt="High resolution satellite overlay"
              className="w-full max-h-[80vh] object-contain rounded-xl"
            />
          </div>
        </div>
      )}
    </div>
  );
};
