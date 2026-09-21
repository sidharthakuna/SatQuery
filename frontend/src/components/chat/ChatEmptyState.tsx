import React, { useState, useRef, useEffect } from 'react';
import {
  Sparkles,
  ArrowUpRight,
  Layers,
  Waves,
  Target,
  FileText,
  Loader2,
  ShieldCheck,
} from 'lucide-react';
import { useChat } from '../../context/ChatContext';
import { SatQueryLogo } from '../ui/SatQueryLogo';

interface QuickPrompt {
  id: number;
  label: string;
  query: string;
  icon: React.ReactNode;
  presetType?: string;
}

const QUICK_PROMPTS: QuickPrompt[] = [
  {
    id: 1,
    label: 'Flood Inundation & Safe Zones',
    query: 'Detect flood inundation, map submerged parcel boundaries, and quantify displacement between pre- and post-disaster dates.',
    icon: <Waves className="w-3.5 h-3.5 text-sky-400" />,
    presetType: 'flood',
  },
  {
    id: 2,
    label: 'Optical + SAR Cloud Fusion',
    query: 'Fuse optical and SAR imagery to penetrate clouds and identify standing water.',
    icon: <Layers className="w-3.5 h-3.5 text-indigo-400" />,
    presetType: 'sar',
  },
  {
    id: 3,
    label: 'Locate Buildings & Infrastructure',
    query: 'Locate and count all buildings and roads in this satellite scene.',
    icon: <Target className="w-3.5 h-3.5 text-emerald-400" />,
    presetType: 'grounding',
  },
  {
    id: 4,
    label: 'Vegetation & Land Cover Analysis',
    query: 'Analyze the vegetation canopy health and delineate land cover categories.',
    icon: <Sparkles className="w-3.5 h-3.5 text-amber-400" />,
    presetType: 'canopy',
  },
  {
    id: 5,
    label: 'Cloud-Free SAR Flood & Safe Zones',
    query: 'Penetrate storm clouds using Sentinel-1 SAR and Sentinel-2 optical imagery, reconstruct a cloud-free ground view, calculate total flooded area, and pinpoint elevated safe evacuation zones.',
    icon: <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />,
    presetType: 'cloud_free_flood',
  },
];

export const ChatEmptyState: React.FC = () => {
  const { activeImages, submitQuery, loadPresetAnalysis, isProcessing } = useChat();
  const [loadingId, setLoadingId] = useState<number | null>(null);
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  const handlePromptClick = async (prompt: QuickPrompt) => {
    if (loadingId !== null || isProcessing) return;
    setLoadingId(prompt.id);

    try {
      if (activeImages.length > 0) {
        await submitQuery(prompt.query);
      } else {
        await loadPresetAnalysis(prompt.presetType || 'flood');
      }
    } catch (err) {
      console.error('Failed to run quick prompt:', err);
    } finally {
      if (isMountedRef.current) setLoadingId(null);
    }
  };

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-4 sm:p-6 w-full max-w-4xl mx-auto select-none my-auto">
      {/* ── Central Hero Section ── */}
      <div className="flex flex-col items-center justify-center text-center max-w-xl mx-auto space-y-4">
        <div className="w-16 h-16 rounded-2xl bg-[var(--bg-card)] border border-[var(--border-subtle)] flex items-center justify-center shadow-claude transition-transform hover:scale-105">
          <SatQueryLogo size={36} />
        </div>

        <div className="space-y-2">
          <h1 className="text-2xl sm:text-3xl font-semibold text-[var(--text-main)] tracking-tight">
            What would you like to analyze?
          </h1>
          <p className="text-xs sm:text-sm text-[var(--text-muted)] max-w-md mx-auto leading-relaxed">
            Upload optical or SAR satellite rasters, or ask a question to begin multimodal Earth observation intelligence, disaster impact analysis, or change detection.
          </p>
        </div>

        {/* ── Subtle Minimalist Prompt Suggestions ── */}
        <div className="pt-2 w-full flex flex-wrap items-center justify-center gap-2 max-w-2xl">
          {QUICK_PROMPTS.map((item) => {
            const isLoadingThis = loadingId === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => handlePromptClick(item)}
                disabled={loadingId !== null}
                className="inline-flex items-center gap-2 px-3.5 py-2 rounded-full border border-[var(--border-subtle)] bg-[var(--bg-card)] hover:bg-[var(--bg-surface)] hover:border-sky-500/40 text-[12px] text-[var(--text-main)] transition-all cursor-pointer shadow-xs hover:shadow-sm group disabled:opacity-50"
              >
                {isLoadingThis ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-sky-400" />
                ) : (
                  item.icon
                )}
                <span className="font-medium">{item.label}</span>
                <ArrowUpRight className="w-3 h-3 text-[var(--text-dim)] group-hover:text-sky-400 transition-colors" />
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};
