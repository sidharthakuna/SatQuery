import React, { useState } from 'react';
import { AlertCircle } from 'lucide-react';
import type { TraceStep } from '../../types/api';
import LatticeLoader from '../ui/LatticeLoader';

interface StreamingStatusProps {
  steps: TraceStep[];
  isStreaming?: boolean;
}

export const StreamingStatus: React.FC<StreamingStatusProps> = ({
  steps,
  isStreaming = false,
}) => {
  if (!isStreaming && (!steps || steps.length === 0)) return null;

  const humanizeStepName = (name: string): string => {
    const upper = name.toUpperCase();
    if (upper.includes('DISPATCH')) return 'Connecting to SatQuery Orchestrator';
    if (upper.includes('VALIDAT')) return 'Validating raster formats & CRS compatibility';
    if (upper.includes('CLASSIF')) return 'Classifying query intent';
    if (upper.includes('TOOL_GROUNDING') || upper.includes('GROUNDING')) {
      return 'Running visual grounding to delineate targets';
    }
    if (upper.includes('CHANGE_DETECTION') || upper.includes('CHANGE')) {
      return 'Computing bi-temporal change mask';
    }
    if (upper.includes('FUSION') || upper.includes('OPTICAL_SAR')) {
      return 'Cross-attention fusion of Optical & SAR data';
    }
    if (upper.includes('VQA') || upper.includes('RS_VQA')) {
      return 'Analyzing remote sensing imagery';
    }
    if (upper.includes('FUSING')) return 'Synthesizing spatial evidence & metrics';
    if (upper.includes('COMPILED') || upper.includes('TRACE')) return 'Analysis trace compiled';
    return name.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
  };

  // If there are less than 2 steps, it's likely a generic/simple question ("Hi").
  // So we don't show the audit trace, but we still show the loader if streaming.
  const showTraces = steps && steps.length > 1;

  if (!isStreaming && !showTraces) {
    return null; // Hide everything if done and no complex trace
  }

  return (
    <div className="w-full mb-3 flex flex-col gap-2">
      {/* Lattice Loader prominently displayed while streaming */}
      {isStreaming && (
        <div className="flex items-center text-[var(--text-main)] mb-1">
          <LatticeLoader
            status="working"
            label="Thinking..."
            pattern="orbit"
            grid={3}
            shape="round"
            color="var(--accent-primary)"
            cellSize={6}
            gap={2}
            fontSize={14}
            step={90}
            idleOpacity={0.15}
            glow={true}
            showTimer={true}
            style={{}}
            elapsed={50}
          />
        </div>
      )}

      {/* Audit trace underneath, small text */}
      {showTraces && (
        <div className="rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-panel)] overflow-hidden text-xs transition-colors">
          <div className="px-3.5 py-2 text-[10px] uppercase font-bold text-[var(--text-muted)] tracking-wider border-b border-[var(--border-subtle)] bg-[var(--bg-surface)]">
            Audit Trace
          </div>
          <div className="px-3.5 pb-2.5 pt-1.5 space-y-1.5 bg-[var(--bg-card)]/50">
            {steps.map((step, idx) => {
              const isLatest = idx === steps.length - 1;
              const isError = step.status === 'ERROR';
              const inProgress = step.status === 'IN_PROGRESS' || (isStreaming && isLatest);

              return (
                <div key={`${step.step_name}_${step.step_index ?? idx}`} className="flex items-center justify-between gap-2 text-[11px] font-mono">
                  <div className="flex items-center gap-2.5 truncate">
                    {isError ? (
                      <AlertCircle className="w-3 h-3 text-[var(--accent-primary)] shrink-0" />
                    ) : inProgress ? (
                      <div className="w-1.5 h-1.5 rounded-full bg-[var(--accent-primary)] animate-pulse shrink-0" />
                    ) : (
                      <div className="w-1.5 h-1.5 rounded-full bg-[var(--text-dim)]/60 shrink-0" />
                    )}

                    <span
                      className={`truncate ${inProgress
                          ? 'text-[var(--text-main)] font-medium'
                          : isError
                            ? 'text-[var(--accent-primary)]'
                            : 'text-[var(--text-muted)]'
                        }`}
                    >
                      {humanizeStepName(step.step_name)}
                    </span>
                  </div>

                  {step.duration_ms > 0 && (
                    <span className="text-[10px] text-[var(--text-dim)] shrink-0">
                      +{step.duration_ms.toFixed(0)}ms
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
