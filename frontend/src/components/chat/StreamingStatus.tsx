import React, { useState } from 'react';
import { Check, ChevronDown, ChevronUp, AlertCircle } from 'lucide-react';
import type { TraceStep } from '../../types/api';

interface StreamingStatusProps {
  steps: TraceStep[];
  isStreaming?: boolean;
}

export const StreamingStatus: React.FC<StreamingStatusProps> = ({
  steps,
  isStreaming = false,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!steps || steps.length === 0) return null;

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

  const latestStep = steps[steps.length - 1];

  return (
    <div className="w-full mb-3 rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-panel)] overflow-hidden text-xs transition-colors">
      {/* Accordion Header (Claude Thinking Style) */}
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-3.5 py-2 flex items-center justify-between text-left hover:bg-[var(--bg-surface)] transition-colors"
      >
        <div className="flex items-center gap-2.5">
          {isStreaming ? (
            <div className="w-2 h-2 rounded-full bg-[#cc785c] animate-pulse" />
          ) : (
            <div className="w-3.5 h-3.5 rounded-full bg-[var(--bg-surface)] border border-[var(--border-subtle)] flex items-center justify-center text-[var(--text-muted)]">
              <Check className="w-2 h-2 text-[#cc785c]" />
            </div>
          )}

          <span className="font-mono text-[var(--text-main)] text-[11.5px]">
            {isStreaming ? humanizeStepName(latestStep.step_name) : `Analyzed ${steps.length} telemetry steps`}
          </span>
        </div>

        <div className="flex items-center gap-1.5 text-[var(--text-dim)] text-[11px] font-mono">
          <span>{isExpanded ? 'Hide' : 'Details'}</span>
          {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </div>
      </button>

      {/* Steps List */}
      {isExpanded && (
        <div className="px-3.5 pb-2.5 pt-1.5 border-t border-[var(--border-subtle)] space-y-1.5 bg-[var(--bg-card)]/50">
          {steps.map((step, idx) => {
            const isLatest = idx === steps.length - 1;
            const isError = step.status === 'ERROR';
            const inProgress = step.status === 'IN_PROGRESS' || (isStreaming && isLatest);

            return (
              <div key={`${step.step_name}_${step.step_index ?? idx}`} className="flex items-center justify-between gap-2 text-[11px] font-mono">
                <div className="flex items-center gap-2.5 truncate">
                  {isError ? (
                    <AlertCircle className="w-3 h-3 text-[#cc785c] shrink-0" />
                  ) : inProgress ? (
                    <div className="w-1.5 h-1.5 rounded-full bg-[#cc785c] animate-pulse shrink-0" />
                  ) : (
                    <div className="w-1.5 h-1.5 rounded-full bg-[var(--text-dim)]/60 shrink-0" />
                  )}

                  <span
                    className={`truncate ${
                      inProgress
                        ? 'text-[var(--text-main)] font-medium'
                        : isError
                        ? 'text-[#cc785c]'
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
      )}
    </div>
  );
};
