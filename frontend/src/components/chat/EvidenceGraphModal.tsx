import React, { useState } from 'react';
import {
  X,
  Share2,
  CheckCircle2,
  AlertTriangle,
  ShieldCheck,
  Cpu,
  Layers,
  FileCheck,
  ArrowRight,
  Info,
  Copy,
  Check,
} from 'lucide-react';
import type { EvidenceGraph, EvidenceNode } from '../../types/api';

interface EvidenceGraphModalProps {
  isOpen: boolean;
  onClose: () => void;
  graph?: EvidenceGraph;
  traceId?: string;
  taskType?: string;
}

export const EvidenceGraphModal: React.FC<EvidenceGraphModalProps> = ({
  isOpen,
  onClose,
  graph,
  traceId = 'UNKNOWN',
  taskType = 'GEOSPATIAL_ANALYSIS',
}) => {
  const [selectedNode, setSelectedNode] = useState<EvidenceNode | null>(null);
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  const nodes = graph?.nodes || [];
  const edges = graph?.edges || [];

  const handleCopyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(graph, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getNodeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'query':
      case 'interpretation':
        return <Info className="w-4 h-4 text-[#0EA5E9]" />;
      case 'gate':
      case 'validation':
        return <ShieldCheck className="w-4 h-4 text-sky-500" />;
      case 'model':
      case 'specialist':
        return <Cpu className="w-4 h-4 text-purple-500" />;
      case 'spatial':
      case 'mask':
      case 'raster':
        return <Layers className="w-4 h-4 text-emerald-500" />;
      case 'verification':
      case 'consistency':
        return <CheckCircle2 className="w-4 h-4 text-teal-500" />;
      default:
        return <FileCheck className="w-4 h-4 text-[var(--text-muted)]" />;
    }
  };

  const getStatusBadge = (status: string) => {
    const s = (status || '').toUpperCase();
    if (s === 'SUPPORTED' || s === 'VERIFIED' || s === 'SUCCESS') {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-medium bg-emerald-500/10 text-emerald-600 border border-emerald-500/20">
          <CheckCircle2 className="w-3 h-3" />
          {s}
        </span>
      );
    }
    if (s === 'CAVEAT' || s === 'UNCERTAIN') {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-medium bg-amber-500/10 text-amber-600 border border-amber-500/20">
          <AlertTriangle className="w-3 h-3" />
          {s}
        </span>
      );
    }
    if (s === 'CONTRADICTED' || s === 'INSUFFICIENT_EVIDENCE') {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-medium bg-rose-500/10 text-rose-600 border border-rose-500/20">
          <AlertTriangle className="w-3 h-3" />
          {s}
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-medium bg-[var(--bg-app)] text-[var(--text-muted)] border border-[var(--border-subtle)]">
        {s || 'INFO'}
      </span>
    );
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6 bg-black/60 backdrop-blur-sm animate-fadeIn">
      <div
        className="relative w-full max-w-5xl h-[88vh] flex flex-col bg-[var(--bg-surface)] rounded-2xl border border-[var(--border-subtle)] shadow-2xl overflow-hidden animate-scaleUp"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header Bar */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border-subtle)] bg-[var(--bg-app)]/50">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-[#0EA5E9]/10 border border-[#0EA5E9]/20 flex items-center justify-center text-[#0EA5E9]">
              <Share2 className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-base font-semibold text-[var(--text-main)] tracking-tight">
                  Evidence Graph DAG & Provenance Trace
                </h3>
                <span className="text-[11px] font-mono bg-[var(--bg-user-bubble)] px-2 py-0.5 rounded border border-[var(--border-subtle)] text-[var(--text-muted)]">
                  {taskType}
                </span>
              </div>
              <p className="text-xs text-[var(--text-muted)] font-mono">
                Observable execution trace ID: {traceId}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopyJson}
              className="px-3 py-1.5 rounded-lg border border-[var(--border-subtle)] hover:bg-[var(--bg-hover)] text-xs text-[var(--text-muted)] hover:text-[var(--text-main)] transition-colors flex items-center gap-1.5 font-mono cursor-pointer"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? 'Copied JSON' : 'Export JSON'}</span>
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg border border-[var(--border-subtle)] hover:bg-[var(--bg-hover)] text-[var(--text-muted)] hover:text-[var(--text-main)] transition-colors cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Content Area: Graph Diagram + Node Inspector */}
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 overflow-hidden">
          {/* Left / Center 2 Cols: Interactive Graph Flow */}
          <div className="lg:col-span-2 overflow-y-auto p-6 space-y-4 border-b lg:border-b-0 lg:border-r border-[var(--border-subtle)] bg-[var(--bg-surface)]">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-[var(--text-dim)]">
                Directed Acyclic Graph ({nodes.length} Nodes, {edges.length} Edges)
              </span>
              <span className="text-[11px] text-[var(--text-dim)]">
                Click any node to inspect telemetry payload
              </span>
            </div>

            {nodes.length === 0 ? (
              <div className="h-64 flex flex-col items-center justify-center text-center p-6 border-2 border-dashed border-[var(--border-subtle)] rounded-xl">
                <Info className="w-8 h-8 text-[var(--text-dim)] mb-2" />
                <p className="text-sm text-[var(--text-muted)]">No evidence nodes recorded for this session.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {nodes.map((node, idx) => {
                  const isSelected = selectedNode?.id === node.id;
                  const outgoingEdges = edges.filter((e) => e.source === node.id);

                  return (
                    <div key={node.id} className="space-y-2">
                      <div
                        onClick={() => setSelectedNode(node)}
                        className={`cursor-pointer p-4 rounded-xl border transition-all ${
                          isSelected
                            ? 'bg-[#0EA5E9]/10 border-[#0EA5E9] shadow-md ring-1 ring-[#0EA5E9]'
                            : 'bg-[var(--bg-app)]/60 hover:bg-[var(--bg-app)] border-[var(--border-subtle)] hover:border-[var(--border-hover)]'
                        }`}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex items-center gap-2.5">
                            <div className="p-2 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)]">
                              {getNodeIcon(node.type)}
                            </div>
                            <div>
                              <div className="flex items-center gap-2">
                                <span className="font-mono text-xs text-[var(--text-muted)] font-semibold">
                                  [{node.id}]
                                </span>
                                <h4 className="text-sm font-medium text-[var(--text-main)]">
                                  {node.title}
                                </h4>
                              </div>
                              <span className="text-[11px] font-mono text-[var(--text-dim)]">
                                Type: {node.type}
                              </span>
                            </div>
                          </div>

                          <div className="flex flex-col items-end gap-1.5">
                            {getStatusBadge(node.status)}
                            {node.confidence !== undefined && node.confidence > 0 && (
                              <span className="text-[11px] font-mono text-[var(--text-muted)]">
                                Conf: {(node.confidence * 100).toFixed(0)}%
                              </span>
                            )}
                          </div>
                        </div>

                        {/* Summary snippet if details available */}
                        {node.details && Object.keys(node.details).length > 0 && (
                          <div className="mt-3 pt-2.5 border-t border-[var(--border-subtle)]/60 flex flex-wrap gap-2 text-[11px] font-mono text-[var(--text-muted)]">
                            {Object.entries(node.details).slice(0, 3).map(([key, val]) => (
                              <span
                                key={key}
                                className="px-2 py-0.5 rounded bg-[var(--bg-surface)] border border-[var(--border-subtle)] truncate max-w-[240px]"
                              >
                                <span className="text-[var(--text-dim)]">{key}:</span>{' '}
                                {typeof val === 'object' ? JSON.stringify(val) : String(val)}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>

                      {/* Edge indicators to next nodes */}
                      {outgoingEdges.length > 0 && idx < nodes.length - 1 && (
                        <div className="flex items-center justify-center gap-2 py-1 text-[var(--text-dim)]">
                          <div className="h-4 w-px bg-[var(--border-subtle)]" />
                          {outgoingEdges.map((e, eIdx) => (
                            <span
                              key={eIdx}
                              className="text-[10px] font-mono px-2 py-0.5 rounded bg-[var(--bg-surface)] border border-[var(--border-subtle)] flex items-center gap-1"
                            >
                              <ArrowRight className="w-2.5 h-2.5 text-[#0EA5E9]" />
                              {e.label || 'flows to'} <span className="text-[var(--text-main)] font-semibold">{e.target}</span>
                            </span>
                          ))}
                          <div className="h-4 w-px bg-[var(--border-subtle)]" />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Right 1 Col: Node Telemetry Inspector */}
          <div className="p-6 overflow-y-auto bg-[var(--bg-app)]/30 space-y-4">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-dim)]">
              Node Telemetry Inspector
            </h4>

            {selectedNode ? (
              <div className="space-y-4 animate-fadeIn">
                <div className="p-4 rounded-xl bg-[var(--bg-surface)] border border-[var(--border-subtle)] space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-bold text-[#0EA5E9]">
                      {selectedNode.id}
                    </span>
                    {getStatusBadge(selectedNode.status)}
                  </div>
                  <h5 className="text-sm font-semibold text-[var(--text-main)]">
                    {selectedNode.title}
                  </h5>
                  <div className="text-xs text-[var(--text-muted)]">
                    Category: <span className="font-mono text-[var(--text-main)]">{selectedNode.type}</span>
                  </div>
                  {selectedNode.confidence > 0 && (
                    <div className="space-y-1 pt-1">
                      <div className="flex justify-between text-[11px] font-mono text-[var(--text-muted)]">
                        <span>Confidence Weight</span>
                        <span>{(selectedNode.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-[var(--bg-app)] rounded-full h-1.5 overflow-hidden">
                        <div
                          className="bg-[#0EA5E9] h-1.5 rounded-full transition-all"
                          style={{ width: `${selectedNode.confidence * 100}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>

                <div className="space-y-2">
                  <span className="text-xs font-medium text-[var(--text-muted)]">Payload Telemetry (JSON)</span>
                  <pre className="p-3.5 rounded-xl bg-[var(--bg-surface)] border border-[var(--border-subtle)] text-[11px] font-mono text-[var(--text-main)] overflow-x-auto whitespace-pre-wrap max-h-96">
                    {JSON.stringify(selectedNode.details, null, 2)}
                  </pre>
                </div>
              </div>
            ) : (
              <div className="h-64 flex flex-col items-center justify-center text-center p-6 border-2 border-dashed border-[var(--border-subtle)] rounded-xl text-[var(--text-dim)]">
                <Info className="w-6 h-6 mb-2" />
                <p className="text-xs">Select any node on the left to inspect its verifiable features, spectral indices, or model inputs.</p>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-[var(--border-subtle)] bg-[var(--bg-app)] flex items-center justify-between text-xs text-[var(--text-muted)] font-mono">
          <span>ISRO PS 26167 Ground Truth Verification Pipeline</span>
          <span>Status: CALIBRATED & VERIFIED</span>
        </div>
      </div>
    </div>
  );
};
