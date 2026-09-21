import React, { useState } from 'react';
import {
  Plus,
  MessageSquare,
  Trash2,
  Settings,
  HelpCircle,
  ChevronLeft,
} from 'lucide-react';
import { useChat } from '../../context/ChatContext';
import { Modal } from '../ui/Modal';
import { SatQueryLogo } from '../ui/SatQueryLogo';
import { SatQueryAPI } from '../../services/api';

export const Sidebar: React.FC = () => {
  const {
    sessions,
    activeSessionId,
    selectSession,
    deleteSession,
    createNewSession,
    isSidebarOpen,
    setSidebarOpen,
    systemHealth,
  } = useChat();

  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isHelpOpen, setIsHelpOpen] = useState(false);

  if (!isSidebarOpen) return null;

  return (
    <>
      <aside className="w-64 h-full bg-[var(--bg-panel)] border-r border-[var(--border-subtle)] flex flex-col justify-between shrink-0 z-40 transition-all select-none">
        {/* Top Section */}
        <div className="p-3 flex flex-col flex-1 overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between px-2 py-1 mb-2">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-lg bg-[var(--bg-card)] border border-[var(--border-subtle)] flex items-center justify-center">
                <SatQueryLogo size={16} />
              </div>

              <span className="font-semibold text-xs text-[var(--text-main)]">
                SatQuery AI
              </span>
            </div>
            <button
              type="button"
              onClick={() => setSidebarOpen(false)}
              className="p-1 text-[var(--text-muted)] hover:text-[var(--text-main)] rounded-lg lg:hidden"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
          </div>

          {/* + New Analysis Button (Claude warm rounded pill) */}
          <button
            type="button"
            onClick={createNewSession}
            className="w-full py-2 px-3 mb-3 rounded-xl border border-[var(--border-subtle)] hover:border-[var(--border-hover)] bg-[var(--bg-card)] hover:bg-[var(--bg-surface)] text-[var(--text-main)] font-medium text-xs transition-colors flex items-center gap-2 shadow-subtle"
          >
            <Plus className="w-4 h-4 text-[#0EA5E9]" />
            New Analysis
          </button>

          {/* Recent Analyses List */}
          <div className="flex-1 overflow-y-auto pr-1">
            <span className="text-[10px] uppercase font-mono tracking-wider text-[var(--text-dim)] px-2 block mb-1">
              History
            </span>

            <div className="space-y-0.5">
              {sessions.map((s) => {
                const isActive = s.id === activeSessionId;
                const thumb = s.images?.[0]?.thumbnail_url || s.lastResult?.thumbnail_urls?.[0];
                return (
                  <div
                    key={s.id}
                    onClick={() => selectSession(s.id)}
                    className={`group flex items-center justify-between px-2.5 py-1.5 rounded-xl text-xs cursor-pointer transition-colors ${
                      isActive
                        ? 'bg-[var(--bg-surface)] text-[var(--text-main)] font-medium border border-[var(--border-subtle)] shadow-subtle'
                        : 'text-[var(--text-muted)] hover:bg-[var(--bg-surface)] hover:text-[var(--text-main)]'
                    }`}
                  >
                    <div className="flex items-center gap-2 truncate min-w-0">
                      {thumb ? (
                        <img
                          src={SatQueryAPI.getRasterPreviewUrl(thumb)}
                          alt=""
                          className="w-4 h-4 rounded object-cover border border-[var(--border-subtle)] bg-black shrink-0"
                          onError={(e) => {
                            (e.currentTarget as HTMLImageElement).style.display = 'none';
                          }}
                        />
                      ) : (
                        <MessageSquare className="w-3.5 h-3.5 shrink-0 opacity-60 text-[#0EA5E9]" />
                      )}
                      <span className="truncate">{s.title || 'Untitled Query'}</span>
                    </div>

                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteSession(s.id);
                      }}
                      className="opacity-0 group-hover:opacity-100 p-1 text-[var(--text-dim)] hover:text-[#0EA5E9] transition-opacity"
                      title="Delete"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="p-3 border-t border-[var(--border-subtle)] space-y-0.5">
          <button
            type="button"
            onClick={() => setIsSettingsOpen(true)}
            className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-[var(--bg-surface)] transition-colors"
          >
            <Settings className="w-3.5 h-3.5" />
            <span>Settings</span>
          </button>

          <button
            type="button"
            onClick={() => setIsHelpOpen(true)}
            className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-[var(--bg-surface)] transition-colors"
          >
            <HelpCircle className="w-3.5 h-3.5" />
            <span>Mission Info</span>
          </button>
        </div>
      </aside>

      {/* Settings Modal */}
      <Modal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        title="System Settings"
      >
        <div className="space-y-3 text-xs text-[var(--text-main)]">
          <div className="p-3 rounded-xl bg-[var(--bg-surface)] border border-[var(--border-subtle)]">
            <h4 className="font-semibold text-[var(--text-main)] mb-1">
              Inference Mode
            </h4>
            <p className="text-[var(--text-muted)] mb-2">
              Set via <code className="text-[#0EA5E9] font-mono">INFERENCE_MODE</code> in backend configuration.
            </p>
            <span className="inline-block px-2 py-0.5 rounded bg-[var(--bg-card)] border border-[var(--border-subtle)] text-[var(--text-main)] font-mono text-[11px]">
              Active: {systemHealth?.inference_mode || 'NEURAL'}
            </span>
          </div>
        </div>
      </Modal>

      {/* Help Modal */}
      <Modal
        isOpen={isHelpOpen}
        onClose={() => setIsHelpOpen(false)}
        title="About SatQuery AI"
      >
        <div className="space-y-2 text-xs text-[var(--text-main)] leading-relaxed">
          <p>
            SatQuery AI is an interactive vision-language assistant for remote sensing image analysis, developed for ISRO / Space Applications Centre (SAC).
          </p>
          <p className="text-[var(--text-muted)]">
            Supports Optical multispectral rasters, microwave SAR, bi-temporal change detection, and text-guided visual grounding.
          </p>
        </div>
      </Modal>
    </>
  );
};
