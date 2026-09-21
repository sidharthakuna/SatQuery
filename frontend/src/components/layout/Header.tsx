import React, { useState, useRef, useEffect } from 'react';
import {
  Menu,
  Sun,
  Moon,
  PanelRight,
  Edit2,
  Check,
  Cpu,
  FileText,
} from 'lucide-react';
import { useChat } from '../../context/ChatContext';
import { useTheme } from '../../context/ThemeContext';
import { SatQueryLogo } from '../ui/SatQueryLogo';

export const Header: React.FC = () => {
  const {
    systemHealth,
    isSidebarOpen,
    setSidebarOpen,
    isAnalysisPanelOpen,
    setAnalysisPanelOpen,
    currentSession,
    renameSession,
    activeAnalysisResult,
    activeImageForAnalysis,
    activeImages,
    openPdfModal,
  } = useChat();
  const { theme, toggleTheme } = useTheme();

  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [titleInput, setTitleInput] = useState('');
  const titleInputRef = useRef<HTMLInputElement | null>(null);

  const startEditing = () => {
    setTitleInput(currentSession?.title || 'New Analysis');
    setIsEditingTitle(true);
  };

  useEffect(() => {
    if (isEditingTitle && titleInputRef.current) {
      titleInputRef.current.focus();
      titleInputRef.current.select();
    }
  }, [isEditingTitle]);

  const handleTitleSubmit = () => {
    if (currentSession && titleInput.trim()) {
      renameSession(currentSession.id, titleInput.trim());
    }
    setIsEditingTitle(false);
  };

  const isHealthy = !!systemHealth;
  const hasTelemetry = !!activeAnalysisResult;

  return (
    <header className="h-13 border-b border-[var(--border-subtle)] bg-[var(--bg-app)]/90 backdrop-blur-md flex items-center justify-between px-4 z-30 select-none transition-colors">
      {/* ── Left Section: Sidebar Toggle & Branding ── */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => setSidebarOpen(!isSidebarOpen)}
          className="p-1.5 text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-[var(--bg-surface)] rounded-lg transition-colors cursor-pointer"
          title={isSidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
        >
          <Menu className="w-4 h-4" />
        </button>

        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-md bg-[var(--bg-panel)] border border-[var(--border-subtle)] flex items-center justify-center">
            <SatQueryLogo size={16} />
          </div>
          <div className="flex flex-col">
            <span className="font-semibold text-xs text-[var(--text-main)] tracking-tight leading-none">
              SatQuery AI
            </span>
            <span className="text-[9px] font-mono text-[var(--text-dim)] leading-tight mt-0.5 hidden sm:inline">
              ISRO SAC • Geospatial Intelligence
            </span>
          </div>
        </div>
      </div>

      {/* ── Center Section: Active Session Title with Inline Rename ── */}
      <div className="hidden md:flex items-center gap-2 max-w-sm lg:max-w-md">
        {isEditingTitle ? (
          <div className="flex items-center gap-1.5 bg-[var(--bg-panel)] border border-[#0EA5E9]/40 rounded-lg px-2 py-0.5 shadow-sm">
            <input
              ref={titleInputRef}
              type="text"
              value={titleInput}
              onChange={(e) => setTitleInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleTitleSubmit();
                if (e.key === 'Escape') setIsEditingTitle(false);
              }}
              onBlur={handleTitleSubmit}
              className="text-xs text-[var(--text-main)] bg-transparent outline-none w-48 font-medium"
              maxLength={40}
            />
            <button
              type="button"
              onClick={handleTitleSubmit}
              className="p-0.5 text-[#0EA5E9] hover:bg-[#0EA5E9]/10 rounded"
            >
              <Check className="w-3 h-3" />
            </button>
          </div>
        ) : (
          <div
            onClick={startEditing}
            className="group flex items-center gap-1.5 px-2.5 py-1 rounded-lg hover:bg-[var(--bg-surface)] transition-colors cursor-pointer"
            title="Click to rename session"
          >
            <span className="text-xs text-[var(--text-muted)] group-hover:text-[var(--text-main)] truncate font-normal transition-colors max-w-[220px]">
              {currentSession?.title || 'New Analysis'}
            </span>
            <Edit2 className="w-2.5 h-2.5 text-[var(--text-dim)] group-hover:text-[#0EA5E9] opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>
        )}
      </div>

      {/* ── Right Section: Backend Status, Theme, Telemetry Drawer ── */}
      <div className="flex items-center gap-2">
        {/* Real-time Telemetry Status Pill */}
        <div
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[var(--bg-panel)] border border-[var(--border-subtle)] text-[11px] font-mono shadow-subtle"
          title={isHealthy ? `Backend Connected: ${systemHealth?.inference_mode || 'ONLINE'}` : 'Connecting to backend...'}
        >
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              isHealthy ? 'bg-[#0EA5E9] animate-pulse' : 'bg-[var(--text-dim)]'
            }`}
          />
          <span className="text-[var(--text-muted)] text-[10.5px]">
            {isHealthy ? (systemHealth?.inference_mode || 'ONLINE') : 'OFFLINE'}
          </span>
          {isHealthy && (
            <Cpu className="w-3 h-3 text-[var(--text-dim)] ml-0.5 hidden lg:inline" />
          )}
        </div>



        {/* Executive PDF Briefing Dossier Action Button */}
        {activeAnalysisResult && (
          <button
            type="button"
            onClick={() => openPdfModal(activeAnalysisResult, activeImageForAnalysis || activeImages[0])}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#0EA5E9] hover:bg-[#0284C7] text-white text-[11px] font-medium transition-colors shadow-sm cursor-pointer"
            title="Open Executive Mission Briefing Dossier (PDF)"
          >
            <FileText className="w-3.5 h-3.5 text-white" />
            <span className="hidden sm:inline">Executive PDF</span>
          </button>
        )}

        {/* Theme Toggle (Claude Sun / Moon with smooth fade) */}
        <button
          type="button"
          onClick={toggleTheme}
          className="p-1.5 text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-[var(--bg-surface)] rounded-lg transition-colors cursor-pointer"
          title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} mode`}
        >
          {theme === 'dark' ? (
            <Sun className="w-4 h-4 text-[#0EA5E9]" />
          ) : (
            <Moon className="w-4 h-4 text-[var(--text-main)]" />
          )}
        </button>

        {/* Toggle Right Telemetry Panel */}
        <button
          type="button"
          onClick={() => setAnalysisPanelOpen(!isAnalysisPanelOpen)}
          className={`p-1.5 rounded-lg transition-all cursor-pointer relative ${
            isAnalysisPanelOpen
              ? 'bg-[var(--bg-surface)] text-[var(--text-main)] border border-[var(--border-subtle)] shadow-subtle'
              : 'text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-[var(--bg-surface)]'
          }`}
          title="Toggle Telemetry Panel"
        >
          <PanelRight className="w-4 h-4" />
          {hasTelemetry && !isAnalysisPanelOpen && (
            <span className="absolute top-1 right-1 w-1.5 h-1.5 rounded-full bg-[#0EA5E9] ring-2 ring-[var(--bg-app)]" />
          )}
        </button>
      </div>

    </header>
  );
};
