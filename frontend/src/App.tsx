import React from 'react';
import { ThemeProvider } from './context/ThemeContext';
import { ChatProvider } from './context/ChatContext';
import { Header } from './components/layout/Header';
import { Sidebar } from './components/layout/Sidebar';
import { AnalysisPanel } from './components/layout/AnalysisPanel';
import { ChatContainer } from './components/chat/ChatContainer';
import { ChatInput } from './components/chat/ChatInput';
import { PdfBriefingModal } from './components/report/PdfBriefingModal';
import { useChat } from './context/ChatContext';
import { useTheme } from './context/ThemeContext';
import AeroShards from './components/ui/AeroShards';

const AppContent: React.FC = () => {
  const { currentSession, isProcessing, isPdfModalOpen, closePdfModal, pdfModalResult, pdfModalImage, activeImages } = useChat();
  const { theme } = useTheme();
  const messages = currentSession?.messages || [];
  const isChatEmpty = messages.length === 0 && !isProcessing && activeImages.length === 0;
  
  const isDarkMode = theme === 'dark';

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[var(--bg-app)] text-[var(--text-main)] font-sans relative">
      {/* Left Sidebar (Claude Style) */}
      <Sidebar />

      {/* Main Center Workspace */}
      <div className={`flex-1 flex flex-col h-full min-w-0 overflow-hidden relative ${isChatEmpty ? 'bg-transparent' : 'bg-[var(--bg-app)]'}`}>
        
        {isChatEmpty && (
          <div style={{ width: '100%', height: '100%', position: 'absolute', top: 0, left: 0, zIndex: 0 }} className="pointer-events-auto">
            <AeroShards
              backgroundColor={isDarkMode ? "#000000" : "#fdfdfd"}
              shardColor={isDarkMode ? "#e80000" : "#79ff8f"}
              accentColor={isDarkMode ? "#0dfff3" : "#ff00fe"}
              placement="full"
              flow="stream"
              material="pearl"
              detail="balanced"
              effect="none"
              scale={1}
              spread={1.1}
              depth={1.05}
              speed={1}
              spin={1}
              interaction="repel"
              density={1.5}
              shardSize={1.1}
              stretch={1}
              turbulence={1}
              glow={1.4}
              edgeSoftness={2}
              bloom={1.8}
              grain={0.05}
              chromaticAberration={0.0075}
              transitionDuration={1.4}
              interactionRadius={1.5}
              interactionStrength={0.25}
              rippleIntensity={0.3}
              holdToGather={true}
              paused={false}
              onError={() => {}}
            />
          </div>
        )}

        {/* Content wrapper with z-10 so it's above the background */}
        <div className="relative z-10 flex flex-col h-full w-full pointer-events-none">
          <div className="pointer-events-auto">
            <Header />
          </div>

          {/* Scrollable Chat Area */}
          <div className="flex-1 overflow-hidden pointer-events-auto">
            <ChatContainer />
          </div>

          {/* Universal Input Dock */}
          <div className="pointer-events-auto">
            <ChatInput />
          </div>
        </div>
      </div>

      {/* Right Analysis Panel */}
      <AnalysisPanel />

      {/* Executive Mission Briefing PDF Modal */}
      <PdfBriefingModal
        isOpen={isPdfModalOpen}
        onClose={closePdfModal}
        result={pdfModalResult}
        activeImage={pdfModalImage}
      />
    </div>
  );
};

export default function App() {
  return (
    <ThemeProvider>
      <ChatProvider>
        <AppContent />
      </ChatProvider>
    </ThemeProvider>
  );
}
