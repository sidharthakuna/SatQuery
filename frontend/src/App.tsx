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

const AppContent: React.FC = () => {
  const { isPdfModalOpen, closePdfModal, pdfModalResult, pdfModalImage } = useChat();

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[var(--bg-app)] text-[var(--text-main)] font-sans">
      {/* Left Sidebar (Claude Style) */}
      <Sidebar />

      {/* Main Center Workspace */}
      <div className="flex-1 flex flex-col h-full min-w-0 overflow-hidden relative bg-[var(--bg-app)]">
        <Header />

        {/* Scrollable Chat Area */}
        <ChatContainer />

        {/* Universal Input Dock */}
        <ChatInput />
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
