import React, { useEffect, useRef, useState, useCallback } from 'react';
import { ArrowDown } from 'lucide-react';
import { useChat } from '../../context/ChatContext';
import { ChatEmptyState } from './ChatEmptyState';
import { ChatMessageItem } from './ChatMessageItem';

export const ChatContainer: React.FC = () => {
  const { currentSession, isProcessing } = useChat();
  const bottomAnchorRef = useRef<HTMLDivElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [showScrollBottom, setShowScrollBottom] = useState(false);

  const messages = currentSession?.messages || [];

  // Check if user is scrolled up
  const handleScroll = useCallback(() => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const isUp = scrollHeight - scrollTop - clientHeight > 160;
    setShowScrollBottom(isUp);
  }, []);

  // Auto-scroll to bottom on new messages, streaming content, or step updates if near bottom
  const lastMsg = messages[messages.length - 1];
  const lastMsgContentLength = lastMsg?.content?.length || 0;
  const lastMsgIsStreaming = lastMsg?.isStreaming;
  const lastStepCount = lastMsg?.streamingSteps?.length || 0;

  useEffect(() => {
    if (!showScrollBottom) {
      bottomAnchorRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages.length, lastMsgContentLength, lastMsgIsStreaming, lastStepCount, isProcessing, showScrollBottom]);

  const scrollToBottom = () => {
    bottomAnchorRef.current?.scrollIntoView({ behavior: 'smooth' });
    setShowScrollBottom(false);
  };

  if (messages.length === 0) {
    return <ChatEmptyState />;
  }

  return (
    <div
      ref={containerRef}
      onScroll={handleScroll}
      className="flex-1 overflow-y-auto px-2 sm:px-4 py-6 scroll-smooth relative"
    >
      <div className="max-w-4xl mx-auto space-y-4">
        {messages.map((msg) => (
          <ChatMessageItem key={msg.id} message={msg} />
        ))}
        <div ref={bottomAnchorRef} className="h-4" />
      </div>

      {/* Floating Scroll to Bottom Button */}
      {showScrollBottom && (
        <div className="sticky bottom-4 flex justify-end pr-4 pointer-events-none z-30">
          <button
            type="button"
            onClick={scrollToBottom}
            className="pointer-events-auto p-2.5 rounded-full bg-[var(--bg-card)] border border-[var(--border-subtle)] text-[var(--text-main)] hover:text-[#cc785c] shadow-claude dark:shadow-claudeDark transition-all duration-200 hover:scale-105 flex items-center justify-center cursor-pointer group relative"
            title="Scroll to latest response"
          >
            <ArrowDown className="w-4 h-4 text-[var(--text-muted)] group-hover:text-[#cc785c] transition-colors" />
            {isProcessing && (
              <span className="absolute top-0 right-0 w-2.5 h-2.5 bg-[#cc785c] rounded-full ring-2 ring-[var(--bg-card)] animate-ping" />
            )}
          </button>
        </div>
      )}
    </div>
  );
};
