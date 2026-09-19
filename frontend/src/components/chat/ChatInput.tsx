import React, { useState, useRef, useEffect } from 'react';
import {
  Paperclip,
  Satellite,
  ArrowUp,
  Square,
  X,
  Layers,
  UploadCloud,
} from 'lucide-react';
import { useChat } from '../../context/ChatContext';
import { SamplePickerModal } from '../input/SamplePickerModal';
import { SatQueryAPI } from '../../services/api';

export const ChatInput: React.FC = () => {
  const {
    activeImages,
    removeImage,
    attachImage,
    submitQuery,
    isProcessing,
    cancelProcessing,
    pendingQueryText,
    setPendingQueryText,
  } = useChat();

  const [text, setText] = useState('');
  const [isSamplePickerOpen, setIsSamplePickerOpen] = useState(false);
  const [isUploadingInline, setIsUploadingInline] = useState(false);
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const hiddenFileInputRef = useRef<HTMLInputElement | null>(null);

  const showError = (msg: string) => {
    setUploadError(msg);
    setTimeout(() => {
      setUploadError((current) => (current === msg ? null : current));
    }, 4500);
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.min(scrollHeight, 160)}px`;
    }
  }, [text]);

  // Populate textarea when a sample/quick-prompt sets pending text
  useEffect(() => {
    if (pendingQueryText) {
      setText(pendingQueryText);
      setPendingQueryText('');
      // Focus so user can immediately edit or hit Enter
      setTimeout(() => textareaRef.current?.focus(), 50);
    }
  }, [pendingQueryText, setPendingQueryText]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    if (!text.trim() || isProcessing) return;
    submitQuery(text.trim());
    setText('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const uploadFile = async (file: File) => {
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    const accepted = ['.tif', '.tiff', '.geotiff'];
    if (!accepted.includes(ext)) {
      showError('Only GeoTIFF satellite rasters (.tif, .tiff, .geotiff) are supported. PNG images are not permitted.');
      return;
    }
    if (activeImages.length >= 2) {
      showError('Maximum 2 images can be attached simultaneously. Replacing second image slot.');
    }
    setIsUploadingInline(true);
    try {
      const uploaded = await SatQueryAPI.uploadImage(file);
      attachImage(uploaded);
    } catch (err: any) {
      showError(err.message || 'Upload failed. Ensure backend is running.');
    } finally {
      setIsUploadingInline(false);
    }
  };

  const handleInlineFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (isUploadingInline) return;
    if (e.target.files && e.target.files.length > 0) {
      const files = Array.from(e.target.files).slice(0, 2);
      for (const file of files) {
        await uploadFile(file);
      }
      if (e.target.files.length > 2) {
        showError('Attached first 2 files (maximum 2 images allowed).');
      }
      e.target.value = '';
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!isUploadingInline) {
      setIsDraggingOver(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingOver(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingOver(false);
    if (isUploadingInline) return;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const files = Array.from(e.dataTransfer.files).slice(0, 2);
      for (const file of files) {
        await uploadFile(file);
      }
      if (e.dataTransfer.files.length > 2) {
        showError('Attached first 2 files (maximum 2 images allowed).');
      }
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="w-full max-w-3xl mx-auto px-4 pb-5 pt-1 select-none">
      {/* Inline Upload / Validation Notice */}
      {uploadError && (
        <div className="mb-2.5 p-2 px-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-700 dark:text-amber-400 text-xs flex items-center justify-between animate-fadeIn">
          <span>{uploadError}</span>
          <button
            type="button"
            onClick={() => setUploadError(null)}
            className="text-xs hover:text-amber-900 dark:hover:text-amber-200 cursor-pointer p-0.5"
            title="Dismiss"
          >
            ✕
          </button>
        </div>
      )}
      {/* Attached Imagery Chips */}
      {activeImages.length > 0 && (
        <div className="mb-2.5 flex flex-wrap items-center gap-2 px-1 animate-fadeIn">
          {activeImages.map((img) => (
            <div
              key={img.file_id}
              className="flex items-center gap-2.5 bg-[var(--bg-panel)] border border-[var(--border-subtle)] rounded-xl p-1.5 pr-3 shadow-subtle transition-all"
            >
              {img.thumbnail_url ? (
                <img
                  src={img.thumbnail_url}
                  alt={img.filename}
                  className="w-7 h-7 rounded-lg object-cover border border-[var(--border-subtle)] bg-[var(--bg-app)]"
                />
              ) : (
                <div className="w-7 h-7 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)] flex items-center justify-center text-[var(--text-muted)]">
                  <Satellite className="w-3.5 h-3.5" />
                </div>
              )}

              <div className="flex flex-col">
                <div className="flex items-center gap-1.5">
                  <span className="text-xs font-medium text-[var(--text-main)] truncate max-w-[140px]">
                    {img.filename}
                  </span>
                  <span className="font-mono text-[10px] text-[#cc785c] bg-[#cc785c]/10 px-1.5 py-0.2 rounded border border-[#cc785c]/20">
                    {img.modality}
                  </span>
                </div>
                <span className="text-[10px] font-mono text-[var(--text-dim)]">
                  {img.width}×{img.height} • {formatFileSize(img.file_size_bytes)}
                </span>
              </div>

              <button
                type="button"
                onClick={() => removeImage(img.file_id)}
                className="ml-1 p-1 text-[var(--text-dim)] hover:text-[var(--text-main)] hover:bg-[var(--bg-surface)] rounded transition-colors cursor-pointer"
                title="Remove"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}

          {activeImages.length === 2 && (
            <span className="text-[11px] font-mono text-[#cc785c] bg-[#cc785c]/10 px-2 py-1 rounded-lg border border-[#cc785c]/20 flex items-center gap-1.5">
              <Layers className="w-3 h-3" /> Pair Ready for Fusion / Change Analysis
            </span>
          )}
        </div>
      )}

      {/* Claude Signature Floating Input Card with Drag & Drop */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative rounded-[26px] bg-[var(--bg-input)] border transition-all duration-200 shadow-claude dark:shadow-claudeDark ${
          isDraggingOver
            ? 'border-[#cc785c] ring-2 ring-[#cc785c]/25 bg-[#cc785c]/5'
            : 'border-[var(--border-subtle)] focus-within:border-[#cc785c]/60 focus-within:ring-2 focus-within:ring-[#cc785c]/15'
        }`}
      >
        {isDraggingOver && (
          <div className="absolute inset-0 rounded-[26px] bg-[var(--bg-card)]/90 backdrop-blur-sm z-20 flex items-center justify-center gap-2 text-xs font-medium text-[#cc785c]">
            <UploadCloud className="w-5 h-5 animate-bounce" />
            <span>Drop GeoTIFF raster to attach</span>
          </div>
        )}

        {/* Text Input */}
        <div className="p-3.5 pb-1">
          <textarea
            ref={textareaRef}
            rows={1}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              activeImages.length === 0
                ? 'Reply to SatQuery AI or attach a GeoTIFF...'
                : activeImages.length === 1
                ? 'Ask a question about this satellite scene...'
                : 'Ask to compare or fuse these two scenes...'
            }
            className="w-full bg-transparent text-[14.5px] text-[var(--text-main)] placeholder-[var(--text-dim)] resize-none outline-none leading-relaxed min-h-[46px] max-h-[160px] font-normal"
          />
        </div>

        {/* Minimal Bottom Bar */}
        <div className="flex items-center justify-between px-3.5 py-2.5 border-t border-[var(--border-subtle)]/60">
          <div className="flex items-center gap-1.5">
            {/* Upload File */}
            <button
              type="button"
              onClick={() => hiddenFileInputRef.current?.click()}
              disabled={isUploadingInline || activeImages.length >= 2}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-[var(--bg-surface)] disabled:opacity-40 transition-colors cursor-pointer"
              title="Upload satellite raster as GeoTIFF or TIFF (.tif, .tiff)."
            >
              <Paperclip className="w-4 h-4 text-[var(--text-dim)]" />
              <span className="hidden sm:inline">
                {isUploadingInline ? 'Uploading...' : 'Attach GeoTIFF'}
              </span>
            </button>

            <input
              ref={hiddenFileInputRef}
              type="file"
              accept=".tif,.tiff,.geotiff"
              onChange={handleInlineFileSelect}
              className="hidden"
            />

            {/* Pre-packaged Server Samples */}
            <button
              type="button"
              onClick={() => setIsSamplePickerOpen(true)}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-[var(--bg-surface)] transition-colors cursor-pointer"
              title="Load ISRO Cartosat / RISAT sample rasters"
            >
              <Satellite className="w-4 h-4 text-[var(--text-dim)]" />
              <span className="hidden sm:inline">ISRO Samples</span>
            </button>
          </div>

          <div>
            {isProcessing ? (
              <button
                type="button"
                onClick={cancelProcessing}
                className="w-8 h-8 rounded-full bg-[#cc785c] hover:bg-[#b8674d] text-white flex items-center justify-center transition-colors shadow-sm cursor-pointer"
                title="Stop generation"
              >
                <Square className="w-3.5 h-3.5 fill-current" />
              </button>
            ) : (
              <button
                type="button"
                onClick={handleSend}
                disabled={!text.trim()}
                className={`w-8 h-8 rounded-full flex items-center justify-center transition-all ${
                  text.trim()
                    ? 'bg-[#cc785c] hover:bg-[#b8674d] text-white shadow-sm cursor-pointer'
                    : 'bg-[var(--border-subtle)] text-[var(--text-dim)] opacity-60 cursor-not-allowed'
                }`}
                title="Send message"
              >
                <ArrowUp className="w-4 h-4 stroke-[2.5]" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* ISRO Samples Modal */}
      <SamplePickerModal
        isOpen={isSamplePickerOpen}
        onClose={() => setIsSamplePickerOpen(false)}
      />
    </div>
  );
};
