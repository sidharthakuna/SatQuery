import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';


const CodeBlock: React.FC<{ code: string; lang?: string }> = ({ code, lang }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="my-3 rounded-xl overflow-hidden border border-[var(--border-subtle)] bg-[var(--bg-card)] shadow-subtle">
      <div className="flex items-center justify-between px-3.5 py-1.5 bg-[var(--bg-surface)] text-[11px] font-mono text-[var(--text-muted)] border-b border-[var(--border-subtle)]">
        <span className="uppercase tracking-wider font-semibold text-[10px] text-[#cc785c]">
          {lang || 'code'}
        </span>
        <button
          type="button"
          onClick={handleCopy}
          className="flex items-center gap-1 text-[11px] text-[var(--text-muted)] hover:text-[var(--text-main)] transition-colors cursor-pointer"
          title="Copy code"
        >
          {copied ? (
            <>
              <Check className="w-3 h-3 text-[#cc785c]" />
              <span className="text-[#cc785c]">Copied</span>
            </>
          ) : (
            <>
              <Copy className="w-3 h-3" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <pre className="p-3.5 font-mono text-[12.5px] text-[var(--text-main)] overflow-x-auto leading-relaxed bg-[var(--bg-app)]/60">
        <code>{code}</code>
      </pre>
    </div>
  );
};

// Helper to format inline tokens: bold (**text**), inline code (`code`), italic (*text*)
const renderInline = (text: string): React.ReactNode[] => {
  const parts: React.ReactNode[] = [];
  const regex = /(\*\*.*?\*\*|`.*?`|\*[^*]+?\*)/g;
  const segments = text.split(regex);

  segments.forEach((seg, idx) => {
    if (!seg) return;
    if (seg.startsWith('**') && seg.endsWith('**') && seg.length >= 4) {
      parts.push(
        <strong key={idx} className="font-semibold text-[var(--text-main)]">
          {seg.slice(2, -2)}
        </strong>
      );
    } else if (seg.startsWith('`') && seg.endsWith('`') && seg.length >= 2) {
      parts.push(
        <code
          key={idx}
          className="px-1.5 py-0.5 rounded bg-[var(--bg-surface)] border border-[var(--border-subtle)] text-[#cc785c] font-mono text-[12px]"
        >
          {seg.slice(1, -1)}
        </code>
      );
    } else if (
      seg.startsWith('*') && seg.endsWith('*') && seg.length >= 2 &&
      !seg.startsWith('**') // guard: don't match bold as italic
    ) {
      parts.push(
        <em key={idx} className="italic text-[var(--text-main)]">
          {seg.slice(1, -1)}
        </em>
      );
    } else {
      parts.push(seg);
    }
  });

  return parts;
};

export interface MarkdownContentProps {
  content: string;
  isStreaming?: boolean;
  compact?: boolean;
  className?: string;
}

export const MarkdownContent: React.FC<MarkdownContentProps> = ({
  content,
  isStreaming = false,
  compact = false,
  className = '',
}) => {
  const rawLines = content.split('\n');
  const elements: React.ReactNode[] = [];

  let i = 0;
  while (i < rawLines.length) {
    const line = rawLines[i];
    const trimmed = line.trim();

    // 1. Skip empty lines
    if (!trimmed) {
      i++;
      continue;
    }

    // 2. Fenced code block
    if (trimmed.startsWith('```')) {
      const lang = trimmed.slice(3).trim();
      const codeLines: string[] = [];
      i++;
      while (i < rawLines.length && !rawLines[i].trim().startsWith('```')) {
        codeLines.push(rawLines[i]);
        i++;
      }
      if (i < rawLines.length) i++; // skip closing ```
      elements.push(
        <CodeBlock
          key={`code-${elements.length}`}
          code={codeLines.join('\n')}
          lang={lang}
        />
      );
      continue;
    }

    // 3. Table parsing (| col | col |)
    if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
      const tableLines: string[] = [];
      while (i < rawLines.length && rawLines[i].trim().startsWith('|') && rawLines[i].trim().endsWith('|')) {
        tableLines.push(rawLines[i].trim());
        i++;
      }

      if (tableLines.length >= 2) {
        const headerRow = tableLines[0]
          .slice(1, -1)
          .split('|')
          .map((c) => c.trim());

        // Check if row 1 is delimiter (| :--- | :--- |)
        let dataStartIdx = 1;
        if (tableLines.length > 1 && tableLines[1].includes('---')) {
          dataStartIdx = 2;
        }

        const bodyRows = tableLines.slice(dataStartIdx).map((r) =>
          r
            .slice(1, -1)
            .split('|')
            .map((c) => c.trim())
        );

        elements.push(
          <div key={`table-${elements.length}`} className="overflow-x-auto my-2.5 rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-card)] shadow-subtle">
            <table className="w-full border-collapse text-left">
              <thead className="bg-[var(--bg-surface)] border-b border-[var(--border-subtle)] text-[#cc785c] font-semibold text-[10.5px] uppercase tracking-wider font-mono">
                <tr>
                  {headerRow.map((h, hIdx) => (
                    <th key={hIdx} className={`${compact ? 'px-3 py-1.5' : 'px-3.5 py-2'} font-semibold border-r border-[var(--border-subtle)] last:border-r-0`}>
                      {renderInline(h)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border-subtle)]">
                {bodyRows.map((row, rIdx) => (
                  <tr key={rIdx} className="hover:bg-[var(--bg-surface)]/60 transition-colors">
                    {row.map((cell, cIdx) => (
                      <td key={cIdx} className={`${compact ? 'px-3 py-1.5 text-[11.5px]' : 'px-3.5 py-2 text-[12.5px]'} text-[var(--text-main)] border-r border-[var(--border-subtle)] last:border-r-0 align-top`}>
                        {renderInline(cell)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
        continue;
      }
    }

    // 4. Headers
    if (trimmed.startsWith('### ')) {
      elements.push(
        <h3 key={`h3-${elements.length}`} className={`${compact ? 'text-[12.5px] font-mono uppercase tracking-wider text-[#cc785c] font-semibold mt-3 mb-1.5' : 'text-[14.5px] font-semibold tracking-tight text-[var(--text-main)] mt-3.5 mb-1.5 font-sans'} flex items-center gap-1.5`}>
          <span className="w-1.5 h-1.5 rounded-full bg-[#cc785c] inline-block shrink-0" />
          <span>{renderInline(trimmed.slice(4))}</span>
        </h3>
      );
      i++;
      continue;
    }

    if (trimmed.startsWith('## ')) {
      elements.push(
        <h2 key={`h2-${elements.length}`} className={`${compact ? 'text-[13.5px] font-semibold' : 'text-base font-semibold'} tracking-tight text-[var(--text-main)] mt-3.5 mb-2 font-serif`}>
          {renderInline(trimmed.slice(3))}
        </h2>
      );
      i++;
      continue;
    }

    if (trimmed.startsWith('# ')) {
      elements.push(
        <h1 key={`h1-${elements.length}`} className={`${compact ? 'text-[15px]' : 'text-lg'} font-semibold tracking-tight text-[var(--text-main)] mt-4 mb-2 font-serif border-b border-[var(--border-subtle)] pb-1`}>
          {renderInline(trimmed.slice(2))}
        </h1>
      );
      i++;
      continue;
    }

    // 5. Horizontal divider
    if (trimmed === '---' || trimmed === '***' || trimmed === '___') {
      elements.push(<hr key={`hr-${elements.length}`} className="my-3 border-[var(--border-subtle)]" />);
      i++;
      continue;
    }

    // 6. Blockquote
    if (trimmed.startsWith('> ')) {
      elements.push(
        <blockquote
          key={`quote-${elements.length}`}
          className="border-l-2 border-[#cc785c] pl-3 py-1 my-2 bg-[#cc785c]/5 rounded-r-md text-[12px] italic text-[var(--text-muted)]"
        >
          {renderInline(trimmed.slice(2))}
        </blockquote>
      );
      i++;
      continue;
    }

    // 7. Unordered list
    if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      const listItems: string[] = [];
      while (i < rawLines.length && (rawLines[i].trim().startsWith('- ') || rawLines[i].trim().startsWith('* '))) {
        listItems.push(rawLines[i].trim().slice(2));
        i++;
      }
      elements.push(
        <ul key={`ul-${elements.length}`} className={`list-none my-2 space-y-1.5 text-[var(--text-main)] ${compact ? 'text-[12px]' : 'text-[13.5px]'}`}>
          {listItems.map((item, itemIdx) => (
            <li key={itemIdx} className="leading-[1.6] flex items-start gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-[#cc785c] mt-1.5 shrink-0" />
              <span className="flex-1">{renderInline(item)}</span>
            </li>
          ))}
        </ul>
      );
      continue;
    }

    // 8. Ordered list
    if (/^\d+\.\s/.test(trimmed)) {
      const listItems: string[] = [];
      while (i < rawLines.length && /^\d+\.\s/.test(rawLines[i].trim())) {
        listItems.push(rawLines[i].trim().replace(/^\d+\.\s/, ''));
        i++;
      }
      elements.push(
        <ol key={`ol-${elements.length}`} className={`list-none my-2 space-y-1.5 text-[var(--text-main)] ${compact ? 'text-[12px]' : 'text-[13.5px]'}`}>
          {listItems.map((item, itemIdx) => (
            <li key={itemIdx} className="leading-[1.6] flex items-start gap-2 font-normal">
              <span className="w-4 h-4 rounded-full bg-[#cc785c]/10 text-[#cc785c] border border-[#cc785c]/25 font-mono text-[9.5px] font-bold flex items-center justify-center shrink-0 mt-0.5">
                {itemIdx + 1}
              </span>
              <span className="flex-1">{renderInline(item)}</span>
            </li>
          ))}
        </ol>
      );
      continue;
    }

    // 9. Standard paragraph
    const isLastLine = i === rawLines.length - 1 || rawLines.slice(i + 1).every((l) => !l.trim());
    elements.push(
      <p key={`p-${elements.length}`} className={`my-1.5 leading-[1.65] text-[var(--text-main)] font-normal ${compact ? 'text-[12px]' : 'text-[13.5px]'}`}>
        {renderInline(trimmed)}
        {/* Blinking streaming cursor on the very last paragraph while still generating */}
        {isStreaming && isLastLine && (
          <span
            className="inline-block w-[2px] h-[1em] bg-[#cc785c] ml-0.5 align-middle animate-pulse"
            aria-hidden="true"
          />
        )}
      </p>
    );
    i++;
  }

  return <div className={`space-y-1 ${className}`}>{elements}</div>;
};
