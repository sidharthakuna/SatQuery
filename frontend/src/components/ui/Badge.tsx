import React from 'react';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'default' | 'neutral' | 'success' | 'warning' | 'purple' | 'cyan' | 'emerald' | 'amber' | 'violet' | 'rose' | 'slate';
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'default',
  className = '',
}) => {
  const variantStyles = {
    default: 'bg-[var(--bg-surface)] text-[var(--text-main)] border-[var(--border-subtle)]',
    neutral: 'bg-[var(--bg-panel)] text-[var(--text-muted)] border-[var(--border-subtle)]',
    success: 'bg-[#0EA5E9]/10 text-[#0EA5E9] border-[#0EA5E9]/25',
    warning: 'bg-[var(--bg-surface)] text-[var(--text-main)] border-[var(--border-subtle)]',
    purple: 'bg-[#0EA5E9]/10 text-[#0EA5E9] border-[#0EA5E9]/25',
    // Aliases mapped cleanly to unified minimalist palette
    cyan: 'bg-[var(--bg-panel)] text-[var(--text-muted)] border-[var(--border-subtle)]',
    emerald: 'bg-[#0EA5E9]/10 text-[#0EA5E9] border-[#0EA5E9]/25',
    amber: 'bg-[#0EA5E9]/10 text-[#0EA5E9] border-[#0EA5E9]/25',
    violet: 'bg-[#0EA5E9]/10 text-[#0EA5E9] border-[#0EA5E9]/25',
    rose: 'bg-[#0EA5E9]/10 text-[#0EA5E9] border-[#0EA5E9]/25',
    slate: 'bg-[var(--bg-panel)] text-[var(--text-muted)] border-[var(--border-subtle)]',
  };

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-mono font-medium border ${variantStyles[variant]} ${className}`}
    >
      {children}
    </span>
  );
};
