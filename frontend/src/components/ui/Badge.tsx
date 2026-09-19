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
    success: 'bg-[#cc785c]/10 text-[#cc785c] border-[#cc785c]/25',
    warning: 'bg-[var(--bg-surface)] text-[var(--text-main)] border-[var(--border-subtle)]',
    purple: 'bg-[#cc785c]/10 text-[#cc785c] border-[#cc785c]/25',
    // Aliases mapped cleanly to unified minimalist palette
    cyan: 'bg-[var(--bg-panel)] text-[var(--text-muted)] border-[var(--border-subtle)]',
    emerald: 'bg-[#cc785c]/10 text-[#cc785c] border-[#cc785c]/25',
    amber: 'bg-[#cc785c]/10 text-[#cc785c] border-[#cc785c]/25',
    violet: 'bg-[#cc785c]/10 text-[#cc785c] border-[#cc785c]/25',
    rose: 'bg-[#cc785c]/10 text-[#cc785c] border-[#cc785c]/25',
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
