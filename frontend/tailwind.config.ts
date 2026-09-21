import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        claude: {
          bg: "var(--bg-app)",
          panel: "var(--bg-panel)",
          card: "var(--bg-card)",
          surface: "var(--bg-surface)",
          input: "var(--bg-input)",
          bubble: "var(--bg-user-bubble)",
          border: "var(--border-subtle)",
          borderHover: "var(--border-hover)",
          borderActive: "var(--border-active)",
          text: "var(--text-main)",
          textMuted: "var(--text-muted)",
          textDim: "var(--text-dim)",
          terracotta: "var(--accent-primary)",
          terracottaHover: "var(--accent-hover)",
          terracottaTint: "var(--accent-tint)",
        },
        chat: {
          bg: "var(--bg-app)",
          panel: "var(--bg-panel)",
          card: "var(--bg-card)",
          surface: "var(--bg-surface)",
          input: "var(--bg-input)",
          border: "var(--border-subtle)",
          borderHover: "var(--border-hover)",
          text: "var(--text-main)",
          textMuted: "var(--text-muted)",
          accent: "var(--accent-primary)",
        },
      },
      fontFamily: {
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
        serif: ["Newsreader", "Georgia", "serif"],
        editorial: ["Newsreader", "Georgia", "serif"],
        mono: ["JetBrains Mono", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: {
        subtle: "0 1px 2px 0 rgba(0, 0, 0, 0.04)",
        card: "0 2px 8px -1px rgba(0, 0, 0, 0.08)",
        claude: "0 4px 20px -2px rgba(0, 0, 0, 0.06)",
        claudeDark: "0 4px 24px -2px rgba(0, 0, 0, 0.35)",
      },
    },
  },
  plugins: [],
};

export default config;
