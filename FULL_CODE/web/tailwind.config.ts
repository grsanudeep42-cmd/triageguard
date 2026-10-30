import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "var(--color-paper)",
        surface: "var(--color-surface)",
        line: "var(--color-line)",
        "line-strong": "var(--color-line-strong)",
        ink: "var(--color-ink)",
        "ink-2": "var(--color-ink-2)",
        "ink-3": "var(--color-ink-3)",
        brand: "var(--color-brand)",
        "brand-hover": "var(--color-brand-hover)",
        "brand-tint": "var(--color-brand-tint)",
        l1: {
          solid: "var(--color-l1-solid)",
          tint: "var(--color-l1-tint)",
          border: "var(--color-l1-border)",
          text: "var(--color-l1-text)",
        },
        l2: {
          solid: "var(--color-l2-solid)",
          tint: "var(--color-l2-tint)",
          border: "var(--color-l2-border)",
          text: "var(--color-l2-text)",
        },
        l3: {
          solid: "var(--color-l3-solid)",
          tint: "var(--color-l3-tint)",
          border: "var(--color-l3-border)",
          text: "var(--color-l3-text)",
        },
        l4: {
          solid: "var(--color-l4-solid)",
          tint: "var(--color-l4-tint)",
          border: "var(--color-l4-border)",
          text: "var(--color-l4-text)",
        },
        l5: {
          solid: "var(--color-l5-solid)",
          tint: "var(--color-l5-tint)",
          border: "var(--color-l5-border)",
          text: "var(--color-l5-text)",
        },
        nurse: {
          solid: "var(--color-nurse-solid)",
          tint: "var(--color-nurse-tint)",
          border: "var(--color-nurse-border)",
          text: "var(--color-nurse-text)",
        },
        vital: {
          normal: "var(--color-vital-normal)",
          abnormal: "var(--color-vital-abnormal)",
          critical: "var(--color-vital-critical)",
          invalid: "var(--color-vital-invalid)",
        },
      },
      fontFamily: {
        sans: ['"Public Sans"', "system-ui", "-apple-system", "sans-serif"],
      },
      fontSize: {
        caption: ["13px", "1.4"],
        body: ["15px", "1.55"],
        h3: ["18px", { lineHeight: "1.4", fontWeight: "600" }],
        h2: ["22px", { lineHeight: "1.3", letterSpacing: "-0.01em", fontWeight: "600" }],
        h1: ["28px", { lineHeight: "1.25", letterSpacing: "-0.01em", fontWeight: "600" }],
        verdict: ["120px", { lineHeight: "1", fontWeight: "800" }],
      },
      boxShadow: {
        raised: "0 1px 0 rgba(18,32,43,.04), 0 8px 24px -8px rgba(18,32,43,.18)",
      },
      borderRadius: {
        verdict: "20px",
        panel: "14px",
        tile: "12px",
        btn: "10px",
        input: "8px",
        pill: "9999px",
      },
      spacing: {
        "4px": "4px",
        "8px": "8px",
        "12px": "12px",
        "16px": "16px",
        "20px": "20px",
        "24px": "24px",
        "32px": "32px",
        "40px": "40px",
        "48px": "48px",
        "64px": "64px",
      },
    },
  },
  plugins: [],
};

export default config;
