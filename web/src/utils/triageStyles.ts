export interface TriageColorSet {
  solid: string;
  tint: string;
  border: string;
  text: string;
  name: string;
}

export const TRIAGE_COLORS: Record<number, TriageColorSet> = {
  1: {
    solid: "#B91C1C",
    tint: "#FEE2E2",
    border: "#FCA5A5",
    text: "#7F1D1D",
    name: "Resuscitation",
  },
  2: {
    solid: "#C2410C",
    tint: "#FFEDD5",
    border: "#FDBA74",
    text: "#7C2D12",
    name: "Emergent",
  },
  3: {
    solid: "#A16207",
    tint: "#FEF3C7",
    border: "#FCD34D",
    text: "#713F12",
    name: "Urgent",
  },
  4: {
    solid: "#15803D",
    tint: "#DCFCE7",
    border: "#86EFAC",
    text: "#14532D",
    name: "Less urgent",
  },
  5: {
    solid: "#1D4ED8",
    tint: "#DBEAFE",
    border: "#93C5FD",
    text: "#1E3A8A",
    name: "Non-urgent",
  },
};

export const NURSE_COLOR: TriageColorSet = {
  solid: "#6D28D9",
  tint: "#EDE9FE",
  border: "#C4B5FD",
  text: "#4C1D95",
  name: "Nurse now",
};

export function getTriageColor(level: number | null): TriageColorSet {
  if (level === null || level === undefined) {
    return NURSE_COLOR;
  }
  return TRIAGE_COLORS[level] || NURSE_COLOR;
}
