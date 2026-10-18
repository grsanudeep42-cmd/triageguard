import React from "react";
import { RedFlag } from "../types";

interface ComplaintDisplayProps {
  complaintText: string;
  redFlags: RedFlag[];
}

export const ComplaintDisplay: React.FC<ComplaintDisplayProps> = ({
  complaintText,
  redFlags,
}) => {
  if (!complaintText) return null;

  // Extract phrases to highlight
  const matchedPhrases = redFlags
    .map((rf) => rf.matched_text.trim())
    .filter((phrase) => phrase.length > 0);

  // Function to highlight matched phrases in text
  const renderHighlightedText = () => {
    if (matchedPhrases.length === 0) {
      return <span>{complaintText}</span>;
    }

    // Escape regex characters
    const escaped = matchedPhrases.map((p) =>
      p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
    );
    const regex = new RegExp(`(${escaped.join("|")})`, "gi");
    const parts = complaintText.split(regex);

    return (
      <>
        {parts.map((part, i) => {
          const isMatched = matchedPhrases.some(
            (phrase) => phrase.toLowerCase() === part.toLowerCase()
          );
          if (isMatched) {
            return (
              <mark
                key={i}
                className="bg-l2-tint text-l2-text px-1 py-0.5 rounded-[4px] font-medium border border-l2-border"
              >
                {part}
              </mark>
            );
          }
          return <span key={i}>{part}</span>;
        })}
      </>
    );
  };

  return (
    <div className="bg-surface rounded-panel border border-line p-6">
      <h3 className="text-h3 font-semibold text-ink mb-3">Complaint as entered</h3>
      <div className="text-[15px] text-ink leading-relaxed bg-paper/40 p-4 rounded-tile border border-line">
        {renderHighlightedText()}
      </div>

      <div className="mt-4 pt-3 border-t border-line text-[13px] text-ink-3">
        Prototype trained on synthetic data. Decision support only. The nurse makes the final call.
      </div>
    </div>
  );
};
