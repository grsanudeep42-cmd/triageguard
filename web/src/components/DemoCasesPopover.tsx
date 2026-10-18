import React from "react";
import { DemoCase } from "../types";
import { X, Sparkles, CheckCircle2 } from "lucide-react";

interface DemoCasesPopoverProps {
  isOpen: boolean;
  onClose: () => void;
  cases: DemoCase[];
  selectedCaseId: string | null;
  onSelectCase: (demoCase: DemoCase) => void;
  showAnnotations: boolean;
  onToggleAnnotations: (val: boolean) => void;
}

export const DemoCasesPopover: React.FC<DemoCasesPopoverProps> = ({
  isOpen,
  onClose,
  cases,
  selectedCaseId,
  onSelectCase,
  showAnnotations,
  onToggleAnnotations,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-end p-4 sm:p-6 md:p-8 bg-ink/20 backdrop-blur-none">
      <div
        className="w-full max-w-md bg-surface rounded-panel border border-line shadow-raised overflow-hidden flex flex-col max-h-[85vh] mt-12 animate-verdict"
        role="dialog"
        aria-label="Demo cases catalog"
      >
        {/* Header */}
        <div className="px-5 py-4 border-b border-line flex items-center justify-between bg-surface">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-brand" strokeWidth={1.75} />
            <h3 className="text-h3 font-semibold text-ink">Demo cases</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-ink-3 hover:text-ink rounded-btn hover:bg-paper transition-colors"
            aria-label="Close demo cases"
          >
            <X className="w-5 h-5" strokeWidth={1.75} />
          </button>
        </div>

        {/* Annotations Toggle */}
        <div className="px-5 py-3 border-b border-line bg-paper/60 flex items-center justify-between">
          <div>
            <div className="text-[14px] font-medium text-ink">Show demo annotations</div>
            <div className="text-[12px] text-ink-3">
              Reveals ground truth & selection method next to verdict
            </div>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={showAnnotations}
              onChange={(e) => onToggleAnnotations(e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-10 h-6 bg-line peer-focus:outline-none rounded-pill peer peer-checked:after:translate-x-full peer-checked:after:border-surface after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-surface after:border-line-strong after:border after:rounded-pill after:h-5 after:w-5 after:transition-all peer-checked:bg-brand"></div>
          </label>
        </div>

        {/* Case List */}
        <div className="overflow-y-auto p-4 space-y-2.5 flex-1">
          {cases.map((c) => {
            const isSelected = selectedCaseId === c.id;
            const title = c.scenario
              .replace(/^case\d+[a-z]?_/, "")
              .replace(/_/g, " ");

            return (
              <button
                key={c.id}
                data-case-id={c.id}
                onClick={() => {
                  onSelectCase(c);
                  onClose();
                }}
                className={`w-full text-left p-3.5 rounded-tile border transition-all text-body ${
                  isSelected
                    ? "bg-brand-tint/60 border-brand text-ink"
                    : "bg-surface border-line hover:border-line-strong hover:bg-paper/50"
                }`}
              >
                <div className="flex items-start justify-between gap-2 mb-1">
                  <div className="font-semibold text-[14px] text-ink capitalize">
                    {title}
                  </div>
                  {isSelected && (
                    <CheckCircle2
                      className="w-4 h-4 text-brand shrink-0 mt-0.5"
                      strokeWidth={1.75}
                    />
                  )}
                </div>
                <div className="text-[13px] text-ink-2 line-clamp-2 leading-relaxed">
                  {c.description}
                </div>
                {c.selection_method && (
                  <div className="mt-2 text-[11px] text-ink-3 bg-paper px-2 py-0.5 rounded-pill inline-block border border-line">
                    Documented experiment
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};
