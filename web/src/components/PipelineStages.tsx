import React from "react";
import { TriageResponse } from "../types";
import { getTriageColor } from "../utils/triageStyles";
import { Check, AlertTriangle, ArrowRight, ShieldCheck, Zap } from "lucide-react";

interface PipelineStagesProps {
  result: TriageResponse | null;
}

export const PipelineStages: React.FC<PipelineStagesProps> = ({ result }) => {
  const levelColors = getTriageColor(result?.triage_level ?? null);

  // If no result yet, render outline-only empty state
  if (!result) {
    const emptyStages = [
      { num: 1, title: "1 Check data", desc: "Validate 5 vitals & syntax" },
      { num: 2, title: "2 Safety rules", desc: "Hard clinical overrides" },
      { num: 3, title: "3 Two models", desc: "Vitals & text fusion" },
      { num: 4, title: "4 Emergency screen", desc: "Calibrated conformal check" },
    ];

    return (
      <div className="bg-surface rounded-panel border border-line p-6">
        <h3 className="text-h3 font-semibold text-ink mb-4">How it decides</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 relative">
          {emptyStages.map((s) => (
            <div
              key={s.num}
              className="p-4 rounded-tile border border-dashed border-line bg-paper/30 flex flex-col items-start"
            >
              <div className="w-7 h-7 rounded-full border border-line-strong flex items-center justify-center text-[12px] font-semibold text-ink-3 mb-2.5">
                {s.num}
              </div>
              <div className="text-[14px] font-semibold text-ink-2 mb-1">{s.title}</div>
              <div className="text-[12px] text-ink-3">{s.desc}</div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Calculate log-scale position for meter (0.0001 to 1.0)
  const calcLogPos = (val: number): number => {
    const clamped = Math.max(0.0001, Math.min(1.0, val));
    const logVal = Math.log10(clamped);
    // Range -4 (0.0001) to 0 (1.0)
    const pct = ((logVal + 4) / 4) * 100;
    return Math.max(0, Math.min(100, pct));
  };

  const pEmPos = calcLogPos(result.p_emergency);
  const tauPos = calcLogPos(result.tau);

  return (
    <div className="bg-surface rounded-panel border border-line p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-h3 font-semibold text-ink">How it decided</h3>
        <div className="text-caption text-ink-3 font-normal">
          Strict clinical policy hierarchy
        </div>
      </div>

      {/* 4 Pipeline stages with connector */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 relative">
        {result.pipeline_stages.map((st, idx) => {
          const isDecided = st.status === "decided";
          const isFired = st.status === "fired" || isDecided;

          return (
            <div
              key={st.stage_number}
              className={`p-4 rounded-tile border transition-all flex flex-col justify-between relative ${
                isDecided
                  ? "bg-surface shadow-sm border-2"
                  : "bg-surface border-line"
              }`}
              style={{
                borderColor: isDecided ? levelColors.solid : undefined,
              }}
            >
              {/* Connector arrow on desktop */}
              {idx < 3 && (
                <div className="hidden lg:block absolute -right-3 top-7 z-10 text-line-strong">
                  <ArrowRight className="w-4 h-4" strokeWidth={1.75} />
                </div>
              )}

              <div>
                <div className="flex items-center justify-between mb-2.5">
                  {/* 28px circle */}
                  <div
                    className={`w-7 h-7 rounded-full flex items-center justify-center text-[12px] font-bold ${
                      isDecided
                        ? "text-surface"
                        : isFired
                        ? "bg-vital-abnormal/15 text-vital-abnormal"
                        : "bg-vital-normal/15 text-vital-normal"
                    }`}
                    style={{
                      backgroundColor: isDecided ? levelColors.solid : undefined,
                    }}
                  >
                    {isDecided ? (
                      <Zap className="w-3.5 h-3.5 stroke-[2.2]" />
                    ) : isFired ? (
                      <AlertTriangle className="w-3.5 h-3.5" strokeWidth={2} />
                    ) : (
                      <Check className="w-3.5 h-3.5" strokeWidth={2.5} />
                    )}
                  </div>

                  {isDecided && (
                    <span
                      className="px-2 py-0.5 text-[11px] font-bold rounded-pill text-surface tabular-nums tracking-wide"
                      style={{ backgroundColor: levelColors.solid }}
                    >
                      Decided here
                    </span>
                  )}
                </div>

                <div className="text-[15px] font-semibold text-ink mb-1">
                  {st.stage_number} {st.title}
                </div>
                <div className="text-[13px] text-ink-2 leading-relaxed">
                  {st.result}
                  {st.stage_number === 3 &&
                    result.views &&
                    result.views.vitals_level !== result.views.text_level &&
                    Math.abs(result.views.vitals_level - result.views.text_level) >= 2 &&
                    !st.result.includes("Overridden") && (
                      <span className="block mt-1 text-[12px] text-vital-abnormal font-medium">
                        {result.source_layer === "rule"
                          ? "Overridden by the safety rule"
                          : result.source_layer === "screen"
                          ? "Overridden by the emergency screen"
                          : null}
                      </span>
                    )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Emergency Screen Log-Scale Meter under stage 4 */}
      <div className="mt-6 pt-5 border-t border-line">
        <div className="flex items-center justify-between text-caption text-ink-2 mb-2">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-ink">Stage 4 emergency probability meter</span>
            <span className="text-[12px] text-ink-3">(logarithmic scale 0.0001 to 1.0)</span>
          </div>
          <div className="flex items-center gap-4 text-[12px] tabular-nums">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-brand inline-block" />
              <span>p_emergency = {result.p_emergency.toFixed(4)}</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-1.5 h-3 bg-vital-abnormal inline-block rounded-[1px]" />
              <span>threshold τ = {result.tau.toFixed(4)}</span>
            </span>
          </div>
        </div>

        {/* Meter track */}
        <div className="relative w-full h-3 bg-line/60 rounded-pill overflow-visible my-3 border border-line">
          {/* Threshold zone (tau to 1.0) */}
          <div
            className="absolute top-0 bottom-0 right-0 bg-l2-tint/60 rounded-r-pill"
            style={{ left: `${tauPos}%` }}
          />

          {/* Tau tick mark */}
          <div
            className="absolute top-[-4px] bottom-[-4px] w-1 bg-vital-abnormal z-10 rounded-[1px]"
            style={{ left: `calc(${tauPos}% - 2px)` }}
            title={`Threshold τ = ${result.tau}`}
          />

          {/* p_emergency point marker */}
          <div
            className="absolute top-[-3px] w-4 h-4 rounded-full bg-brand border-2 border-surface shadow-sm z-20 -ml-2 transition-all"
            style={{ left: `${pEmPos}%` }}
            title={`p_emergency = ${result.p_emergency}`}
          />
        </div>

        {/* Meter ticks labels */}
        <div className="flex justify-between text-[11px] text-ink-3 tabular-nums px-0.5">
          <span>0.0001</span>
          <span>0.001</span>
          <span>0.01</span>
          <span>0.10</span>
          <span>1.00</span>
        </div>
      </div>
    </div>
  );
};
