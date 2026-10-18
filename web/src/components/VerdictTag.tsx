import React from "react";
import { TriageResponse, DemoCase } from "../types";
import { getTriageColor, NURSE_COLOR } from "../utils/triageStyles";
import { UserCheck, Clock, ShieldAlert, Cpu, Sparkles } from "lucide-react";

interface VerdictTagProps {
  result: TriageResponse;
  activeDemoCase: DemoCase | null;
  showAnnotations: boolean;
}

export const VerdictTag: React.FC<VerdictTagProps> = ({
  result,
  activeDemoCase,
  showAnnotations,
}) => {
  const isAbstain = result.triage_level === null;
  const colors = getTriageColor(result.triage_level);

  // Abstain plain-English explanation
  const getAbstainReasonText = (reason: string | null) => {
    if (reason === "needs_remeasure") {
      return "Some vitals look invalid or are missing. Re-measure them.";
    }
    if (reason === "model_disagreement") {
      return "The complaint and the vitals point to different urgency. A nurse should assess.";
    }
    if (reason === "high_conformal_uncertainty") {
      return "High uncertainty across multiple acuity levels. A nurse should assess.";
    }
    return "A nurse should perform an in-person physical assessment.";
  };

  const getSourceLayerLabel = (source: string) => {
    if (source === "rule") return "Hard safety rule";
    if (source === "screen") return "Emergency screen";
    if (source === "fusion") return "Dual-model fusion";
    if (source === "abstain") return "Safety abstention";
    return source;
  };

  const forcedRuleId = result.rules_fired.length > 0 && result.source_layer === "rule"
    ? result.rules_fired[0].id
    : null;

  return (
    <div className="bg-surface rounded-verdict border border-line p-7 shadow-raised min-h-[220px] flex flex-col md:flex-row gap-6 items-stretch animate-verdict">
      {/* Left Block (200px wide, solid triage color, radius 14 inset) */}
      <div
        className="w-full md:w-[200px] shrink-0 rounded-panel flex flex-col items-center justify-center p-6 text-surface select-none shadow-sm"
        style={{ backgroundColor: colors.solid }}
      >
        {isAbstain ? (
          <div className="flex flex-col items-center text-center">
            <UserCheck className="w-20 h-20 mb-2 stroke-[1.75]" />
            <div className="text-[18px] font-semibold tracking-wide">
              Nurse now
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center text-center">
            <div className="text-[110px] sm:text-[120px] font-extrabold leading-none tracking-tight tabular-nums">
              {result.triage_level}
            </div>
            <div className="text-[18px] font-semibold mt-1">
              {result.level_label}
            </div>
          </div>
        )}
      </div>

      {/* Right Block */}
      <div className="flex-1 flex flex-col justify-between py-1">
        <div>
          {/* Main Headline */}
          {isAbstain ? (
            <div>
              <h1 className="text-h1 font-semibold text-ink tracking-tight mb-2">
                Send to nurse now
              </h1>
              <p className="text-body text-ink-2 mb-3 leading-relaxed">
                {getAbstainReasonText(result.abstain_reason)}
              </p>
              <div className="text-[15px] font-medium text-ink-3">
                Department: the nurse decides
              </div>
            </div>
          ) : (
            <div>
              <h1 className="text-h1 font-semibold text-ink tracking-tight mb-1">
                {result.primary_department || "General Medicine"}
              </h1>
              {result.suggested_specialty && result.suggested_specialty !== result.primary_department && (
                <div className="text-[15px] text-ink-2 mb-3 font-normal">
                  Suggested specialty:{" "}
                  <span className="font-medium text-ink">
                    {result.suggested_specialty}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Fact pills */}
          <div className="flex flex-wrap items-center gap-2 mt-4">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-paper border border-line rounded-pill text-[13px] text-ink-2 font-medium">
              <Cpu className="w-3.5 h-3.5 text-ink-3" strokeWidth={1.75} />
              <span>Decided by: {getSourceLayerLabel(result.source_layer)}</span>
            </div>

            {result.conformal_set.length > 0 && !isAbstain && (
              result.source_layer === "rule" ? (
                <div className="inline-flex items-center gap-1 px-3 py-1 bg-paper border border-line rounded-pill text-[13px] text-ink-2 font-medium">
                  <span>Models alone:</span>
                  <span className="font-semibold text-ink tabular-nums">
                    L{result.views?.vitals_level ?? result.views?.text_level ?? result.conformal_set[0] ?? 3}
                  </span>
                </div>
              ) : (
                <div className="inline-flex items-center gap-1 px-3 py-1 bg-paper border border-line rounded-pill text-[13px] text-ink-2 font-medium">
                  <span>Plausible levels:</span>
                  <span className="font-semibold text-ink tabular-nums">
                    {result.conformal_set.join(", ")}
                  </span>
                </div>
              )
            )}

            <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-paper border border-line rounded-pill text-[13px] text-ink-2 font-medium">
              <Clock className="w-3.5 h-3.5 text-ink-3" strokeWidth={1.75} />
              <span className="tabular-nums">{result.latency_ms} ms</span>
            </div>

            {forcedRuleId && (
              <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-l1-tint border border-l1-border text-l1-text rounded-pill text-[13px] font-medium">
                <ShieldAlert className="w-3.5 h-3.5 text-l1-solid shrink-0" strokeWidth={1.75} />
                <span>Hard-coded rule {forcedRuleId} forced this level</span>
              </div>
            )}
          </div>
        </div>

        {/* Demo annotations bar (Honest display of ground-truth & gaming selection) */}
        {showAnnotations && activeDemoCase && (
          <div className="mt-5 pt-4 border-t border-line/70 flex flex-wrap items-center justify-between gap-3 text-[13px] bg-brand-tint/30 px-3.5 py-2.5 rounded-tile border border-brand/20">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-brand shrink-0" strokeWidth={1.75} />
              <span className="font-semibold text-ink">Demo annotation:</span>
              <span className="text-ink-2">
                Ground truth: <strong className="text-ink">Level {activeDemoCase.ground_truth.true_level}</strong> ({activeDemoCase.ground_truth.true_department})
              </span>
            </div>

            {activeDemoCase.injected_dramatic_phrase && (
              <div className="text-[12px] text-ink-2">
                <span className="font-medium text-vital-abnormal">Gamed input:</span> "{activeDemoCase.injected_dramatic_phrase}"
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
