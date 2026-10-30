import React from "react";
import { TriageResponse, VitalStatus } from "../types";
import { Check, ShieldAlert, AlertTriangle, AlertCircle } from "lucide-react";

interface VitalsAndRulesProps {
  result: TriageResponse;
}

export const VitalsAndRules: React.FC<VitalsAndRulesProps> = ({ result }) => {
  const { vitals, red_flags, rules_fired } = result;

  // Formatting vital tiles
  const renderVitalTile = (
    label: string,
    vitalObj: VitalStatus,
    unit: string,
    bpDiaObj?: VitalStatus
  ) => {
    const isCritical = vitalObj.status === "critical" || (bpDiaObj && bpDiaObj.status === "critical");
    const isInvalidOrMissing =
      vitalObj.status === "invalid" ||
      vitalObj.status === "missing" ||
      (bpDiaObj && (bpDiaObj.status === "invalid" || bpDiaObj.status === "missing"));

    let displayVal: string = "—";
    if (vitalObj.value !== null && vitalObj.value !== undefined) {
      if (bpDiaObj && bpDiaObj.value !== null && bpDiaObj.value !== undefined) {
        displayVal = `${Math.round(vitalObj.value)}/${Math.round(bpDiaObj.value)}`;
      } else {
        displayVal = Number.isInteger(vitalObj.value)
          ? `${vitalObj.value}`
          : `${vitalObj.value.toFixed(1)}`;
      }
    }

    // Status label & color
    let statusText = "Normal";
    let statusDotColor = "bg-vital-normal";
    let statusTextColor = "text-vital-normal";

    if (isCritical) {
      statusText = "Critical";
      statusDotColor = "bg-l1-solid";
      statusTextColor = "text-l1-text font-semibold";
    } else if (isInvalidOrMissing) {
      statusText = "Re-measure";
      statusDotColor = "bg-nurse-solid";
      statusTextColor = "text-nurse-text font-semibold";
      displayVal = "—";
    } else if (vitalObj.status === "abnormal" || (bpDiaObj && bpDiaObj.status === "abnormal")) {
      statusText = "Abnormal";
      statusDotColor = "bg-vital-abnormal";
      statusTextColor = "text-vital-abnormal";
    }

    return (
      <div
        className={`p-4 rounded-tile transition-all flex flex-col justify-between min-h-[110px] ${
          isCritical
            ? "border-2 border-l1-solid bg-l1-tint/50"
            : isInvalidOrMissing
            ? "border border-nurse-border bg-nurse-tint/40"
            : "border border-line bg-surface"
        }`}
      >
        <div className="text-[13px] text-ink-3 font-medium">{label}</div>

        <div className="flex flex-wrap items-baseline gap-x-1.5 gap-y-0.5 my-1">
          <span className="text-[22px] sm:text-[26px] min-[1600px]:text-[28px] font-semibold text-ink tabular-nums leading-none">
            {displayVal}
          </span>
          {displayVal !== "—" && (
            <span className="text-[12px] sm:text-[13px] text-ink-3 font-normal">{unit}</span>
          )}
        </div>

        <div className="flex items-center gap-1.5 pt-1 text-[13px]">
          <span className={`w-2 h-2 rounded-full ${statusDotColor} shrink-0`} />
          <span className={`capitalize ${statusTextColor}`}>{statusText}</span>
        </div>
      </div>
    );
  };

  // Only show red flags with non-empty matched_text
  const validRedFlags = red_flags.filter((rf) => rf.matched_text && rf.matched_text.trim() !== "");

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Left Column: Vitals 2x2 grid */}
      <div className="bg-surface rounded-panel border border-line p-6">
        <h3 className="text-h3 font-semibold text-ink mb-4">Vitals</h3>
        <div className="grid grid-cols-2 gap-3">
          {renderVitalTile("Blood pressure", vitals.bp_sys, "mmHg", vitals.bp_dia)}
          {renderVitalTile("Pulse rate", vitals.pulse, "bpm")}
          {renderVitalTile("Temperature", vitals.temp_c, "°C")}
          {renderVitalTile("Oxygen saturation", vitals.spo2, "%")}
        </div>
      </div>

      {/* Right Column: Red flags and rules */}
      <div className="bg-surface rounded-panel border border-line p-6 flex flex-col justify-between">
        <div>
          <h3 className="text-h3 font-semibold text-ink mb-4">Red flags and rules</h3>

          {/* Red Flag Chips */}
          <div className="mb-4">
            <div className="text-[12px] font-medium text-ink-3 tracking-wide mb-2">
              Clinical red flags
            </div>
            {validRedFlags.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {validRedFlags.map((rf) => (
                  <div
                    key={rf.id}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-l2-tint border border-l2-border text-l2-text rounded-pill text-[13px] font-medium"
                  >
                    <AlertTriangle className="w-3.5 h-3.5 text-l2-solid shrink-0" strokeWidth={1.75} />
                    <span>Matched: "{rf.matched_text}"</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex items-center gap-2 text-[13px] text-ink-3 bg-paper/60 px-3 py-2 rounded-tile border border-line">
                <Check className="w-4 h-4 text-vital-normal shrink-0" strokeWidth={2} />
                <span>No red-flag symptoms detected</span>
              </div>
            )}
          </div>

          {/* Fired Rules Rows */}
          <div>
            <div className="text-[12px] font-medium text-ink-3 tracking-wide mb-2">
              Safety escalation rules
            </div>
            {rules_fired.length > 0 ? (
              <div className="space-y-2">
                {rules_fired.map((rf) => (
                  <div
                    key={rf.id}
                    className="p-3 bg-l1-tint/40 border border-l1-border rounded-tile flex items-start gap-2.5"
                  >
                    <span className="px-2 py-0.5 bg-l1-solid text-surface text-[11px] font-bold rounded-pill shrink-0 mt-0.5">
                      {rf.id}
                    </span>
                    <div className="text-[13px]">
                      <div className="font-semibold text-l1-text">{rf.name}</div>
                      <div className="text-ink-2 mt-0.5">{rf.detail}</div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex items-center gap-2 text-[13px] text-ink-3 bg-paper/60 px-3 py-2 rounded-tile border border-line">
                <Check className="w-4 h-4 text-vital-normal shrink-0" strokeWidth={2} />
                <span>No safety rule fired</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
