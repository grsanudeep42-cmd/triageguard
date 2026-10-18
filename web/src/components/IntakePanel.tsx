import React, { useState, useEffect, useRef } from "react";
import { PatientIntake } from "../types";
import { AlertCircle, Plus, X, Edit3 } from "lucide-react";

interface IntakePanelProps {
  intake: PatientIntake;
  onChange: (updated: PatientIntake) => void;
  onSubmit: () => void;
  onClear: () => void;
  loading: boolean;
  hasResult: boolean;
}

export const IntakePanel: React.FC<IntakePanelProps> = ({
  intake,
  onChange,
  onSubmit,
  onClear,
  loading,
  hasResult,
}) => {
  const [collapsedOnMobile, setCollapsedOnMobile] = useState(false);
  const [newCondition, setNewCondition] = useState("");
  const [newMedication, setNewMedication] = useState("");
  const complaintRef = useRef<HTMLTextAreaElement>(null);

  // Autofocus the complaint field on mount
  useEffect(() => {
    complaintRef.current?.focus();
  }, []);

  // Collapse intake panel on mobile when a result is received
  useEffect(() => {
    if (hasResult && window.innerWidth < 1024) {
      setCollapsedOnMobile(true);
    }
  }, [hasResult]);

  // Handle Ctrl/Cmd+Enter to submit
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      onSubmit();
    }
  };

  // Helper plausibility warnings (non-blocking amber helper text)
  const isImplausible = (field: keyof PatientIntake, val: number | null): boolean => {
    if (val === null || val === undefined) return false;
    if (field === "bp_sys") return val <= 0 || val < 50 || val > 240;
    if (field === "bp_dia") return val <= 0 || val < 20 || val > 150;
    if (field === "pulse") return val <= 0 || val < 25 || val > 250;
    if (field === "temp_c") return val < 34.0 || val > 42.5; // includes Fahrenheit slips
    if (field === "spo2") return val <= 0 || val < 50 || val > 100;
    return false;
  };

  const handleNumericChange = (field: keyof PatientIntake, rawStr: string) => {
    if (rawStr === "") {
      onChange({ ...intake, [field]: null });
      return;
    }
    const parsed = parseFloat(rawStr);
    onChange({ ...intake, [field]: isNaN(parsed) ? null : parsed });
  };

  const addCondition = () => {
    const trimmed = newCondition.trim();
    if (trimmed && !intake.conditions.includes(trimmed)) {
      onChange({ ...intake, conditions: [...intake.conditions, trimmed] });
      setNewCondition("");
    }
  };

  const removeCondition = (c: string) => {
    onChange({ ...intake, conditions: intake.conditions.filter((item) => item !== c) });
  };

  const addMedication = () => {
    const trimmed = newMedication.trim();
    if (trimmed && !intake.medications.includes(trimmed)) {
      onChange({ ...intake, medications: [...intake.medications, trimmed] });
      setNewMedication("");
    }
  };

  const removeMedication = (m: string) => {
    onChange({ ...intake, medications: intake.medications.filter((item) => item !== m) });
  };

  // Mobile collapsed view after submission
  if (collapsedOnMobile) {
    return (
      <div className="bg-surface rounded-panel border border-line p-4 shadow-raised flex items-center justify-between lg:hidden mb-6">
        <div className="text-body truncate mr-4">
          <span className="font-semibold text-ink">{intake.age ?? "—"}y / {intake.sex}</span>
          <span className="text-ink-2 mx-2">—</span>
          <span className="text-ink-3 italic truncate">
            {intake.complaint_text || "No complaint entered"}
          </span>
        </div>
        <button
          onClick={() => setCollapsedOnMobile(false)}
          className="flex items-center gap-1.5 px-3 py-1.5 text-[13px] font-semibold text-brand bg-brand-tint rounded-btn border border-brand/20 shrink-0"
        >
          <Edit3 className="w-3.5 h-3.5" />
          <span>Edit</span>
        </button>
      </div>
    );
  }

  return (
    <div
      onKeyDown={handleKeyDown}
      className="bg-surface rounded-panel border border-line p-6 shadow-raised flex flex-col w-full max-w-full"
    >
      <div className="flex items-center justify-between pb-4 border-b border-line">
        <h2 className="text-h2 font-semibold text-ink">New patient</h2>
        {hasResult && (
          <button
            onClick={() => setCollapsedOnMobile(true)}
            className="text-[13px] text-ink-3 hover:text-ink lg:hidden"
          >
            Collapse
          </button>
        )}
      </div>

      <div className="divide-y divide-line">
        {/* Section 1: Patient info */}
        <div className="py-5 space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label htmlFor="patient-age" className="block text-caption font-medium text-ink-2 mb-1.5">
                Age
              </label>
              <input
                id="patient-age"
                type="number"
                min="0"
                max="125"
                value={intake.age ?? ""}
                onChange={(e) => handleNumericChange("age", e.target.value)}
                placeholder="e.g. 45"
                className="w-full h-11 px-3 rounded-input border border-line-strong bg-surface text-ink tabular-nums text-[15px]"
              />
            </div>
            <div>
              <label className="block text-caption font-medium text-ink-2 mb-1.5">
                Sex
              </label>
              <div className="grid grid-cols-3 h-11 bg-paper p-1 rounded-input border border-line">
                {["M", "F", "O"].map((opt) => (
                  <button
                    key={opt}
                    type="button"
                    onClick={() => onChange({ ...intake, sex: opt })}
                    className={`h-full text-[13px] font-medium rounded-[6px] transition-all ${
                      intake.sex === opt
                        ? "bg-surface text-ink font-semibold shadow-sm border border-line-strong/30"
                        : "text-ink-2 hover:text-ink"
                    }`}
                  >
                    {opt}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Section 2: Vitals */}
        <div className="py-5 space-y-3">
          <div className="text-caption font-semibold text-ink uppercase tracking-wider mb-2">
            Vitals
          </div>
          <div className="grid grid-cols-2 gap-3">
            {/* BP Systolic */}
            <div>
              <label htmlFor="bp-sys" className="block text-caption font-medium text-ink-2 mb-1.5">
                BP systolic
              </label>
              <div className="relative">
                <input
                  id="bp-sys"
                  type="number"
                  value={intake.bp_sys ?? ""}
                  onChange={(e) => handleNumericChange("bp_sys", e.target.value)}
                  placeholder="120"
                  className="w-full h-11 pl-3 pr-14 rounded-input border border-line-strong bg-surface text-ink tabular-nums text-[15px]"
                />
                <span className="absolute right-3 top-3 text-[13px] text-ink-3 pointer-events-none tabular-nums">
                  mmHg
                </span>
              </div>
              {isImplausible("bp_sys", intake.bp_sys) && (
                <div className="flex items-center gap-1 mt-1 text-[12px] text-vital-abnormal leading-tight">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" strokeWidth={1.75} />
                  <span>Looks unusual. It will be flagged for re-measurement.</span>
                </div>
              )}
            </div>

            {/* BP Diastolic */}
            <div>
              <label htmlFor="bp-dia" className="block text-caption font-medium text-ink-2 mb-1.5">
                BP diastolic
              </label>
              <div className="relative">
                <input
                  id="bp-dia"
                  type="number"
                  value={intake.bp_dia ?? ""}
                  onChange={(e) => handleNumericChange("bp_dia", e.target.value)}
                  placeholder="80"
                  className="w-full h-11 pl-3 pr-14 rounded-input border border-line-strong bg-surface text-ink tabular-nums text-[15px]"
                />
                <span className="absolute right-3 top-3 text-[13px] text-ink-3 pointer-events-none tabular-nums">
                  mmHg
                </span>
              </div>
              {isImplausible("bp_dia", intake.bp_dia) && (
                <div className="flex items-center gap-1 mt-1 text-[12px] text-vital-abnormal leading-tight">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" strokeWidth={1.75} />
                  <span>Looks unusual. It will be flagged for re-measurement.</span>
                </div>
              )}
            </div>

            {/* Pulse */}
            <div>
              <label htmlFor="pulse" className="block text-caption font-medium text-ink-2 mb-1.5">
                Pulse
              </label>
              <div className="relative">
                <input
                  id="pulse"
                  type="number"
                  value={intake.pulse ?? ""}
                  onChange={(e) => handleNumericChange("pulse", e.target.value)}
                  placeholder="72"
                  className="w-full h-11 pl-3 pr-12 rounded-input border border-line-strong bg-surface text-ink tabular-nums text-[15px]"
                />
                <span className="absolute right-3 top-3 text-[13px] text-ink-3 pointer-events-none tabular-nums">
                  bpm
                </span>
              </div>
              {isImplausible("pulse", intake.pulse) && (
                <div className="flex items-center gap-1 mt-1 text-[12px] text-vital-abnormal leading-tight">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" strokeWidth={1.75} />
                  <span>Looks unusual. It will be flagged for re-measurement.</span>
                </div>
              )}
            </div>

            {/* Temperature */}
            <div>
              <label htmlFor="temp-c" className="block text-caption font-medium text-ink-2 mb-1.5">
                Temperature
              </label>
              <div className="relative">
                <input
                  id="temp-c"
                  type="number"
                  step="0.1"
                  value={intake.temp_c ?? ""}
                  onChange={(e) => handleNumericChange("temp_c", e.target.value)}
                  placeholder="37.0"
                  className="w-full h-11 pl-3 pr-10 rounded-input border border-line-strong bg-surface text-ink tabular-nums text-[15px]"
                />
                <span className="absolute right-3 top-3 text-[13px] text-ink-3 pointer-events-none tabular-nums">
                  °C
                </span>
              </div>
              {isImplausible("temp_c", intake.temp_c) && (
                <div className="flex items-center gap-1 mt-1 text-[12px] text-vital-abnormal leading-tight">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" strokeWidth={1.75} />
                  <span>Looks unusual. It will be flagged for re-measurement.</span>
                </div>
              )}
            </div>

            {/* SpO2 */}
            <div>
              <label htmlFor="spo2" className="block text-caption font-medium text-ink-2 mb-1.5">
                SpO₂
              </label>
              <div className="relative">
                <input
                  id="spo2"
                  type="number"
                  step="0.1"
                  value={intake.spo2 ?? ""}
                  onChange={(e) => handleNumericChange("spo2", e.target.value)}
                  placeholder="98"
                  disabled={intake.spo2 === null && false}
                  className="w-full h-11 pl-3 pr-10 rounded-input border border-line-strong bg-surface text-ink tabular-nums text-[15px]"
                />
                <span className="absolute right-3 top-3 text-[13px] text-ink-3 pointer-events-none tabular-nums">
                  %
                </span>
              </div>
              {isImplausible("spo2", intake.spo2) && (
                <div className="flex items-center gap-1 mt-1 text-[12px] text-vital-abnormal leading-tight">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" strokeWidth={1.75} />
                  <span>Looks unusual. It will be flagged for re-measurement.</span>
                </div>
              )}
            </div>

            {/* Not measured toggle for SpO2 */}
            <div className="flex flex-col justify-end">
              <button
                type="button"
                onClick={() => onChange({ ...intake, spo2: intake.spo2 === null ? 98 : null })}
                className={`h-11 px-3 rounded-input text-[13px] font-medium border text-center transition-colors ${
                  intake.spo2 === null
                    ? "bg-nurse-tint border-nurse-border text-nurse-text"
                    : "bg-paper border-line text-ink-2 hover:text-ink"
                }`}
              >
                {intake.spo2 === null ? "SpO₂ not measured" : "Set not measured"}
              </button>
            </div>
          </div>
        </div>

        {/* Section 3: Complaint */}
        <div className="py-5 space-y-2">
          <label htmlFor="complaint-text" className="block text-caption font-medium text-ink-2">
            Chief complaint
          </label>
          <textarea
            id="complaint-text"
            ref={complaintRef}
            rows={4}
            value={intake.complaint_text}
            onChange={(e) => onChange({ ...intake, complaint_text: e.target.value })}
            placeholder="Describe it in the patient's own words. English, Telugu, Hindi, any script."
            className="w-full min-h-[112px] p-3 rounded-input border border-line-strong bg-surface text-ink text-[15px] resize-y leading-relaxed"
          />
        </div>

        {/* Section 4: History */}
        <div className="py-5 space-y-4">
          {/* Conditions */}
          <div>
            <label className="block text-caption font-medium text-ink-2 mb-1.5">
              Known conditions
            </label>
            <div className="flex gap-2 mb-2">
              <input
                type="text"
                value={newCondition}
                onChange={(e) => setNewCondition(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    addCondition();
                  }
                }}
                placeholder="e.g. Type 2 Diabetes"
                className="flex-1 h-9 px-3 rounded-input border border-line text-[13px] bg-surface text-ink"
              />
              <button
                type="button"
                onClick={addCondition}
                className="h-9 px-3 rounded-input border border-line bg-paper hover:bg-line text-ink text-[13px] font-medium flex items-center gap-1"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Add</span>
              </button>
            </div>
            {intake.conditions.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {intake.conditions.map((c) => (
                  <span
                    key={c}
                    className="inline-flex items-center gap-1 px-2.5 py-1 bg-paper border border-line rounded-pill text-[12px] text-ink"
                  >
                    {c}
                    <button
                      type="button"
                      onClick={() => removeCondition(c)}
                      className="text-ink-3 hover:text-ink"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Current medicines */}
          <div>
            <label className="block text-caption font-medium text-ink-2 mb-1.5">
              Current medicines
            </label>
            <div className="flex gap-2 mb-2">
              <input
                type="text"
                value={newMedication}
                onChange={(e) => setNewMedication(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    addMedication();
                  }
                }}
                placeholder="e.g. Metformin 500mg"
                className="flex-1 h-9 px-3 rounded-input border border-line text-[13px] bg-surface text-ink"
              />
              <button
                type="button"
                onClick={addMedication}
                className="h-9 px-3 rounded-input border border-line bg-paper hover:bg-line text-ink text-[13px] font-medium flex items-center gap-1"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Add</span>
              </button>
            </div>
            {intake.medications.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {intake.medications.map((m) => (
                  <span
                    key={m}
                    className="inline-flex items-center gap-1 px-2.5 py-1 bg-paper border border-line rounded-pill text-[12px] text-ink"
                  >
                    {m}
                    <button
                      type="button"
                      onClick={() => removeMedication(m)}
                      className="text-ink-3 hover:text-ink"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Footer controls inside panel */}
      <div className="pt-5 mt-2 border-t border-line space-y-2">
        <button
          type="button"
          onClick={onSubmit}
          disabled={loading}
          className="w-full h-12 bg-brand hover:bg-brand-hover active:bg-brand-hover text-surface text-[15px] font-semibold rounded-btn transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
        >
          {loading ? (
            <span>Assessing...</span>
          ) : (
            <>
              <span>Assess patient</span>
              <kbd className="hidden sm:inline-block px-1.5 py-0.5 text-[11px] font-normal bg-brand-hover/80 text-surface/80 rounded border border-white/20">
                ⌘↵
              </kbd>
            </>
          )}
        </button>

        <button
          type="button"
          onClick={onClear}
          className="w-full py-2 text-[14px] text-ink-3 hover:text-ink font-medium text-center transition-colors"
        >
          Clear
        </button>
      </div>
    </div>
  );
};
