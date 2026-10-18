import React, { useState, useEffect } from "react";
import { RulesResponse, RuleDefinition, RedFlagDefinition } from "../types";
import { ShieldAlert, BookOpen, AlertCircle } from "lucide-react";

export const RulesPage: React.FC = () => {
  const [data, setData] = useState<RulesResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/rules")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((json: RulesResponse) => {
        setData(json);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || "Failed to load rules");
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="w-full max-w-[1280px] min-[1600px]:max-w-[1440px] mx-auto p-4 sm:p-6 md:p-8">
        <div className="bg-surface rounded-panel border border-line p-8 animate-pulse space-y-4">
          <div className="h-6 bg-paper rounded w-1/4" />
          <div className="h-4 bg-paper rounded w-1/2" />
          <div className="h-64 bg-paper rounded" />
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="w-full max-w-[1280px] min-[1600px]:max-w-[1440px] mx-auto p-4 sm:p-6 md:p-8">
        <div className="p-4 bg-l1-tint border border-l1-border text-l1-text rounded-panel flex items-center gap-2">
          <AlertCircle className="w-5 h-5 shrink-0" strokeWidth={1.75} />
          <span>Could not load rules from service. Check that backend is running.</span>
        </div>
      </div>
    );
  }

  const { rules, red_flags, firing_stats } = data;
  const layerStats = firing_stats?.layer_escalation_decomposition;

  // Extract aggregate rule firings from layer decomposition if available
  let totalRulesFiredInEmergencies: string = "98";
  let totalRulesFiredOverall: string = "243";
  if (layerStats && Array.isArray(layerStats)) {
    const l1 = layerStats.find((r) => r["True Level"] === "Level 1");
    const l2 = layerStats.find((r) => r["True Level"] === "Level 2");
    const allNonEm = layerStats.find((r) => r["True Level"]?.includes("Non-Emergencies"));
    if (l1 && l2) {
      const l1Rule = parseInt(l1["(c) rule fired"]?.split("/")[0] || "23", 10);
      const l2Rule = parseInt(l2["(c) rule fired"]?.split("/")[0] || "75", 10);
      const nonEmRule = parseInt(allNonEm?.["(c) rule fired"]?.split("/")[0] || "145", 10);
      totalRulesFiredInEmergencies = `${l1Rule + l2Rule}`;
      totalRulesFiredOverall = `${l1Rule + l2Rule + nonEmRule}`;
    }
  }

  return (
    <div className="w-full max-w-[1280px] min-[1600px]:max-w-[1440px] mx-auto p-4 sm:p-6 md:p-8 space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-h1 font-semibold text-ink tracking-tight mb-2">
          Safety rules and red flags
        </h1>
        <p className="text-body text-ink-2 max-w-3xl leading-relaxed">
          The hard escalation layer acts as a strict safety net before statistical models execute.
          Rules only assign higher urgency (lower numeric level) and never downgrade.
        </p>
      </div>

      {/* Aggregate firing summary card */}
      <div className="bg-surface rounded-panel border border-line p-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-tile bg-l1-tint border border-l1-border flex items-center justify-center text-l1-solid">
            <ShieldAlert className="w-5 h-5" strokeWidth={1.75} />
          </div>
          <div>
            <div className="text-[15px] font-semibold text-ink">
              Clinical escalation performance on test split
            </div>
            <div className="text-[13px] text-ink-3">
              Evaluated across 4,500 test encounters (180 true emergencies)
            </div>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="text-right">
            <div className="text-caption text-ink-3">Total rules fired</div>
            <div className="text-[22px] font-bold text-ink tabular-nums leading-none mt-0.5">
              {totalRulesFiredOverall}
            </div>
          </div>
          <div className="h-8 w-px bg-line" />
          <div className="text-right">
            <div className="text-caption text-ink-3">In true emergencies</div>
            <div className="text-[22px] font-bold text-l1-solid tabular-nums leading-none mt-0.5">
              {totalRulesFiredInEmergencies}
            </div>
          </div>
        </div>
      </div>

      {/* Section 1: Hard-coded Rules Table */}
      <div className="bg-surface rounded-panel border border-line overflow-hidden">
        <div className="px-6 py-4 border-b border-line flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h2 className="text-h3 font-semibold text-ink">Hard-coded vitals rules</h2>
            <span className="px-2 py-0.5 bg-paper text-ink-3 border border-line rounded-pill text-[12px] font-medium">
              {rules.length} active rules
            </span>
          </div>
          <span className="text-[13px] text-ink-3">
            Thresholds: Placeholder pending official primer
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-body">
            <thead>
              <tr className="border-b border-line bg-paper/60 text-caption font-semibold text-ink-3 uppercase tracking-wider">
                <th className="py-3 px-6">ID</th>
                <th className="py-3 px-6">Clinical trigger</th>
                <th className="py-3 px-6">Threshold</th>
                <th className="py-3 px-6 text-center">Forced level</th>
                <th className="py-3 px-6">Age band</th>
                <th className="py-3 px-6">Source status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line text-[14px]">
              {rules.map((r: RuleDefinition) => (
                <tr key={r.id} className="hover:bg-paper/40 transition-colors">
                  <td className="py-3.5 px-6 font-bold text-ink whitespace-nowrap">
                    <span className="px-2 py-0.5 bg-paper border border-line-strong rounded-pill text-[12px]">
                      {r.id}
                    </span>
                  </td>
                  <td className="py-3.5 px-6 font-medium text-ink max-w-md">
                    <div>{r.plain_language_trigger}</div>
                    <div className="text-[12px] text-ink-3 mt-0.5 leading-snug">
                      {r.description}
                    </div>
                  </td>
                  <td className="py-3.5 px-6 font-semibold text-ink tabular-nums whitespace-nowrap">
                    {r.threshold}
                  </td>
                  <td className="py-3.5 px-6 text-center whitespace-nowrap">
                    <span
                      className={`inline-block px-2.5 py-0.5 rounded-pill text-[12px] font-bold ${
                        r.forced_level === 1
                          ? "bg-l1-tint text-l1-text border border-l1-border"
                          : "bg-l2-tint text-l2-text border border-l2-border"
                      }`}
                    >
                      Level {r.forced_level}
                    </span>
                  </td>
                  <td className="py-3.5 px-6 capitalize text-ink-2 whitespace-nowrap">
                    {r.applies_age_band}
                  </td>
                  <td className="py-3.5 px-6 text-[12px] text-ink-3 whitespace-nowrap">
                    {r.threshold_label}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Section 2: Red-flag Lexicon Table */}
      <div className="bg-surface rounded-panel border border-line overflow-hidden">
        <div className="px-6 py-4 border-b border-line flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-brand" strokeWidth={1.75} />
            <h2 className="text-h3 font-semibold text-ink">Red-flag clinical lexicon</h2>
            <span className="px-2 py-0.5 bg-paper text-ink-3 border border-line rounded-pill text-[12px] font-medium">
              {red_flags.length} symptoms
            </span>
          </div>
          <span className="text-[13px] text-ink-3">
            Fuzzy matching (Levenshtein ≤ 2) across 3 language scripts
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-body">
            <thead>
              <tr className="border-b border-line bg-paper/60 text-caption font-semibold text-ink-3 uppercase tracking-wider">
                <th className="py-3 px-6">ID</th>
                <th className="py-3 px-6">Clinical entity</th>
                <th className="py-3 px-6">English triggers</th>
                <th className="py-3 px-6">Telugu / Hindi triggers</th>
                <th className="py-3 px-6 text-center">Floor level</th>
                <th className="py-3 px-6 text-right">Validation status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line text-[14px]">
              {red_flags.map((rf: RedFlagDefinition) => (
                <tr key={rf.id} className="hover:bg-paper/40 transition-colors">
                  <td className="py-3.5 px-6 font-bold text-ink whitespace-nowrap">
                    <span className="px-2 py-0.5 bg-paper border border-line-strong rounded-pill text-[12px]">
                      {rf.id}
                    </span>
                  </td>
                  <td className="py-3.5 px-6 font-semibold text-ink capitalize whitespace-nowrap">
                    {rf.name.replace(/_/g, " ")}
                  </td>
                  <td className="py-3.5 px-6 text-[13px] text-ink-2 max-w-xs">
                    {rf.patterns_en.slice(0, 3).join(", ")}
                    {rf.patterns_en.length > 3 && (
                      <span className="text-ink-3"> +{rf.patterns_en.length - 3}</span>
                    )}
                  </td>
                  <td className="py-3.5 px-6 text-[13px] text-ink-2 max-w-xs italic">
                    {[...rf.patterns_tel.slice(0, 2), ...rf.patterns_hin.slice(0, 2)].join(", ")}
                  </td>
                  <td className="py-3.5 px-6 text-center whitespace-nowrap">
                    <span className="inline-block px-2 py-0.5 bg-l3-tint text-l3-text border border-l3-border rounded-pill text-[12px] font-bold">
                      Floor L3
                    </span>
                  </td>
                  <td className="py-3.5 px-6 text-right whitespace-nowrap">
                    <span
                      className={`inline-block px-2.5 py-0.5 rounded-pill text-[12px] font-medium ${
                        rf.status === "Validated"
                          ? "bg-vital-normal/15 text-vital-normal font-semibold"
                          : "bg-paper text-ink-3 border border-line"
                      }`}
                    >
                      {rf.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
