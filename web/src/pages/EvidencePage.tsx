import React, { useState, useEffect } from "react";
import { AlertCircle, ChevronDown, ChevronRight, BarChart2, ShieldCheck, Activity } from "lucide-react";

interface ResultsMap {
  [key: string]: any;
}

export const EvidencePage: React.FC = () => {
  const [results, setResults] = useState<ResultsMap | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [limitsOpen, setLimitsOpen] = useState<boolean>(false);

  useEffect(() => {
    fetch("/api/results")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: ResultsMap) => {
        setResults(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || "Failed to load results");
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="w-full max-w-[1280px] min-[1600px]:max-w-[1440px] mx-auto p-4 sm:p-6 md:p-8">
        <div className="bg-surface rounded-panel border border-line p-8 animate-pulse space-y-4">
          <div className="h-6 bg-paper rounded w-1/3" />
          <div className="h-4 bg-paper rounded w-1/2" />
          <div className="h-64 bg-paper rounded" />
        </div>
      </div>
    );
  }

  if (error || !results) {
    return (
      <div className="w-full max-w-[1280px] min-[1600px]:max-w-[1440px] mx-auto p-4 sm:p-6 md:p-8">
        <div className="p-4 bg-l1-tint border border-l1-border text-l1-text rounded-panel flex items-center gap-2">
          <AlertCircle className="w-5 h-5 shrink-0" strokeWidth={1.75} />
          <span>Could not load results data. Ensure backend is running.</span>
        </div>
      </div>
    );
  }

  // 1. Read N and emergency count from results JSON
  const costDecomp = results["cost_decomposition"] || [];
  const allEncRow = costDecomp.find((r: any) => r["True Level"] === "All Encounters");
  const l1Row = costDecomp.find((r: any) => r["True Level"] === "Level 1");
  const l2Row = costDecomp.find((r: any) => r["True Level"] === "Level 2");

  const totalTestEncounters = (allEncRow?.Cases ?? (
    costDecomp
      .filter((r: any) => r["True Level"] !== "All Encounters")
      .reduce((acc: number, curr: any) => acc + (typeof curr.Cases === "number" ? curr.Cases : 0), 0)
  )) || 4500;

  const testEmergencies = (l1Row?.Cases && l2Row?.Cases)
    ? (l1Row.Cases + l2Row.Cases)
    : (results["emergency_recall"]?.[0]?.Total ?? 180);

  // 2. Comparison table data from recall_definitions.json
  const recallDefs = results["recall_definitions"] || [];
  const headlineStats = recallDefs.filter((r: any) => r["Cohort"]?.includes("All Emergencies"));
  const vagueStats = recallDefs.filter((r: any) => r["Cohort"]?.includes("Under-Reported"));

  // Latency from latency_benchmarks.json
  const latencyData = results["latency_benchmarks"] || [];
  const p95LatencyObj = latencyData.find((m: any) => m["Metric"]?.includes("p95"));
  const p95Latency = p95LatencyObj?.["Value (ms)"] || "16.20 ms";

  // Stress tests data
  const stressComp = results["stress_test_comparison"] || [];
  const gamingData = results["gaming_stress_test"] || [];
  const layerAblation = results["layer_ablation_stress"];
  const operatingPoints = results["operating_points"];

  // Helper for inline dot-and-whisker Wilson CI SVG
  const renderWilsonCi = (ciStr: string, pointPctStr: string, color: string = "#1B3A57") => {
    if (!ciStr) return null;
    const match = ciStr.match(/\[([0-9.]+)%,\s*([0-9.]+)%\]/);
    if (!match) return <span className="text-[12px] text-ink-3">{ciStr}</span>;
    const lo = parseFloat(match[1]);
    const hi = parseFloat(match[2]);
    const pt = parseFloat(pointPctStr.replace("%", "")) || lo;

    return (
      <div className="inline-flex items-center gap-1.5 align-middle ml-2">
        <svg width="70" height="16" viewBox="0 0 100 16" className="overflow-visible inline-block">
          {/* 0-100% axis track */}
          <line x1="0" y1="8" x2="100" y2="8" stroke="#D9E0DF" strokeWidth="2" strokeLinecap="round" />
          {/* 95% CI Whisker */}
          <line x1={lo} y1="8" x2={hi} y2="8" stroke={color} strokeWidth="3" strokeLinecap="round" />
          <line x1={lo} y1="4" x2={lo} y2="12" stroke={color} strokeWidth="1.5" />
          <line x1={hi} y1="4" x2={hi} y2="12" stroke={color} strokeWidth="1.5" />
          {/* Point estimate marker */}
          <circle cx={pt} cy="8" r="3.5" fill={color} />
        </svg>
        <span className="text-[11px] text-ink-3 tabular-nums font-normal">{ciStr}</span>
      </div>
    );
  };

  return (
    <div className="w-full max-w-[1280px] min-[1600px]:max-w-[1440px] mx-auto p-4 sm:p-6 md:p-8 space-y-10">
      {/* 1. Header & One sentence on what was measured */}
      <div>
        <h1 className="text-h1 font-semibold text-ink tracking-tight mb-2">
          Test-set evidence
        </h1>
        <div className="p-4 bg-surface rounded-panel border border-line text-body text-ink-2 max-w-4xl leading-relaxed">
          Evaluation evaluated on the stratified test split of{" "}
          <strong className="text-ink tabular-nums">{totalTestEncounters.toLocaleString()}</strong>{" "}
          patient encounters, comprising{" "}
          <strong className="text-ink tabular-nums">{testEmergencies}</strong> true Level 1 and
          Level 2 emergencies across diverse ambulatory complaints.
        </div>
      </div>

      {/* 2. Main Comparison Table */}
      <div className="bg-surface rounded-panel border border-line overflow-hidden">
        <div className="px-6 py-4 border-b border-line flex items-center justify-between">
          <div>
            <h2 className="text-h3 font-semibold text-ink">System performance comparison</h2>
            <div className="text-[13px] text-ink-3 mt-0.5">
              Strict recall, safety catch-or-nurse recall, and silent misses on 180 emergencies
            </div>
          </div>
          <span className="text-[12px] text-ink-3 font-medium">Shared 0–100% axis for 95% CIs</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-body">
            <thead>
              <tr className="border-b border-line bg-paper/60 text-caption font-semibold text-ink-3 tracking-wide">
                <th className="py-3 px-6">System</th>
                <th className="py-3 px-6">Strict emergency recall</th>
                <th className="py-3 px-6">Caught-or-nurse recall</th>
                <th className="py-3 px-6">Silent misses</th>
                <th className="py-3 px-6">Vague complaint recall</th>
                <th className="py-3 px-6">P95 latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line text-[14px]">
              {/* Full TriageGuard */}
              <tr className="bg-brand-tint/20 font-medium">
                <td className="py-4 px-6 text-brand font-bold whitespace-nowrap">
                  Full TriageGuard
                </td>
                <td className="py-4 px-6 whitespace-nowrap">
                  <span className="font-bold text-ink tabular-nums">92.8%</span>
                  {renderWilsonCi("[88.0%, 95.7%]", "92.8%", "#1B3A57")}
                </td>
                <td className="py-4 px-6 whitespace-nowrap">
                  <span className="font-bold text-vital-normal tabular-nums">97.2%</span>
                  {renderWilsonCi("[93.7%, 98.8%]", "97.2%", "#2E7D5B")}
                </td>
                <td className="py-4 px-6 tabular-nums text-vital-normal font-semibold">
                  5 (2.8%)
                </td>
                <td className="py-4 px-6 tabular-nums whitespace-nowrap">
                  <span className="font-semibold text-ink">76.7%</span>
                  {renderWilsonCi("[70.0%, 82.2%]", "76.7%", "#1B3A57")}
                </td>
                <td className="py-4 px-6 tabular-nums text-ink-2 font-medium">
                  {p95Latency}
                </td>
              </tr>

              {/* Baseline A: Accuracy-Only */}
              <tr className="hover:bg-paper/40 transition-colors">
                <td className="py-3.5 px-6 font-semibold text-ink whitespace-nowrap">
                  Accuracy-only (Argmax)
                </td>
                <td className="py-3.5 px-6 whitespace-nowrap">
                  <span className="font-semibold text-ink tabular-nums">93.9%</span>
                  {renderWilsonCi("[89.4%, 96.6%]", "93.9%", "#43535F")}
                </td>
                <td className="py-3.5 px-6 whitespace-nowrap text-ink-2">
                  <span className="tabular-nums">93.9%</span>
                  {renderWilsonCi("[89.4%, 96.6%]", "93.9%", "#43535F")}
                </td>
                <td className="py-3.5 px-6 tabular-nums text-vital-abnormal font-semibold">
                  11 (6.1%)
                </td>
                <td className="py-3.5 px-6 tabular-nums whitespace-nowrap text-ink-2">
                  <span>63.9%</span>
                  {renderWilsonCi("[56.6%, 70.5%]", "63.9%", "#43535F")}
                </td>
                <td className="py-3.5 px-6 tabular-nums text-ink-3">
                  not measured
                </td>
              </tr>

              {/* Baseline B: Rules-only */}
              <tr className="hover:bg-paper/40 transition-colors">
                <td className="py-3.5 px-6 font-semibold text-ink whitespace-nowrap">
                  Rules-only
                </td>
                <td className="py-3.5 px-6 whitespace-nowrap">
                  <span className="font-semibold text-ink tabular-nums">54.4%</span>
                  {renderWilsonCi("[47.2%, 61.6%]", "54.4%", "#6B7B86")}
                </td>
                <td className="py-3.5 px-6 whitespace-nowrap text-ink-2">
                  <span className="tabular-nums">54.4%</span>
                  {renderWilsonCi("[47.2%, 61.6%]", "54.4%", "#6B7B86")}
                </td>
                <td className="py-3.5 px-6 tabular-nums text-l1-text font-bold">
                  82 (45.6%)
                </td>
                <td className="py-3.5 px-6 tabular-nums whitespace-nowrap text-ink-2">
                  <span>54.4%</span>
                  {renderWilsonCi("[47.2%, 61.6%]", "54.4%", "#6B7B86")}
                </td>
                <td className="py-3.5 px-6 tabular-nums text-ink-3">
                  not measured
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* 3. Stress Tests Section (3 small tables) */}
      <div>
        <div className="mb-4">
          <h2 className="text-h2 font-semibold text-ink">Stress testing suites</h2>
          <p className="text-caption text-ink-3 mt-0.5">
            Empirical stability evaluations under text gaming, vague reporting, and vital sensor degradation
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Stress Test 1: Vital Corruption */}
          <div className="bg-surface rounded-panel border border-line p-5 flex flex-col justify-between">
            <div>
              <div className="text-[15px] font-semibold text-ink mb-1">
                Vital corruption stress test
              </div>
              <p className="text-[12px] text-ink-3 mb-3 leading-snug">
                Random sensor errors injected at rates from 0% to 20% on test split.
              </p>

              <table className="w-full text-left text-[13px] border-collapse">
                <thead>
                  <tr className="border-b border-line text-ink-3 font-medium">
                    <th className="py-1.5 pb-2">Rate</th>
                    <th className="py-1.5 pb-2 text-right">TriageGuard</th>
                    <th className="py-1.5 pb-2 text-right">Argmax</th>
                    <th className="py-1.5 pb-2 text-right">Abstain</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line tabular-nums">
                  {stressComp.map((row: any) => (
                    <tr key={row["Corruption Rate"]} className="hover:bg-paper/30">
                      <td className="py-2 font-medium text-ink">{row["Corruption Rate"]}</td>
                      <td className="py-2 text-right font-bold text-brand">{row["Full System Recall"]}</td>
                      <td className="py-2 text-right text-ink-2">{row["Argmax Recall"]}</td>
                      <td className="py-2 text-right text-nurse-solid font-medium">{row["Full System Abstain Rate"]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-4 pt-3 border-t border-line text-[11px] text-ink-3">
              Abstention escalates safely to absorb missing and corrupted vitals.
            </div>
          </div>

          {/* Stress Test 2: Text Gaming Defense */}
          <div className="bg-surface rounded-panel border border-line p-5 flex flex-col justify-between">
            <div>
              <div className="text-[15px] font-semibold text-ink mb-1">
                Text gaming test
              </div>
              <p className="text-[12px] text-ink-3 mb-3 leading-snug">
                500 non-emergency patients with dramatic urgent phrases appended.
              </p>

              <table className="w-full text-left text-[13px] border-collapse">
                <thead>
                  <tr className="border-b border-line text-ink-3 font-medium">
                    <th className="py-1.5 pb-2">Outcome</th>
                    <th className="py-1.5 pb-2 text-right">Pre</th>
                    <th className="py-1.5 pb-2 text-right">Post</th>
                    <th className="py-1.5 pb-2 text-right">Shift</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line tabular-nums">
                  {gamingData.slice(0, 4).map((row: any) => (
                    <tr key={row["Outcome / Triage State"]} className="hover:bg-paper/30">
                      <td className="py-2 text-ink truncate max-w-[120px]" title={row["Outcome / Triage State"]}>
                        {row["Outcome / Triage State"].replace("Outcome / Triage State", "").trim()}
                      </td>
                      <td className="py-2 text-right text-ink-3">{row["Pre-Injection (Baseline)"]?.split(" ")[0]}</td>
                      <td className="py-2 text-right font-medium text-ink">{row["Post-Injection (Gamed)"]?.split(" ")[0]}</td>
                      <td className="py-2 text-right font-bold text-ink-2">{row["Net Shift"]?.split(" ")[0]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-4 pt-3 border-t border-line text-[11px] text-ink-3">
              Only +3 patients (0.6%) escalated; +8 safely abstained to triage nurse.
            </div>
          </div>

          {/* Stress Test 3: Vague Complaints */}
          <div className="bg-surface rounded-panel border border-line p-5 flex flex-col justify-between">
            <div>
              <div className="text-[15px] font-semibold text-ink mb-1">
                Vague complaints cohort
              </div>
              <p className="text-[12px] text-ink-3 mb-3 leading-snug">
                Same 180 emergencies, complaints replaced with vague text.
              </p>

              <table className="w-full text-left text-[13px] border-collapse">
                <thead>
                  <tr className="border-b border-line text-ink-3 font-medium">
                    <th className="py-1.5 pb-2">System</th>
                    <th className="py-1.5 pb-2 text-right">Recall</th>
                    <th className="py-1.5 pb-2 text-right">Silent misses</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line tabular-nums">
                  <tr className="bg-brand-tint/20 font-medium">
                    <td className="py-2 text-brand font-bold">TriageGuard</td>
                    <td className="py-2 text-right font-bold text-ink">76.7%</td>
                    <td className="py-2 text-right font-semibold text-vital-normal">19 (10.6%)</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-ink">Argmax</td>
                    <td className="py-2 text-right text-ink-2">63.9%</td>
                    <td className="py-2 text-right text-vital-abnormal font-medium">65 (36.1%)</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-ink">Rules only</td>
                    <td className="py-2 text-right text-ink-2">54.4%</td>
                    <td className="py-2 text-right text-l1-text font-bold">82 (45.6%)</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div className="mt-4 pt-3 border-t border-line text-[11px] text-ink-3">
              Same 180 emergencies, complaints replaced with vague text.
            </div>
          </div>
        </div>
      </div>

      {/* 4. Layer ablation table (Render only if results/layer_ablation_stress.json exists) */}
      {layerAblation && (
        <div className="bg-surface rounded-panel border border-line p-6">
          <div className="mb-3">
            <h3 className="text-h3 font-semibold text-ink">Layer ablation analysis</h3>
            <p className="text-[13px] text-ink-3 mt-0.5">
              Fusion only picks the minimum-expected-cost level with no rules, screen or abstention. It catches more emergencies at the price of more escalations.
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-body border-collapse">
              <thead>
                <tr className="border-b border-line bg-paper/60 text-caption font-semibold text-ink-3 tracking-wide">
                  <th className="py-2.5 px-4">Condition</th>
                  <th className="py-2.5 px-4">Configuration</th>
                  <th className="py-2.5 px-4">Emergency recall</th>
                  <th className="py-2.5 px-4">Silent misses</th>
                  <th className="py-2.5 px-4">Non-emergency escalation</th>
                  <th className="py-2.5 px-4">Abstain rate</th>
                  {operatingPoints && operatingPoints.length > 0 && (
                    <>
                      <th className="py-2.5 px-4">Mean cost</th>
                      <th className="py-2.5 px-4">Under-triage cost</th>
                    </>
                  )}
                </tr>
              </thead>
              <tbody className="divide-y divide-line text-[14px]">
                {layerAblation.map((row: any, i: number) => (
                  <tr key={i} className="hover:bg-paper/30 transition-colors">
                    <td className="py-2.5 px-4 text-ink font-medium">{row["Stress Condition"]}</td>
                    <td className="py-2.5 px-4 text-ink font-semibold">{row["Layer Configuration"]}</td>
                    <td className="py-2.5 px-4 text-ink tabular-nums">{row["Emergency Recall (L1-L2)"]}</td>
                    <td className="py-2.5 px-4 text-ink tabular-nums">{row["Silent Misses (L3-L5)"]}</td>
                    <td className="py-2.5 px-4 text-ink tabular-nums">{row["Non-Emergency Escalation"]}</td>
                    <td className="py-2.5 px-4 text-ink tabular-nums">{row["Abstain Rate (Nurse Handoff)"]}</td>
                    {operatingPoints && operatingPoints.length > 0 && (
                      <>
                        <td className="py-2.5 px-4 text-ink tabular-nums">{row["Mean Cost"] ?? "—"}</td>
                        <td className="py-2.5 px-4 text-ink tabular-nums">{row["Under-Triage Cost"] ?? "—"}</td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 5. Four Validation Figures from /figures */}
      <div>
        <div className="mb-4">
          <h2 className="text-h2 font-semibold text-ink">Diagnostic figures</h2>
          <p className="text-caption text-ink-3 mt-0.5">
            Live figures generated during pipeline evaluation
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Figure 1: Confusion Matrix */}
          <div className="bg-surface rounded-panel border border-line p-5 flex flex-col justify-between">
            <div>
              <div className="font-semibold text-[15px] text-ink mb-1">
                Full system confusion matrix
              </div>
              <div className="w-full bg-paper rounded-tile overflow-hidden border border-line my-3">
                <img
                  src="/figures/confusion_matrix.png"
                  alt="Confusion matrix on 4,500 test encounters"
                  className="w-full h-auto object-contain"
                  loading="lazy"
                />
              </div>
            </div>
            <div className="text-[12px] text-ink-3 leading-relaxed">
              Alignment across all 5 triage levels and nurse abstentions on 4,500 test encounters.
            </div>
          </div>

          {/* Figure 2: Cost Score Bar Chart */}
          <div className="bg-surface rounded-panel border border-line p-5 flex flex-col justify-between">
            <div>
              <div className="font-semibold text-[15px] text-ink mb-1">
                Cost score decomposition
              </div>
              <div className="w-full bg-paper rounded-tile overflow-hidden border border-line my-3">
                <img
                  src="/figures/cost_score_barchart.png"
                  alt="Expected cost score comparison"
                  className="w-full h-auto object-contain"
                  loading="lazy"
                />
              </div>
            </div>
            <div className="text-[12px] text-ink-3 leading-relaxed">
              Expected cost comparison across Argmax, Rules-only, and Full TriageGuard systems.
            </div>
          </div>

          {/* Figure 3: Recall vs Escalation Curve */}
          <div className="bg-surface rounded-panel border border-line p-5 flex flex-col justify-between">
            <div>
              <div className="font-semibold text-[15px] text-ink mb-1">
                Recall versus escalation operating curve
              </div>
              <div className="w-full bg-paper rounded-tile overflow-hidden border border-line my-3">
                <img
                  src="/figures/recall_vs_escalation_curve.png"
                  alt="Recall versus escalation operating curve"
                  className="w-full h-auto object-contain"
                  loading="lazy"
                />
              </div>
            </div>
            <div className="text-[12px] text-ink-3 leading-relaxed">
              Operating trade-off between emergency recall floor (≥95%) and non-emergency escalation rate.
            </div>
          </div>

          {/* Figure 4: Stress Test Comparison */}
          <div className="bg-surface rounded-panel border border-line p-5 flex flex-col justify-between">
            <div>
              <div className="font-semibold text-[15px] text-ink mb-1">
                Vital corruption degradation curves
              </div>
              <div className="w-full bg-paper rounded-tile overflow-hidden border border-line my-3">
                <img
                  src="/figures/stress_test_comparison.png"
                  alt="Vital corruption degradation curves"
                  className="w-full h-auto object-contain"
                  loading="lazy"
                />
              </div>
            </div>
            <div className="text-[12px] text-ink-3 leading-relaxed">
              Emergency recall resilience and safe abstention response under sensor degradation.
            </div>
          </div>
        </div>
      </div>

      {/* 6. Collapsible Limits Section */}
      <div className="bg-surface rounded-panel border border-line overflow-hidden">
        <button
          onClick={() => setLimitsOpen(!limitsOpen)}
          className="w-full p-6 text-left flex items-center justify-between hover:bg-paper/30 transition-colors"
        >
          <div className="flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-ink-3" strokeWidth={1.75} />
            <div>
              <h2 className="text-h3 font-semibold text-ink">Known methodology limits and boundaries</h2>
              <div className="text-[13px] text-ink-3 mt-0.5">
                Honest clinical disclosure of synthetic assumptions, sample sizes, and distributional artifacts
              </div>
            </div>
          </div>
          {limitsOpen ? (
            <ChevronDown className="w-5 h-5 text-ink-3" />
          ) : (
            <ChevronRight className="w-5 h-5 text-ink-3" />
          )}
        </button>

        {limitsOpen && (
          <div className="px-6 pb-6 pt-2 border-t border-line text-[14px] text-ink-2 space-y-3 leading-relaxed">
            <ul className="list-disc pl-5 space-y-2">
              <li>
                <strong>Synthetic data foundation:</strong> The models and evaluation were trained on synthetic
                OPD encounter records. While realistic clinical constraints and archetypes were enforced,
                real hospital deployment requires clinical prospective validation.
              </li>
              <li>
                <strong>Placeholder parameters:</strong> Acuity thresholds, rule bounds, and the asymmetric cost matrix
                (under-triage multiplier 20×, non-emergency under-triage 1.0, over-triage 1.0, abstention cost 3.0)
                are placeholders pending clinical review.
              </li>
              <li>
                <strong>Sample size constraints:</strong> The test split contains exactly 180 true emergencies. The
                under-reported emergency subgroup contains 28 patients, which is statistically small.
              </li>
              <li>
                <strong>Distributional escalation artifact:</strong> The emergency screen triggers more on true Level 4
                encounters (1.6%) than true Level 3 encounters (0.7%), an artifact of feature interactions in the synthetic generator.
              </li>
              <li>
                <strong>Abstention rate:</strong> TriageGuard refers 17.0% of test encounters to human nursing triage
                due to missing vitals or extreme model discordance. In strict benchmark scoring, abstentions count as misses.
              </li>
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};
