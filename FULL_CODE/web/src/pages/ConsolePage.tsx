import React, { useState, useEffect } from "react";
import { PatientIntake, TriageResponse, DemoCase } from "../types";
import { IntakePanel } from "../components/IntakePanel";
import { VerdictTag } from "../components/VerdictTag";
import { PipelineStages } from "../components/PipelineStages";
import { VitalsAndRules } from "../components/VitalsAndRules";
import { ComplaintDisplay } from "../components/ComplaintDisplay";
import { AlertCircle, ArrowUpRight } from "lucide-react";

interface ConsolePageProps {
  intake: PatientIntake;
  onChangeIntake: (updated: PatientIntake) => void;
  result: TriageResponse | null;
  loading: boolean;
  error: string | null;
  onSubmit: () => void;
  onClear: () => void;
  activeDemoCase: DemoCase | null;
  showAnnotations: boolean;
  onOpenDemoCases: () => void;
}

export const ConsolePage: React.FC<ConsolePageProps> = ({
  intake,
  onChangeIntake,
  result,
  loading,
  error,
  onSubmit,
  onClear,
  activeDemoCase,
  showAnnotations,
  onOpenDemoCases,
}) => {
  // Show skeleton only after 150ms delay
  const [showSkeleton, setShowSkeleton] = useState(false);

  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (loading) {
      timer = setTimeout(() => {
        setShowSkeleton(true);
      }, 150);
    } else {
      setShowSkeleton(false);
    }
    return () => clearTimeout(timer);
  }, [loading]);

  return (
    <div className="w-full max-w-[1280px] min-[1600px]:max-w-[1440px] mx-auto p-4 sm:p-6 md:p-8">
      {/* Error banner above verdict */}
      {error && (
        <div className="mb-6 p-4 bg-l1-tint border border-l1-border text-l1-text rounded-panel flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-l1-solid shrink-0 mt-0.5" strokeWidth={1.75} />
          <div>
            <div className="font-semibold text-[14px]">Service connection error</div>
            <div className="text-[13px] text-l1-text mt-0.5">
              Can't reach the triage service. Check that the backend is running on port 8000, then try again.
            </div>
          </div>
        </div>
      )}

      {/* Two columns layout */}
      <div className="flex flex-col lg:flex-row items-start gap-8">
        {/* Left column: Intake panel (sticky on desktop) */}
        <aside className="w-full lg:w-[380px] xl:w-[420px] shrink-0 lg:sticky lg:top-[84px] max-h-[calc(100vh-96px)] flex flex-col">
          <IntakePanel
            intake={intake}
            onChange={onChangeIntake}
            onSubmit={onSubmit}
            onClear={onClear}
            loading={loading}
            hasResult={result !== null}
          />
        </aside>

        {/* Right column: Decision column */}
        <main className="flex-1 w-full space-y-6">
          {loading ? (
            /* Loading: Skeleton of verdict tag only after 150ms delay */
            showSkeleton ? (
              <div className="bg-surface rounded-verdict border border-line p-7 shadow-raised min-h-[220px] animate-pulse flex flex-col md:flex-row gap-6">
                <div className="w-full md:w-[200px] h-[160px] bg-paper rounded-panel shrink-0" />
                <div className="flex-1 space-y-4 py-2">
                  <div className="h-8 bg-paper rounded-btn w-3/4" />
                  <div className="h-5 bg-paper rounded-btn w-1/2" />
                  <div className="flex gap-2 pt-4">
                    <div className="h-6 bg-paper rounded-pill w-32" />
                    <div className="h-6 bg-paper rounded-pill w-24" />
                    <div className="h-6 bg-paper rounded-pill w-20" />
                  </div>
                </div>
              </div>
            ) : (
              <div className="min-h-[220px]" />
            )
          ) : result ? (
            /* Render verdict and clinical details */
            <>
              <VerdictTag
                result={result}
                activeDemoCase={activeDemoCase}
                showAnnotations={showAnnotations}
              />

              <PipelineStages result={result} />

              <VitalsAndRules result={result} />

              <ComplaintDisplay
                complaintText={intake.complaint_text}
                redFlags={result.red_flags}
              />
            </>
          ) : (
            /* Empty state (before first result) */
            <div className="space-y-6">
              <div className="bg-surface rounded-panel border border-line p-8 text-center space-y-4">
                <div className="max-w-md mx-auto space-y-2">
                  <h3 className="text-h3 font-semibold text-ink">Ready for assessment</h3>
                  <p className="text-body text-ink-2 leading-relaxed">
                    Fill in the intake form and select Assess patient. The result appears here in under a second.
                  </p>
                </div>

                <div className="pt-2">
                  <button
                    onClick={onOpenDemoCases}
                    className="inline-flex items-center gap-1.5 px-4 py-2 bg-brand-tint text-brand hover:bg-brand-tint/80 rounded-btn text-[13px] font-semibold transition-colors"
                  >
                    <span>Or pick from 8 benchmark demo cases</span>
                    <ArrowUpRight className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <PipelineStages result={null} />
            </div>
          )}
        </main>
      </div>
    </div>
  );
};
