import React, { useState, useEffect } from "react";
import { Header } from "./components/Header";
import { DemoCasesPopover } from "./components/DemoCasesPopover";
import { ConsolePage } from "./pages/ConsolePage";
import { WaitingRoomPage } from "./pages/WaitingRoomPage";
import { RulesPage } from "./pages/RulesPage";
import { EvidencePage } from "./pages/EvidencePage";
import { PatientIntake, TriageResponse, DemoCase, QueuePatient } from "./types";

const DEFAULT_INTAKE: PatientIntake = {
  patient_id: "",
  age: 45,
  sex: "M",
  bp_sys: 120,
  bp_dia: 80,
  pulse: 72,
  temp_c: 36.8,
  spo2: 98,
  complaint_text: "",
  conditions: [],
  medications: [],
};

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<string>("console");
  const [intake, setIntake] = useState<PatientIntake>(DEFAULT_INTAKE);
  const [result, setResult] = useState<TriageResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [demoCases, setDemoCases] = useState<DemoCase[]>([]);
  const [selectedDemoCaseId, setSelectedDemoCaseId] = useState<string | null>(null);
  const [activeDemoCase, setActiveDemoCase] = useState<DemoCase | null>(null);
  const [showAnnotations, setShowAnnotations] = useState<boolean>(false);
  const [demoPopoverOpen, setDemoPopoverOpen] = useState<boolean>(false);
  const [modelsReady, setModelsReady] = useState<boolean>(false);

  const [waitingQueue, setWaitingQueue] = useState<QueuePatient[]>([]);

  // Check health and load demo cases on mount
  useEffect(() => {
    // Health check
    fetch("/api/health")
      .then((r) => r.json())
      .then((data) => {
        if (data.status === "ok" && data.models_loaded) {
          setModelsReady(true);
        }
      })
      .catch(() => setModelsReady(false));

    // Demo cases
    fetch("/api/demo-cases")
      .then((r) => r.json())
      .then((cases) => setDemoCases(cases))
      .catch((err) => console.error("Failed to load demo cases", err));

    // Seed queue for waiting room
    fetch("/api/seed-queue")
      .then((r) => r.json())
      .then((queue) => setWaitingQueue(queue))
      .catch((err) => console.error("Failed to load seed queue", err));
  }, []);

  const handleTriageSubmit = async (customIntake?: PatientIntake) => {
    const intakeToSubmit = customIntake || intake;
    setLoading(true);
    setError(null);

    try {
      const resp = await fetch("/api/triage", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(intakeToSubmit),
      });

      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }

      const data: TriageResponse = await resp.json();
      setResult(data);

      // Also add newly assessed patient to the waiting room queue
      const tokenNum = 100 + waitingQueue.length + 1;
      const newPatient: QueuePatient = {
        token: `T-${tokenNum}`,
        patient_id: intakeToSubmit.patient_id || `P-${tokenNum}`,
        age: intakeToSubmit.age ?? 0,
        sex: intakeToSubmit.sex,
        complaint_snippet: intakeToSubmit.complaint_text.slice(0, 70),
        triage_level: data.triage_level,
        level_label: data.level_label,
        department: data.primary_department,
        source_layer: data.source_layer,
        abstain_reason: data.abstain_reason,
        fcfs_rank: waitingQueue.length + 1,
        waited_seconds: 0,
        waited_display: "00:00",
        arrival_time: new Date().toISOString(),
        input: intakeToSubmit,
      };

      setWaitingQueue((prev) => [newPatient, ...prev]);
    } catch (e: any) {
      setError(e.message || "Failed to reach triage service");
    } finally {
      setLoading(false);
    }
  };

  const handleSelectDemoCase = (dc: DemoCase) => {
    setSelectedDemoCaseId(dc.id);
    setActiveDemoCase(dc);
    setIntake(dc.input);
    handleTriageSubmit(dc.input);
  };

  const handleClear = () => {
    setIntake(DEFAULT_INTAKE);
    setResult(null);
    setSelectedDemoCaseId(null);
    setActiveDemoCase(null);
    setError(null);
  };

  const handleLoadSeedQueue = async () => {
    try {
      const resp = await fetch("/api/seed-queue");
      if (resp.ok) {
        const queue = await resp.json();
        setWaitingQueue(queue);
      }
    } catch (e) {
      console.error("Failed to reload seed queue", e);
    }
  };

  return (
    <div className="min-h-screen bg-paper text-ink flex flex-col">
      <Header
        currentTab={currentTab}
        onTabChange={(tab) => setCurrentTab(tab)}
        onOpenDemoCases={() => setDemoPopoverOpen(true)}
        modelsReady={modelsReady}
        waitingCount={waitingQueue.length}
      />

      <DemoCasesPopover
        isOpen={demoPopoverOpen}
        onClose={() => setDemoPopoverOpen(false)}
        cases={demoCases}
        selectedCaseId={selectedDemoCaseId}
        onSelectCase={handleSelectDemoCase}
        showAnnotations={showAnnotations}
        onToggleAnnotations={(val) => setShowAnnotations(val)}
      />

      <div className="flex-1">
        {currentTab === "console" && (
          <ConsolePage
            intake={intake}
            onChangeIntake={(updated) => setIntake(updated)}
            result={result}
            loading={loading}
            error={error}
            onSubmit={() => handleTriageSubmit()}
            onClear={handleClear}
            activeDemoCase={activeDemoCase}
            showAnnotations={showAnnotations}
            onOpenDemoCases={() => setDemoPopoverOpen(true)}
          />
        )}

        {currentTab === "waiting-room" && (
          <WaitingRoomPage
            queue={waitingQueue}
            onLoadSeedQueue={handleLoadSeedQueue}
            onSelectPatient={(p) => {
              setIntake(p.input);
              setCurrentTab("console");
              handleTriageSubmit(p.input);
            }}
          />
        )}

        {currentTab === "rules" && <RulesPage />}

        {currentTab === "evidence" && <EvidencePage />}
      </div>
    </div>
  );
};

export default App;
