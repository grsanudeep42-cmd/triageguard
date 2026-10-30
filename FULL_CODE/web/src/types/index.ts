export interface VitalStatus {
  value: number | null;
  status: "normal" | "abnormal" | "critical" | "invalid" | "missing";
}

export interface RedFlag {
  id: string;
  label: string;
  matched_text: string;
}

export interface RuleFired {
  id: string;
  name: string;
  detail: string;
}

export interface Views {
  vitals_level: number | null;
  text_level: number | null;
}

export interface PipelineStage {
  stage_number: number;
  title: string;
  status: "passed" | "fired" | "not_reached" | "decided";
  result: string;
  detail?: string;
}

export interface TriageResponse {
  triage_level: number | null;
  level_label: string;
  primary_department: string | null;
  suggested_specialty: string | null;
  source_layer: "rule" | "screen" | "fusion" | "abstain";
  abstain_reason: "needs_remeasure" | "model_disagreement" | "high_conformal_uncertainty" | null;
  conformal_set: number[];
  p_emergency: number;
  tau: number;
  red_flags: RedFlag[];
  rules_fired: RuleFired[];
  vitals: {
    bp_sys: VitalStatus;
    bp_dia: VitalStatus;
    pulse: VitalStatus;
    temp_c: VitalStatus;
    spo2: VitalStatus;
  };
  views: Views;
  latency_ms: number;
  pipeline_stages: PipelineStage[];
}

export interface PatientIntake {
  patient_id?: string;
  age: number | null;
  sex: string;
  bp_sys: number | null;
  bp_dia: number | null;
  pulse: number | null;
  temp_c: number | null;
  spo2: number | null;
  complaint_text: string;
  conditions: string[];
  medications: string[];
}

export interface DemoCase {
  id: string;
  scenario: string;
  description: string;
  selection_method?: string;
  original_complaint_text?: string;
  injected_dramatic_phrase?: string;
  combined_gamed_complaint?: string;
  input: PatientIntake;
  expected_prediction: {
    triage_level: number | null;
    primary_department: string;
    suggested_specialty: string;
    red_flags: string[];
    source_layer: string;
    conformal_set: number[];
    abstain_reason: string | null;
    p_emergency: number;
    latency_ms: number;
  };
  ground_truth: {
    true_level: number;
    true_department: string;
  };
}

export interface RuleDefinition {
  id: string;
  description: string;
  plain_language_trigger: string;
  threshold: string;
  threshold_label: string;
  forced_level: number;
  applies_age_band: string;
  source: string;
}

export interface RedFlagDefinition {
  id: string;
  name: string;
  patterns_en: string[];
  patterns_tel: string[];
  patterns_hin: string[];
  all_patterns: string[];
  escalate_to_max_level: number;
  status: "Validated" | "Not validated";
}

export interface RulesResponse {
  rules: RuleDefinition[];
  red_flags: RedFlagDefinition[];
  firing_stats: {
    layer_escalation_decomposition?: Array<Record<string, any>>;
  };
}

export interface QueuePatient {
  token: string;
  patient_id: string;
  age: number;
  sex: string;
  complaint_snippet: string;
  triage_level: number | null;
  level_label: string;
  department: string | null;
  source_layer: string;
  abstain_reason: string | null;
  fcfs_rank: number;
  waited_seconds: number;
  waited_display: string;
  arrival_time: string;
  input: PatientIntake;
}
