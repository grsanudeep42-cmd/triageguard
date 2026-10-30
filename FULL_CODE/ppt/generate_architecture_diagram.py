import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_flawless_architecture():
    # 11 x 9.8 inches at 300 DPI (matches the 310 x 280 pt box on Slide 4)
    fig, ax = plt.subplots(figsize=(11, 9.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    fig.patch.set_facecolor('#ffffff')

    # Palette
    HEADER_BLUE = "#0f2744"
    TEXT_DARK = "#0f172a"
    TEXT_MUTED = "#334155"

    P_INTAKE = ("#f8fafc", "#0284c7", "#0284c7", "#ffffff", "#0369a1")
    P_T1     = ("#fff1f2", "#e11d48", "#e11d48", "#ffffff", "#be123c")
    P_T2     = ("#fff7ed", "#ea580c", "#ea580c", "#ffffff", "#c2410c")
    P_T3     = ("#eff6ff", "#2563eb", "#2563eb", "#ffffff", "#1d4ed8")
    P_T4     = ("#faf5ff", "#9333ea", "#9333ea", "#ffffff", "#7e22ce")
    P_T5     = ("#f0fdf4", "#16a34a", "#16a34a", "#ffffff", "#15803d")

    P_OUT_RED   = ("#fef2f2", "#dc2626", "#dc2626", "#ffffff", "#991b1b")
    P_OUT_PURP  = ("#faf5ff", "#7e22ce", "#7e22ce", "#ffffff", "#581c87")
    P_OUT_GREEN = ("#f0fdf4", "#15803d", "#15803d", "#ffffff", "#14532d")

    # Title Banner
    banner = FancyBboxPatch((3, 94.2), 94, 5.0, boxstyle="round,pad=0.2,rounding_size=0.8",
                            facecolor=HEADER_BLUE, edgecolor="none")
    ax.add_patch(banner)
    ax.text(50, 96.7, "TriageGuard: 5-Tier Safe Clinical Decision Architecture",
            fontsize=13, fontweight='bold', color="#ffffff", ha='center', va='center', fontfamily='sans-serif')

    def draw_card(x, y, w, h, palette, badge, title, subtitle, bullets=[]):
        bg, edge, b_bg, b_txt, t_col = palette
        
        card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.9",
                              facecolor=bg, edgecolor=edge, linewidth=1.4)
        ax.add_patch(card)
        
        top_y = y + h - 1.4
        bw = max(4.8, len(badge) * 0.92 + 2.2)
        b_patch = FancyBboxPatch((x + 1.2, top_y - 1.35), bw, 2.2, boxstyle="round,pad=0.1,rounding_size=0.5",
                                 facecolor=b_bg, edgecolor="none")
        ax.add_patch(b_patch)
        ax.text(x + 1.2 + bw/2.0, top_y - 0.25, badge, fontsize=7.2, fontweight='bold',
                color=b_txt, ha='center', va='center', fontfamily='sans-serif')
        
        ax.text(x + 1.2 + bw + 1.2, top_y - 0.25, title, fontsize=8.8, fontweight='bold',
                color=TEXT_DARK, va='center', fontfamily='sans-serif')
        
        sub_y = top_y - 2.6
        if subtitle:
            ax.text(x + 1.5, sub_y, subtitle, fontsize=7.5, fontweight='bold',
                    color=t_col, va='center', fontfamily='sans-serif')
            
        cur_y = sub_y - 2.1
        for b in bullets:
            ax.text(x + 1.8, cur_y, f"• {b}", fontsize=7.0, color=TEXT_MUTED,
                    va='center', fontfamily='sans-serif')
            cur_y -= 1.95

    # -------------------------------------------------------------------------
    # ROW 1: MULTIMODAL INTAKE (y = 80.5 to 92.5)
    # -------------------------------------------------------------------------
    # Left Input: Continuous Vitals
    draw_card(9.0, 80.5, 42.5, 12.0, P_INTAKE,
              "INPUT 1", "Continuous Vitals Stream", "Real-Time Sensor Ingestion",
              ["Continuous SBP, DBP, SpO2, HR, Temp",
               "Plausibility Gate: Flags loose leads & inverted inputs",
               "Shock Index (SI = HR / SBP) auto-computed on intake"])

    # Right Input: Multilingual Chief Complaint
    draw_card(53.5, 80.5, 42.5, 12.0, P_INTAKE,
              "INPUT 2", "Multilingual Chief Complaint", "Transliterated Clinical NLP",
              ["Native Telugu, Hindi, and English transliterated input",
               "Typo-Tolerant Damerau-Levenshtein n-gram extractor",
               "Extracts acute descriptors, onset duration, & trauma"])

    # -------------------------------------------------------------------------
    # ROW 2: TIER 1 & TIER 2 SAFETY GUARDS (y = 60.5 to 77.0)
    # -------------------------------------------------------------------------
    # Tier 1: Deterministic Vitals Rules
    draw_card(9.0, 60.5, 42.5, 16.5, P_T1,
              "TIER 1", "Deterministic Vitals Rules", "Hard Physiological Guardrails (rules.yaml)",
              ["13 deterministic thresholds, placeholder values pending clinical primer",
               "Mathematical Invariant: Can ONLY escalate, NEVER downgrade",
               "Critical vitals immediately bypass ML to eliminate delay",
               "Deterministic safety net; rule-triggered cases are never downgraded"])

    # Tier 2: Multilingual NLP Red Flag Screen
    draw_card(53.5, 60.5, 42.5, 16.5, P_T2,
              "TIER 2", "Multilingual Red Flag Screen", "Clinical Floor & Anti-Gaming Defense",
              ["10 clinical emergency syndromes (chest pain, stroke, syncope)",
               "Anti-Gaming Decoupling: Free-text NEVER assigns L1/L2 alone",
               "Locks Level 3 clinical floor (prevents queue-jumping hacks)",
               "79.6% held at true level, 18.4% routed to nurse, 0.6% (3/500) reached L1/L2"])

    # -------------------------------------------------------------------------
    # ROW 3: TIER 3 (ML CORE) & TIER 4 (SAFETY ABSTENTION) (y = 39.5 to 57.0)
    # -------------------------------------------------------------------------
    # Tier 3: Monotonic LightGBM & Conformal Screen
    draw_card(9.0, 39.5, 42.5, 17.5, P_T3,
              "TIER 3", "Monotonic ML & Conformal Screen", "Calibrated Safety Bounds (conformal.joblib)",
              ["Monotonic LightGBM + Text Model (TF-IDF + Logistic Regression)",
               "Platt temperature scaling: Posterior probability calibration",
               "Mondrian Conformal Screen (τ = 0.0922 at α = 0.05)",
               "Calibrated screen targeting 95% coverage; measured recall 92.8%",
               "76.7% (138/180), vague-complaint stress, vs Baseline 68.9% (124/180)"])

    # Tier 4: Safety Abstention Gate
    draw_card(53.5, 39.5, 42.5, 17.5, P_T4,
              "TIER 4", "Safety Abstention Gate", "Refuse to Guess ('Doctor's Conscience')",
              ["Triggers immediate nurse referral on 3 safety conditions:",
               "  1. Sensor failure (≥ 2 corrupted/physically impossible vitals)",
               "  2. Model discordance (≥ 2 acuity levels inter-model conflict)",
               "  3. Ambiguity / High Uncertainty (Prediction set |C(x)| ≥ 3)",
               "Safely diverts 17% of clean cases (37.1% noisy) to nurse triage"])

    # -------------------------------------------------------------------------
    # ROW 4: TIER 5 (BAYES ENGINE & 12-DEPT ROUTER) (y = 19.5 to 35.5)
    # -------------------------------------------------------------------------
    draw_card(9.0, 19.5, 87.0, 16.0, P_T5,
              "TIER 5", "Asymmetric Bayes Risk Minimization & 12-Dept Routing",
              "Optimal Acuity Assignment & Clinical Specialty Placement (12.8 ms mean, 16.2 ms P95)",
              ["Evaluates expected clinical loss over Conformal Set C(x) with a 20x asymmetric penalty on under-triage",
               "Selects final ESI Level 1–5 minimizing patient risk + routes to 12 specialties at 88.5% (on the 3,737 non-abstained cases)",
               "Bayes risk minimization with 100% on-premise execution; no patient data leaves the hospital network",
               "Produces comprehensive audit trail: rules triggered, conformal sets, model probabilities, and triage rationale"])

    # -------------------------------------------------------------------------
    # ROW 5: OPERATIONAL CLINICAL OUTCOMES (y = 2.5 to 15.5)
    # -------------------------------------------------------------------------
    # Outcome 1: Immediate Resuscitation Bypass (Left)
    draw_card(2.0, 2.5, 29.5, 13.0, P_OUT_RED,
              "ESI 1 / 2", "Emergency Resuscitation", "Immediate Bed Assignment",
              ["ESI Level 1 (Resuscitation)",
               "ESI Level 2 (Emergent)",
               "Bypasses general OPD queue",
               "Trauma & ICU team paged"])

    # Outcome 2: Dynamic Waiting Room & Dispatch (Center)
    draw_card(34.5, 2.5, 32.5, 13.0, P_OUT_GREEN,
              "DISPATCH", "Acuity Queue & 12 Depts", "Dynamic Priority Queue",
              ["Acuity-sorted waiting room",
               "12 Departments (Cardio, etc.)",
               "FIFO within acuity band",
               "Live real-time nurse console"])

    # Outcome 3: Nurse Escalation (Right)
    draw_card(70.0, 2.5, 28.0, 13.0, P_OUT_PURP,
              "NURSE", "Nurse Escalation", "Human-in-the-Loop Review",
              ["Safe refusal to guess",
               "Remeasure noisy vitals",
               "Senior triage nurse audit",
               "Eliminates silent failure"])

    # -------------------------------------------------------------------------
    # CONNECTING ARROWS & LABELS (Zero overlap)
    # -------------------------------------------------------------------------
    # 1. Input 1 -> Tier 1
    ax.annotate('', xy=(30.2, 77.0), xytext=(30.2, 80.5),
                arrowprops=dict(arrowstyle="-|>", color="#0284c7", lw=2.0, mutation_scale=12))

    # 2. Input 2 -> Tier 2
    ax.annotate('', xy=(74.8, 77.0), xytext=(74.8, 80.5),
                arrowprops=dict(arrowstyle="-|>", color="#0284c7", lw=2.0, mutation_scale=12))

    # 3. TIER 1 DIRECT EMERGENCY BYPASS (Left Gutter: x = 5.0)
    ax.plot([9.0, 5.0, 5.0], [68.5, 68.5, 15.7], color="#dc2626", lw=2.2, linestyle='--')
    ax.annotate('', xy=(5.0, 15.5), xytext=(5.0, 17.0),
                arrowprops=dict(arrowstyle="-|>", color="#dc2626", lw=2.2, mutation_scale=14))
    
    ax.text(5.0, 42.0, "Vitals Rule Triggered\n(Immediate Bypass)", fontsize=6.8, fontweight='bold',
            color="#b91c1c", ha='center', va='center', rotation=90,
            bbox=dict(boxstyle="round,pad=0.25", fc="#fff1f2", ec="#e11d48", lw=0.9))

    # 4. Tier 1 Normal Vitals -> Tier 3 ML
    ax.annotate('', xy=(30.2, 57.0), xytext=(30.2, 60.5),
                arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=2.0, mutation_scale=12))
    ax.text(30.2, 58.7, "Normal Vitals", fontsize=6.6, fontweight='bold', color="#334155",
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec="#cbd5e1", lw=0.7))

    # 5. Tier 2 Red Flags -> Tier 3 (Sets Acuity Floor)
    ax.annotate('', xy=(51.5, 52.0), xytext=(60.0, 60.5),
                arrowprops=dict(arrowstyle="-|>", color="#ea580c", lw=1.8, mutation_scale=12))
    ax.text(58.0, 58.0, "Lock L3 Floor", fontsize=6.5, fontweight='bold', color="#c2410c",
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.15", fc="#fff7ed", ec="#ea580c", lw=0.7))

    # 6. Tier 3 ML <-> Tier 4 Abstention Check
    ax.annotate('', xy=(53.5, 48.0), xytext=(51.5, 48.0),
                arrowprops=dict(arrowstyle="<|-|>", color="#9333ea", lw=1.8, mutation_scale=11))
    ax.text(52.5, 50.0, "Verify", fontsize=6.5, fontweight='bold', color="#7e22ce",
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.15", fc="#faf5ff", ec="#9333ea", lw=0.7))

    # 7. Tier 3 ML -> Tier 5 Bayes Engine (Down to Tier 5)
    ax.annotate('', xy=(30.2, 35.5), xytext=(30.2, 39.5),
                arrowprops=dict(arrowstyle="-|>", color="#16a34a", lw=2.0, mutation_scale=12))
    ax.text(30.2, 37.5, "Prediction Set C(x)", fontsize=6.6, fontweight='bold', color="#15803d",
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.15", fc="#f0fdf4", ec="#16a34a", lw=0.7))

    # 8. Tier 4 Abstention -> Outcome 3 (Nurse Escalation)
    # Notice: To avoid crossing Tier 5, let's route out of Tier 4 along the RIGHT GUTTER (x = 97.5)!
    ax.plot([96.0, 97.5, 97.5, 84.0, 84.0], [48.0, 48.0, 17.5, 17.5, 15.7], color="#9333ea", lw=2.2, linestyle='--')
    ax.annotate('', xy=(84.0, 15.5), xytext=(84.0, 17.0),
                arrowprops=dict(arrowstyle="-|>", color="#9333ea", lw=2.2, mutation_scale=14))
    ax.text(97.5, 33.0, "Sensor Fault / Discordance / Ambiguity\n(Abstain to Nurse)", fontsize=6.6, fontweight='bold',
            color="#6b21a8", ha='center', va='center', rotation=270,
            bbox=dict(boxstyle="round,pad=0.25", fc="#faf5ff", ec="#9333ea", lw=0.9))

    # 9. Tier 5 Bayes Engine -> Outcome 2 (Dispatch & Queue)
    ax.annotate('', xy=(50.8, 15.5), xytext=(50.8, 19.5),
                arrowprops=dict(arrowstyle="-|>", color="#15803d", lw=2.2, mutation_scale=13))
    ax.text(50.8, 17.5, "Assigned Acuity L1–L5 + 12-Dept Routing", fontsize=6.6, fontweight='bold', color="#14532d",
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.15", fc="#f0fdf4", ec="#15803d", lw=0.7))

    plt.tight_layout()
    out_path = "ppt/architecture_diagram.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Generated flawless architecture at {out_path}!")

if __name__ == '__main__':
    create_flawless_architecture()
