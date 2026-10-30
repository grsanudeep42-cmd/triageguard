import os
import pymupdf

def build_final_pdf():
    src_pdf = "ppt/Round1_Template submission.pdf"
    out_pdf = "ppt/Round1_TriageGuard_Submission.pdf"
    
    doc = pymupdf.open(src_pdf)
    print(f"Loaded template {src_pdf} ({len(doc)} pages).")

    WHITE = (1.0, 1.0, 1.0)
    BG_BLUE = (0.122, 0.278, 0.533)
    SKY_BLUE = (0.22, 0.74, 0.97)
    ACCENT_CYAN = (0.0, 0.643, 0.937)

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    p1 = doc[0]
    for needle in ["[Enter Team Name]", "[Enter Leader Name]", "[Select Track]"]:
        for r in p1.search_for(needle):
            p1.add_redact_annot(r, fill=BG_BLUE)
    p1.apply_redactions()

    # Team Name
    p1.insert_text((79.3, 246.0), "Ronin Labs", fontname="hebo", fontsize=15, color=WHITE)
    
    # Team Leader (Only Anudeep Gedela R S)
    p1.insert_text((403.6, 246.0), "Anudeep Gedela R S", fontname="hebo", fontsize=15, color=WHITE)

    # Track
    track_text = "H1. OPD Triage Assistant for an Overloaded Government Hospital"
    track_rect = pymupdf.Rect(79.3, 295.0, 380.0, 360.0)
    p1.insert_textbox(track_rect, track_text, fontname="hebo", fontsize=11, color=SKY_BLUE)

    # Project Headline & Clinical Disclaimer
    p1.insert_text((100.0, 185.0), "TriageGuard: Clinical Decision Support System for Emergency Triage", fontname="hebo", fontsize=13, color=WHITE)
    p1.insert_textbox(pymupdf.Rect(50.0, 198.0, 670.0, 215.0), "Prototype on 30,000 synthetic encounters; prospective validation required before clinical use.", fontname="helv", fontsize=7.8, color=SKY_BLUE, align=pymupdf.TEXT_ALIGN_CENTER)

    # =========================================================================
    # SLIDE 2: USE CASE (Rich Content, Balanced Density)
    # =========================================================================
    p2 = doc[1]
    p2.add_redact_annot(pymupdf.Rect(40.0, 105.0, 690.0, 235.0), fill=WHITE)
    p2.add_redact_annot(pymupdf.Rect(40.0, 255.0, 690.0, 398.0), fill=WHITE)
    p2.apply_redactions()

    html2_prob = """
<div style='font-family: sans-serif; font-size: 8.8pt; color: #1e293b; line-height: 1.38;'>
  <p style='margin: 0 0 6pt 0;'><b>• Clinical Pain Point &amp; Cognitive Overload:</b> High-volume government district casualty wards handle 800+ patients daily with &lt;90 seconds per triage encounter. Severe cognitive fatigue leads to a documented 15–20% under-triage rate on subtle, life-threatening presentations (silent STEMI, occult sepsis, internal shock).</p>
  <p style='margin: 0 0 6pt 0;'><b>• Multilingual &amp; Demographic Communication Barriers:</b> Over 65% of patients in regional and district hospitals communicate in Telugu or Hindi, frequently describing life-critical symptoms through regional metaphors (<i>"gunde lo manta"</i>, <i>"dummu pattukundi"</i>) that standard medical NLP and non-native clinical staff misunderstand.</p>
  <p style='margin: 0;'><b>• Quantified Patient &amp; Systemic Impact:</b> Every 10-minute delay in emergency resuscitation increases patient mortality by 8–12%. Current hospital triage relies on paper tokens with 30% inter-nurse variability, while black-box LLMs hallucinate on corrupted vitals and get manipulated by patient queue-gaming keywords.</p>
</div>
"""
    p2.insert_htmlbox(pymupdf.Rect(43.2, 108.0, 680.0, 238.0), html2_prob)

    html2_mkt = """
<div style='font-family: sans-serif; font-size: 8.8pt; color: #1e293b; line-height: 1.38;'>
  <p style='margin: 0 0 6pt 0;'><b>• Market Size &amp; Rapid Sector Growth:</b> The Global Clinical Decision Support System (CDSS) market stands at &#36;2.1B (2025) and is projected to reach &#36;4.8B by 2030 (18.2% CAGR). Digital emergency adoption is accelerated by India's Ayushman Bharat Digital Mission (ABDM) and public smart-hospital modernization mandates.</p>
  <p style='margin: 0 0 6pt 0;'><b>• Target Audience &amp; Hospital Scale:</b> Primary deployment targets high-footfall government district casualty wards (AP, Telangana, PAN-India public healthcare infrastructure) and tertiary trauma networks requiring auditable ESI Level 1–5 intake.</p>
  <p style='margin: 0;'><b>• Competitive Advantage &amp; Strategic Moat:</b> Manual ESI/Manchester triage is slow, paper-bound, and subjective; generic generative AI lacks clinical safety bounds. TriageGuard provides deterministic safety rules, calibrated Mondrian bounds, 12.8 ms mean, 16.2 ms P95 latency, and on-premise execution; no patient data leaves the hospital network.</p>
</div>
"""
    p2.insert_htmlbox(pymupdf.Rect(43.2, 258.0, 680.0, 396.0), html2_mkt)

    # =========================================================================
    # SLIDE 3: PROPOSED SOLUTION (Full-Width UI Prototype)
    # =========================================================================
    p3 = doc[2]
    p3.add_redact_annot(pymupdf.Rect(40.0, 105.0, 690.0, 204.0), fill=WHITE)
    p3.add_redact_annot(pymupdf.Rect(35.0, 204.0, 685.0, 398.0), fill=WHITE)
    p3.apply_redactions()

    html3 = """
<div style='font-family: sans-serif; font-size: 8.0pt; color: #1e293b; line-height: 1.30;'>
  <p style='margin: 0 0 3pt 0;'><b>• TriageGuard CDSS Architecture:</b> Prototype on 30,000 synthetic encounters; prospective validation required before clinical use. Ultra-fast inference (12.8 ms mean, 16.2 ms P95) classifies ED acuity (ESI Levels 1–5) and routes to 12 specialties (88.5% on the 3,737 non-abstained cases).</p>
  <p style='margin: 0 0 3pt 0;'><b>• 5-Stage Safety Pipeline:</b> (1) Input Cleaning &amp; Plausibility &rarr; (2) Deterministic Rules (13 deterministic thresholds, placeholder values pending clinical primer) &amp; 10 Red Flags &rarr; (3) Monotonic LightGBM + Text Model (TF-IDF + Logistic Regression) &rarr; (4) Mondrian Conformal Screen &amp; Safety Abstention Gate &rarr; (5) Cost Bayes Engine &amp; 12-Dept Router.</p>
  <p style='margin: 0 0 3pt 0;'><b>• Benchmark Metrics:</b> Emergency recall: 76.7% (138/180), vague-complaint stress, vs Baseline 68.9% (124/180); silent misses: 5 vs 11 on clean data (54%); 19 vs 56 under vague stress (66%); 4.4% non-emergency over-escalation (192/4,320); anti-gaming: 79.6% held at true level, 18.4% routed to nurse, 0.6% (3/500) reached L1/L2.</p>
  <p style='margin: 0;'><b>• Safety Philosophy:</b> Rule-triggered cases are never downgraded. ML never holds a solitary life-or-death vote. Fails safely by abstaining 17% of clean cases (37.1% under 20% sensor corruption) to senior triage nurses.</p>
</div>
"""
    p3.insert_htmlbox(pymupdf.Rect(43.2, 108.0, 680.0, 204.0), html3)

    img3_path = "results/ui_screens/case1_true_emergency_forced_by_rule_1440.png"
    img3_rect = pymupdf.Rect(43.2, 208.0, 676.8, 396.0)
    p3.draw_rect(img3_rect, color=(0.82, 0.88, 0.95), width=1.0)
    p3.insert_image(img3_rect, filename=img3_path, keep_proportion=True)

    # =========================================================================
    # SLIDE 4: DESIGN STEPS & ARCHITECTURE
    # =========================================================================
    p4 = doc[3]
    p4.add_redact_annot(pymupdf.Rect(40.0, 105.0, 360.0, 396.0), fill=WHITE)
    p4.add_redact_annot(pymupdf.Rect(365.0, 105.0, 690.0, 396.0), fill=WHITE)
    p4.apply_redactions()

    html4 = """
<div style='font-family: sans-serif; font-size: 7.9pt; color: #1e293b; line-height: 1.34;'>
  <p style='margin: 0 0 6pt 0;'><b>1. Multimodal Intake &amp; Sensor Validation:</b><br/>Continuous vitals (BP, SpO2, HR, Temp) + chief complaint text. Plausibility filters detect sensor faults; &ge;2 invalid vitals trigger nurse remeasurement.</p>
  <p style='margin: 0 0 6pt 0;'><b>2. Rule &amp; Red-Flag Floor Screen:</b><br/>13 deterministic thresholds, placeholder values pending clinical primer, plus 10 red flags (Telugu, Hindi, English n-grams). Free text enforces Level 3 floor without allowing artificial Level 1/2 gaming (79.6% held at true level, 18.4% routed to nurse, 0.6% [3/500] reached L1/L2).</p>
  <p style='margin: 0 0 6pt 0;'><b>3. Monotonic LightGBM &amp; Text Model:</b><br/>Monotonic gradient boosting (SpO2 &darr;, Shock Index &uarr;) fused with TF-IDF + logistic regression text model. Platt temperature scaling produces calibrated probabilities.</p>
  <p style='margin: 0 0 6pt 0;'><b>4. Mondrian Conformal Screen &amp; Safety Gate:</b><br/>Calibrated screen targeting 95% coverage; measured recall 92.8% (167/180, CI 88.0-95.7%). Safety gate abstains 17% of clean cases (37.1% under 20% corruption).</p>
  <p style='margin: 0;'><b>5. Asymmetric Bayes Risk &amp; 12-Dept Router:</b><br/>Minimizes expected risk with a 20x under-triage penalty inside prediction set C(x) and routes to 12 specialties at 88.5% (on the 3,737 non-abstained cases) in 12.8 ms mean, 16.2 ms P95.</p>
</div>
"""
    p4.insert_htmlbox(pymupdf.Rect(43.2, 108.0, 355.0, 396.0), html4)

    arch_path = "ppt/architecture_diagram.png"
    arch_rect = pymupdf.Rect(365.0, 106.0, 685.0, 396.0)
    p4.draw_rect(arch_rect, color=(0.82, 0.88, 0.95), width=0.8)
    p4.insert_image(arch_rect, filename=arch_path, keep_proportion=True)

    # =========================================================================
    # SLIDE 5: CHALLENGES & MITIGATION (Evenly Distributed with Metrics)
    # =========================================================================
    p5 = doc[4]
    p5.add_redact_annot(pymupdf.Rect(43.2, 100.0, 680.0, 162.0), fill=WHITE)
    p5.add_redact_annot(pymupdf.Rect(43.2, 185.0, 680.0, 252.0), fill=WHITE)
    p5.add_redact_annot(pymupdf.Rect(43.2, 275.0, 680.0, 390.0), fill=WHITE)
    p5.apply_redactions()

    html5_c1 = """
<div style='font-family: sans-serif; font-size: 8.0pt; color: #1e293b; line-height: 1.34;'>
  <p style='margin: 0 0 2pt 0;'><b style='color: #b91c1c;'>Problem:</b> Patients inject alarming keywords (<i>"unbearable dying pain please see me now"</i>, <i>"chast pain emergency"</i>) at kiosks or intake desks to jump the triage waiting queue.</p>
  <p style='margin: 0 0 2pt 0;'><b style='color: #166534;'>Mitigation:</b> Architectural channel decoupling: free-text red flags <b>can never assign Level 1 or 2 by themselves</b>; they only enforce an acuity floor of Level 3.</p>
  <p style='margin: 0;'><b style='color: #0369a1;'>Empirical Result:</b> In 500 stress-tested gaming attacks: <b>79.6% held at true level, 18.4% routed to nurse, 0.6% (3/500) reached L1/L2</b>.</p>
</div>
"""
    p5.insert_htmlbox(pymupdf.Rect(57.7, 102.0, 680.0, 160.0), html5_c1)

    html5_c2 = """
<div style='font-family: sans-serif; font-size: 8.0pt; color: #1e293b; line-height: 1.34;'>
  <p style='margin: 0 0 2pt 0;'><b style='color: #b91c1c;'>Problem:</b> Busy casualty wards suffer from detached pulse oximeter probes, inverted systolic/diastolic values, Fahrenheit entered in Celsius fields, or transient signal dropouts.</p>
  <p style='margin: 0 0 2pt 0;'><b style='color: #166534;'>Mitigation:</b> Deterministic physical plausibility gates. Rather than blindly guessing with statistical imputation, TriageGuard triggers <code>needs_remeasure</code> nurse intervention when &ge;2 vitals fail.</p>
  <p style='margin: 0;'><b style='color: #0369a1;'>Empirical Result:</b> Abstention scales safely from <b>17% of clean cases</b> to <b>37.1% under 20% sensor corruption</b> while strictly preserving <b>92.2% emergency recall (166/180)</b>.</p>
</div>
"""
    p5.insert_htmlbox(pymupdf.Rect(57.7, 188.0, 680.0, 250.0), html5_c2)

    html5_c3 = """
<div style='font-family: sans-serif; font-size: 8.0pt; color: #1e293b; line-height: 1.34;'>
  <p style='margin: 0 0 2pt 0;'><b style='color: #b91c1c;'>Problem:</b> Standard ML models optimize aggregate accuracy, silently missing rare crises (1–4% prevalence) such as atypical silent STEMI or occult respiratory failure.</p>
  <p style='margin: 0 0 2pt 0;'><b style='color: #166534;'>Mitigation:</b> Mondrian Conformal Safety Screen + 20x Asymmetric Cost Matrix. Calibrated on split emergencies (&tau;=0.0922, &alpha;=0.05).</p>
  <p style='margin: 0;'><b style='color: #0369a1;'>Empirical Result:</b> Emergency recall: <b>76.7% (138/180), vague-complaint stress, vs Baseline 68.9% (124/180)</b>. Silent misses: <b>5 vs 11 on clean data (54%); 19 vs 56 under vague stress (66%)</b>.</p>
</div>
"""
    p5.insert_htmlbox(pymupdf.Rect(57.7, 276.0, 680.0, 360.0), html5_c3)

    # =========================================================================
    # SLIDE 6: UNIQUE SELLING PROPOSITION (USP)
    # =========================================================================
    p6 = doc[5]
    p6.add_redact_annot(pymupdf.Rect(40.0, 112.0, 250.0, 255.0), fill=WHITE)
    p6.add_redact_annot(pymupdf.Rect(255.0, 112.0, 465.0, 255.0), fill=WHITE)
    p6.add_redact_annot(pymupdf.Rect(470.0, 112.0, 690.0, 255.0), fill=WHITE)
    p6.add_redact_annot(pymupdf.Rect(43.2, 288.0, 680.0, 396.0), fill=WHITE)
    p6.apply_redactions()

    html6_d1 = """
<div style='font-family: sans-serif; font-size: 7.6pt; color: #1e293b; line-height: 1.32; text-align: center;'>
  <p style='margin: 0 0 2pt 0;'><b>Swiss-Cheese Safety Model</b></p>
  <p style='margin: 0 0 3pt 0; color: #be123c; font-size: 7.2pt;'><b>Deterministic Physiological Net</b></p>
  <p style='margin: 0; color: #475569;'>13 deterministic thresholds, placeholder values pending clinical primer, wrap the ML core. Invariant: Rules can ONLY escalate, NEVER downgrade. SpO2 &lt; 85% or profound shock overrides statistical predictions; rule-triggered cases are never downgraded.</p>
</div>
"""
    p6.insert_htmlbox(pymupdf.Rect(43.2, 114.0, 245.0, 250.0), html6_d1)

    html6_d2 = """
<div style='font-family: sans-serif; font-size: 7.6pt; color: #1e293b; line-height: 1.32; text-align: center;'>
  <p style='margin: 0 0 2pt 0;'><b>Mondrian Conformal Screen</b></p>
  <p style='margin: 0 0 3pt 0; color: #1d4ed8; font-size: 7.2pt;'><b>Calibrated Statistical Bounds</b></p>
  <p style='margin: 0; color: #475569;'>Replaces heuristics with calibrated prediction sets C(x). Calibrated screen targeting 95% coverage; measured recall 92.8% (167/180, CI 88.0-95.7%) while limiting non-emergency over-escalation to 4.4% (192/4,320).</p>
</div>
"""
    p6.insert_htmlbox(pymupdf.Rect(258.0, 114.0, 462.0, 250.0), html6_d2)

    html6_d3 = """
<div style='font-family: sans-serif; font-size: 7.6pt; color: #1e293b; line-height: 1.32; text-align: center;'>
  <p style='margin: 0 0 2pt 0;'><b>Honest Clinical Abstention</b></p>
  <p style='margin: 0 0 3pt 0; color: #7e22ce; font-size: 7.2pt;'><b>Doctor's Conscience Safety Gate</b></p>
  <p style='margin: 0; color: #475569;'>Clinical AI must know when NOT to guess. TriageGuard safely abstains under &ge;2 corrupted vitals, inter-model discordance (&ge;2 levels), or high ambiguity, routing 17% of clean cases (37.1% under 20% sensor corruption) safely to senior triage nurses.</p>
</div>
"""
    p6.insert_htmlbox(pymupdf.Rect(475.0, 114.0, 676.8, 250.0), html6_d3)

    html6_ca = """
<div style='font-family: sans-serif; font-size: 8.3pt; color: #1e293b; line-height: 1.36;'>
  <p style='margin: 0 0 3.5pt 0;'><b>• On-Premise Execution &amp; Local Privacy:</b> Inference latency of 12.8 ms mean, 16.2 ms P95 enables 100% on-premise execution; no patient data leaves the hospital network, eliminating recurring cloud API costs.</p>
  <p style='margin: 0 0 3.5pt 0;'><b>• Multimodal &amp; Multilingual Synergy:</b> Fuses monotonic LightGBM with TF-IDF + logistic regression text model and typo-tolerant n-grams (Telugu, Hindi, English) to eliminate language and spelling barriers in district casualty wards.</p>
  <p style='margin: 0;'><b>• Asymmetric Bayes Risk &amp; 12-Dept Routing:</b> Evaluates expected loss with a 20x under-triage penalty; routes to 12 specialties at 88.5% (on the 3,737 non-abstained cases). Prototype on 30,000 synthetic encounters; prospective validation required before clinical use.</p>
</div>
"""
    p6.insert_htmlbox(pymupdf.Rect(43.2, 292.0, 680.0, 396.0), html6_ca)

    # =========================================================================
    # SLIDE 7: DEMO & POC VIDEO (Clean Interactive Action Buttons)
    # =========================================================================
    p7 = doc[6]
    p7.add_redact_annot(pymupdf.Rect(40.0, 75.0, 680.0, 400.0), fill=WHITE)
    p7.apply_redactions()

    # Section title
    p7.insert_text((43.2, 85.0), "Live Demonstration & Interactive Prototype System", fontname="hebo", fontsize=13, color=BG_BLUE)

    # Video Player Container
    player_rect = pymupdf.Rect(55.0, 96.0, 665.0, 335.0)
    p7.draw_rect(player_rect, color=ACCENT_CYAN, width=1.5)

    img_path = "results/ui_screens/waiting_room_1440.png"
    img_rect = pymupdf.Rect(56.0, 97.0, 664.0, 334.0)
    p7.insert_image(img_rect, filename=img_path, keep_proportion=False)

    # 4 Perfectly Balanced Interactive Action Buttons (Identical 150pt width)
    # Full un-truncated URLs on slide 7, zero ellipses (...)
    buttons = [
        {
            "rect": pymupdf.Rect(43.2, 345.0, 193.2, 392.0),
            "bg": (0.99, 0.95, 0.95), "border": (0.86, 0.15, 0.15),
            "title": "[VIDEO] YouTube Demo",
            "sub": "https://youtu.be/yqNZq92zks0",
            "uri": "https://youtu.be/yqNZq92zks0"
        },
        {
            "rect": pymupdf.Rect(204.2, 345.0, 354.2, 392.0),
            "bg": (0.94, 0.99, 0.95), "border": (0.09, 0.64, 0.29),
            "title": "[CODE] GitHub Repo",
            "sub": "https://github.com/grsanudeep42-cmd/triageguard",
            "uri": "https://github.com/grsanudeep42-cmd/triageguard"
        },
        {
            "rect": pymupdf.Rect(365.2, 345.0, 515.2, 392.0),
            "bg": (0.94, 0.96, 1.0), "border": (0.15, 0.39, 0.92),
            "title": "[DEPLOY] Live App",
            "sub": "https://web-production-bb683.up.railway.app",
            "uri": "https://web-production-bb683.up.railway.app"
        },
        {
            "rect": pymupdf.Rect(526.2, 345.0, 676.2, 392.0),
            "bg": (0.98, 0.96, 1.0), "border": (0.49, 0.23, 0.93),
            "title": "[DOCS] Swagger API",
            "sub": "https://web-production-bb683.up.railway.app/docs",
            "uri": "https://web-production-bb683.up.railway.app/docs"
        }
    ]

    # Delete any lingering link annotations
    for l in p7.get_links():
        p7.delete_link(l)

    # Insert button visual cards and solid clickable link annotations
    for b in buttons:
        p7.draw_rect(b["rect"], color=b["border"], fill=b["bg"], width=1.2)
        p7.insert_text((b["rect"].x0 + 8, b["rect"].y0 + 17), b["title"], fontname="hebo", fontsize=8.6, color=(0.06, 0.09, 0.16))
        p7.insert_text((b["rect"].x0 + 6, b["rect"].y0 + 33), b["sub"], fontname="helv", fontsize=5.3, color=(0.20, 0.25, 0.35))
        p7.insert_link({"kind": pymupdf.LINK_URI, "from": b["rect"], "uri": b["uri"]})

    # Save final optimized PDF
    doc.save(out_pdf, deflate=True, garbage=4, clean=True)
    print(f"Final submission PDF saved successfully to {out_pdf} (optimized)!")

if __name__ == '__main__':
    build_final_pdf()
