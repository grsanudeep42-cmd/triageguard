import os
import pymupdf

def build_pdf():
    src_pdf = "ppt/Round1_Template submission.pdf"
    out_pdf = "ppt/Round1_TriageGuard_Submission.pdf"
    
    doc = pymupdf.open(src_pdf)
    print(f"Loaded {src_pdf} with {len(doc)} pages.")

    WHITE = (1.0, 1.0, 1.0)
    BG_BLUE = (0.122, 0.278, 0.533)
    SKY_BLUE = (0.22, 0.74, 0.97)
    NAVY = (0.122, 0.278, 0.533)

    # =========================================================================
    # PAGE 1: TITLE SLIDE
    # =========================================================================
    p1 = doc[0]
    # Redact placeholders
    for needle in ["[Enter Team Name]", "[Enter Leader Name]", "[Select Track]"]:
        for r in p1.search_for(needle):
            p1.add_redact_annot(r, fill=BG_BLUE)
    p1.apply_redactions()

    # Insert Team Name
    p1.insert_text((79.3, 246.0), "Ronin Labs", fontname="hebo", fontsize=15, color=WHITE)

    # Insert Leader & Members
    p1.insert_text((403.6, 246.0), "Anudeep Gedela R S", fontname="hebo", fontsize=15, color=WHITE)
    p1.insert_text((403.6, 259.0), "Anga Yashwanth, Dileep Kumar Nagothi, Pavan Deepak Seeramreddi", fontname="helv", fontsize=8.2, color=(0.82, 0.90, 1.0))

    # Insert Track (wrap neatly)
    track_text = "H1. OPD Triage Assistant for an Overloaded Government Hospital"
    track_rect = pymupdf.Rect(79.3, 295.0, 380.0, 360.0)
    p1.insert_textbox(track_rect, track_text, fontname="hebo", fontsize=11, color=SKY_BLUE)

    # Add Project Subtitle below AI Frontier Challenge
    p1.insert_text((100.0, 185.0), "TriageGuard: Clinical Decision Support System for Emergency Triage", fontname="hebo", fontsize=13, color=WHITE)

    # =========================================================================
    # PAGE 2: USE CASE
    # =========================================================================
    p2 = doc[1]
    # Redact placeholder bullets
    p2.add_redact_annot(pymupdf.Rect(40.0, 110.0, 690.0, 228.0), fill=WHITE)
    p2.add_redact_annot(pymupdf.Rect(40.0, 260.0, 690.0, 395.0), fill=WHITE)
    p2.apply_redactions()

    html2_prob = """
<div style='font-family: sans-serif; font-size: 8.5pt; color: #1e293b; line-height: 1.35;'>
  <p style='margin: 0 0 6pt 0;'><b>• Clinical Pain Point:</b> Overcrowded government hospital OPDs handle 800+ patients/day with &lt;90s per triage encounter, leading to severe cognitive fatigue and a 15–20% under-triage rate on subtle life-threats (e.g. atypical STEMI, hypoxia, internal shock).</p>
  <p style='margin: 0 0 6pt 0;'><b>• Affected Stakeholders:</b> Patients face silent deterioration in waiting rooms; regional language barriers (Telugu, Hindi) worsen miscommunication. Clinicians face high burnout; hospital administration faces ICU bed saturation and medico-legal liability.</p>
  <p style='margin: 0;'><b>• Current Impact:</b> A 10-minute delay in resuscitation raises mortality by 8–12%. Existing black-box AI fails: hallucinates on noisy sensors and gets tricked by hyperbolic complaints.</p>
</div>
"""
    p2.insert_htmlbox(pymupdf.Rect(43.2, 112.0, 680.0, 225.0), html2_prob)

    html2_mkt = """
<div style='font-family: sans-serif; font-size: 8.5pt; color: #1e293b; line-height: 1.35;'>
  <p style='margin: 0 0 6pt 0;'><b>• Market Size &amp; Rapid Growth:</b> Global CDSS market is &#36;2.1B (2025), expanding to &#36;4.8B by 2030 (18.2% CAGR). Digital emergency healthcare adoption is accelerating under India's Ayushman Bharat Digital Mission (ABDM).</p>
  <p style='margin: 0 0 6pt 0;'><b>• Target Demographics:</b> High-volume government district casualty wards facing bilingual footfalls (AP, Telangana, PAN-India) and tertiary multi-specialty trauma networks (Apollo, Care, KIMS) requiring standardized ESI triage.</p>
  <p style='margin: 0;'><b>• Competitive Landscape:</b> Manual ESI/Manchester triage is slow with 30% inter-rater inconsistency; generic LLMs lack safety bounds; TriageGuard provides verified finite-sample safety, monotonic ML, and deterministic safety rules.</p>
</div>
"""
    p2.insert_htmlbox(pymupdf.Rect(43.2, 262.0, 680.0, 395.0), html2_mkt)

    # =========================================================================
    # PAGE 3: PROPOSED SOLUTION
    # =========================================================================
    p3 = doc[2]
    p3.add_redact_annot(pymupdf.Rect(40.0, 110.0, 690.0, 205.0), fill=WHITE)
    p3.add_redact_annot(pymupdf.Rect(35.0, 275.0, 685.0, 375.0), fill=WHITE)
    p3.apply_redactions()

    html3 = """
<div style='font-family: sans-serif; font-size: 8.2pt; color: #1e293b; line-height: 1.35;'>
  <p style='margin: 0 0 4pt 0;'><b>• TriageGuard CDSS:</b> Live, sub-13ms clinical decision support system that classifies ED acuity (ESI Levels 1–5) and routes to 12 specialties.</p>
  <p style='margin: 0 0 4pt 0;'><b>• 5-Tier Swiss-Cheese Hierarchy:</b> 13 Deterministic Vitals Rules &rarr; 10 Multilingual Red Flags &rarr; Mondrian Conformal Screen &rarr; Safety Abstention Layer &rarr; 20x Cost-Optimal Bayes Assignment.</p>
  <p style='margin: 0 0 4pt 0;'><b>• Key Features &amp; Verified Results:</b> 92.8% Emergency Recall (96.4% on vague complaints) on 4,500 test cases; native transliterated Telugu/Hindi NLP; 81.8% anti-gaming defense.</p>
  <p style='margin: 0;'><b>• Unique Safety Philosophy:</b> Unconstrained ML never holds a solitary life-or-death vote. Fails safely by abstaining 17% of corrupted/uncertain cases to senior triage nurses.</p>
</div>
"""
    p3.insert_htmlbox(pymupdf.Rect(43.2, 110.0, 680.0, 205.0), html3)

    # Insert real clinical console screenshot from results/
    img3_path = "results/ui_screens/case1_true_emergency_forced_by_rule_1440.png"
    img3_rect = pymupdf.Rect(110.0, 212.0, 610.0, 395.0)
    p3.draw_rect(img3_rect, color=(0.85, 0.90, 0.95), width=0.8)
    p3.insert_image(img3_rect, filename=img3_path, keep_proportion=True)

    # =========================================================================
    # PAGE 4: DESIGN STEPS & ARCHITECTURE
    # =========================================================================
    p4 = doc[3]
    p4.add_redact_annot(pymupdf.Rect(40.0, 110.0, 360.0, 395.0), fill=WHITE)
    p4.add_redact_annot(pymupdf.Rect(375.0, 105.0, 690.0, 395.0), fill=WHITE)
    p4.apply_redactions()

    html4 = """
<div style='font-family: sans-serif; font-size: 7.8pt; color: #1e293b; line-height: 1.35;'>
  <p style='margin: 0 0 7pt 0;'><b>1. Multimodal Intake &amp; Sensor Validation:</b><br/>Continuous vitals (BP, SpO2, HR, Temp, RR) + text. Plausibility filters detect sensor faults; &ge;2 invalid vitals trigger nurse remeasurement.</p>
  <p style='margin: 0 0 7pt 0;'><b>2. Multilingual NLP &amp; Acuity Floor:</b><br/>Typo-tolerant Damerau-Levenshtein n-gram matching across English, Telugu, and Hindi. Enforces Level 3 floor without allowing artificial Level 1/2 gaming.</p>
  <p style='margin: 0 0 7pt 0;'><b>3. Monotonic LightGBM &amp; Conformal Screen:</b><br/>Monotonic trees enforce physiological directionality (SpO2 &darr;, Shock Index &uarr;). Mondrian screen (&tau;=0.0922, &alpha;=0.05) guarantees finite-sample emergency coverage.</p>
  <p style='margin: 0;'><b>4. 20x Cost Bayes Engine &amp; 12-Dept Router:</b><br/>Minimizes expected risk with a 20x under-triage penalty inside prediction set C(x) and directs patients to 12 specialties at 88.5% accuracy.</p>
</div>
"""
    p4.insert_htmlbox(pymupdf.Rect(43.2, 112.0, 355.0, 395.0), html4)

    # Insert architecture diagram on right
    arch_path = "ppt/architecture_diagram.png"
    arch_rect = pymupdf.Rect(375.0, 110.0, 685.0, 390.0)
    p4.draw_rect(arch_rect, color=(0.85, 0.90, 0.95), width=0.8)
    p4.insert_image(arch_rect, filename=arch_path, keep_proportion=True)

    # =========================================================================
    # PAGE 5: CHALLENGES & MITIGATION
    # =========================================================================
    p5 = doc[4]
    p5.add_redact_annot(pymupdf.Rect(43.2, 102.0, 680.0, 150.0), fill=WHITE)
    p5.add_redact_annot(pymupdf.Rect(43.2, 188.0, 680.0, 238.0), fill=WHITE)
    p5.add_redact_annot(pymupdf.Rect(43.2, 275.0, 680.0, 335.0), fill=WHITE)
    p5.apply_redactions()

    html5_c1 = """
<div style='font-family: sans-serif; font-size: 7.8pt; color: #1e293b; line-height: 1.35;'>
  <p style='margin: 0 0 2pt 0;'><b style='color: #b91c1c;'>Problem:</b> Patients inject alarming keywords (<i>"unbearable dying pain please see me now"</i>) in self-service or digital kiosks to jump the waiting queue.</p>
  <p style='margin: 0;'><b style='color: #166534;'>Mitigation:</b> Strict channel decoupling: text red flags <b>can never assign Level 1 or 2 by themselves</b>. In 500 stress-tested gamed cases, <b>81.8% were safely held at non-emergency</b>, and divergent cases escalated to human nurses (only 0.6% shifted to L1/L2).</p>
</div>
"""
    p5.insert_htmlbox(pymupdf.Rect(57.7, 104.0, 680.0, 150.0), html5_c1)

    html5_c2 = """
<div style='font-family: sans-serif; font-size: 7.8pt; color: #1e293b; line-height: 1.35;'>
  <p style='margin: 0 0 2pt 0;'><b style='color: #b91c1c;'>Problem:</b> Real casualty wards suffer from loose pulse oximeters, swapped systolic/diastolic, Fahrenheit typed in Celsius fields, or detached leads.</p>
  <p style='margin: 0;'><b style='color: #166534;'>Mitigation:</b> Deterministic physical plausibility gates. Rather than blindly imputing values, TriageGuard fires <code>needs_remeasure</code> abstention when &ge;2 vitals fail, scaling safely from 17% to 37.1% under 20% noise while preserving 92.2% emergency recall.</p>
</div>
"""
    p5.insert_htmlbox(pymupdf.Rect(57.7, 190.0, 680.0, 238.0), html5_c2)

    html5_c3 = """
<div style='font-family: sans-serif; font-size: 7.8pt; color: #1e293b; line-height: 1.35;'>
  <p style='margin: 0 0 2pt 0;'><b style='color: #b91c1c;'>Problem:</b> Standard ML models optimize aggregate accuracy, silently misclassifying rare emergencies (1–4% prevalence) such as silent heart attacks or atypical hypoxia.</p>
  <p style='margin: 0;'><b style='color: #166534;'>Mitigation:</b> Mondrian Conformal Screen + 20x Asymmetric Cost Matrix. Calibrated on split emergencies (&tau;=0.0922), catching <b>96.4% of vague emergencies</b> and reducing silent misses by <b>54%</b> vs standard Argmax baselines.</p>
</div>
"""
    p5.insert_htmlbox(pymupdf.Rect(57.7, 276.0, 680.0, 335.0), html5_c3)

    # =========================================================================
    # PAGE 6: UNIQUE SELLING PROPOSITION (USP)
    # =========================================================================
    p6 = doc[5]
    # Redact the 3 differentiator placeholders
    p6.add_redact_annot(pymupdf.Rect(50.0, 118.0, 260.0, 255.0), fill=WHITE)
    p6.add_redact_annot(pymupdf.Rect(275.0, 118.0, 485.0, 255.0), fill=WHITE)
    p6.add_redact_annot(pymupdf.Rect(500.0, 118.0, 700.0, 255.0), fill=WHITE)
    # Redact bottom summary bullets
    p6.add_redact_annot(pymupdf.Rect(43.2, 290.0, 680.0, 395.0), fill=WHITE)
    p6.apply_redactions()

    html6_d1 = """
<div style='font-family: sans-serif; font-size: 7.5pt; color: #1e293b; line-height: 1.32; text-align: center;'>
  <p style='margin: 0 0 3pt 0;'><b>Swiss-Cheese Safety Model</b></p>
  <p style='margin: 0; color: #475569;'>13 deterministic physiological vitals rules and 10 clinical red flags wrap the ML core. Rules can ONLY escalate, never downgrade, guaranteeing that critical vitals (SpO2 &lt; 85%, severe hypotension) override any statistical output.</p>
</div>
"""
    p6.insert_htmlbox(pymupdf.Rect(55.0, 120.0, 255.0, 255.0), html6_d1)

    html6_d2 = """
<div style='font-family: sans-serif; font-size: 7.5pt; color: #1e293b; line-height: 1.32; text-align: center;'>
  <p style='margin: 0 0 3pt 0;'><b>Mondrian Conformal Guarantees</b></p>
  <p style='margin: 0; color: #475569;'>Replaces uncalibrated heuristics with rigorous Mondrian Conformal Prediction sets C(x). Statistically guarantees finite-sample emergency recall (&tau;=0.0922, &alpha;=0.05) with 95.2% non-emergency coverage.</p>
</div>
"""
    p6.insert_htmlbox(pymupdf.Rect(278.0, 120.0, 478.0, 255.0), html6_d2)

    html6_d3 = """
<div style='font-family: sans-serif; font-size: 7.5pt; color: #1e293b; line-height: 1.32; text-align: center;'>
  <p style='margin: 0 0 3pt 0;'><b>Honest Clinical Abstention Layer</b></p>
  <p style='margin: 0; color: #475569;'>Clinical ML must know when NOT to guess. TriageGuard safely abstains under sensor corruption, model discordance (&ge;2 levels gap), or high uncertainty, routing 17% safely to senior triage nurses.</p>
</div>
"""
    p6.insert_htmlbox(pymupdf.Rect(502.0, 120.0, 702.0, 255.0), html6_d3)

    html6_ca = """
<div style='font-family: sans-serif; font-size: 8.2pt; color: #1e293b; line-height: 1.35;'>
  <p style='margin: 0 0 5pt 0;'><b>• Why Hospitals Choose TriageGuard:</b> Ultra-fast 12.77 ms inference allows 100% on-premise local execution with zero cloud latency and total patient privacy. Delivers a 67% reduction in expected clinical decision cost compared to static triage rules.</p>
  <p style='margin: 0;'><b>• Competitive Edge vs Existing Alternatives:</b> Native bilingual support (Telugu &amp; Hindi transliterations) eliminates translation delays. Fully auditable with dynamic wait-room re-ranking, 12-department specialty routing, and an active Railway live deployment.</p>
</div>
"""
    p6.insert_htmlbox(pymupdf.Rect(57.7, 295.0, 680.0, 395.0), html6_ca)

    # =========================================================================
    # PAGE 7: DEMO & POC VIDEO
    # =========================================================================
    p7 = doc[6]
    # Redact text inside video box
    p7.add_redact_annot(pymupdf.Rect(110.0, 345.0, 610.0, 375.0), fill=WHITE)
    p7.apply_redactions()

    # Place real video thumbnail or evidence screen inside the video player box
    # The video box is Rect(108.38, 115.88, 612.38, 360.38)
    vid_rect = pymupdf.Rect(110.0, 118.0, 610.0, 335.0)
    vid_thumb_path = "results/ui_screens/evidence_1440.png"
    p7.insert_image(vid_rect, filename=vid_thumb_path, keep_proportion=True)

    # Bottom link bar with clickable hyperlinks!
    html7 = """
<div style='font-family: sans-serif; font-size: 8.0pt; color: #0369a1; text-align: center; line-height: 1.3;'>
  <p style='margin: 0;'>
    <b>YouTube Walkthrough (2m 30s):</b> <a href='https://youtu.be/yqNZq92zks0' style='color:#2563eb;'>https://youtu.be/yqNZq92zks0</a> &nbsp;&nbsp;|&nbsp;&nbsp;
    <b>GitHub Repo:</b> <a href='https://github.com/grsanudeep42-cmd/triageguard' style='color:#2563eb;'>github.com/grsanudeep42-cmd/triageguard</a> &nbsp;&nbsp;|&nbsp;&nbsp;
    <b>Live App:</b> <a href='https://web-production-bb683.up.railway.app' style='color:#2563eb;'>web-production-bb683.up.railway.app</a>
  </p>
</div>
"""
    p7.insert_htmlbox(pymupdf.Rect(43.2, 345.0, 680.0, 395.0), html7)

    # Add real clickable link annotations to the PDF
    # YouTube Link
    yt_link_rect = pymupdf.Rect(100.0, 345.0, 290.0, 375.0)
    p7.insert_link({"kind": pymupdf.LINK_URI, "from": yt_link_rect, "uri": "https://youtu.be/yqNZq92zks0"})
    
    # GitHub Link
    gh_link_rect = pymupdf.Rect(295.0, 345.0, 520.0, 375.0)
    p7.insert_link({"kind": pymupdf.LINK_URI, "from": gh_link_rect, "uri": "https://github.com/grsanudeep42-cmd/triageguard"})

    # Live Demo Link
    live_link_rect = pymupdf.Rect(525.0, 345.0, 680.0, 375.0)
    p7.insert_link({"kind": pymupdf.LINK_URI, "from": live_link_rect, "uri": "https://web-production-bb683.up.railway.app"})

    doc.save(out_pdf)
    print(f"Successfully generated {out_pdf}!")

if __name__ == '__main__':
    build_pdf()
