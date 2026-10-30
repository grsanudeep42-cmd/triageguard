import pymupdf

doc = pymupdf.open('ppt/Round1_Template submission.pdf')
page = doc[3]

WHITE = (1.0, 1.0, 1.0)

# Redact left implementation placeholder
page.add_redact_annot(pymupdf.Rect(40.0, 110.0, 360.0, 395.0), fill=WHITE)
# Redact right diagram box & text
page.add_redact_annot(pymupdf.Rect(375.0, 105.0, 690.0, 395.0), fill=WHITE)
page.apply_redactions()

html = '''
<div style='font-family: sans-serif; font-size: 7.8pt; color: #1e293b; line-height: 1.35;'>
  <p style='margin: 0 0 7pt 0;'><b>1. Multimodal Intake &amp; Sensor Validation:</b><br/>Continuous vitals (BP, SpO2, HR, Temp, RR) + text. Plausibility filters detect sensor faults; &ge;2 invalid vitals trigger nurse remeasurement.</p>
  <p style='margin: 0 0 7pt 0;'><b>2. Multilingual NLP &amp; Acuity Floor:</b><br/>Typo-tolerant Damerau-Levenshtein n-gram matching across English, Telugu, and Hindi. Enforces Level 3 floor without allowing artificial Level 1/2 gaming.</p>
  <p style='margin: 0 0 7pt 0;'><b>3. Monotonic LightGBM &amp; Conformal Screen:</b><br/>Monotonic trees enforce physiological directionality (SpO2 &darr;, Shock Index &uarr;). Mondrian screen (&tau;=0.0922, &alpha;=0.05) guarantees finite-sample emergency coverage.</p>
  <p style='margin: 0;'><b>4. 20x Cost Bayes Engine &amp; 12-Dept Router:</b><br/>Minimizes expected risk with a 20x under-triage penalty inside prediction set C(x) and directs patients to 12 specialties at 88.5% accuracy.</p>
</div>
'''
page.insert_htmlbox(pymupdf.Rect(43.2, 112.0, 355.0, 395.0), html)

# Insert real architecture diagram on right
arch_path = 'ppt/architecture_diagram.png'
arch_rect = pymupdf.Rect(375.0, 110.0, 685.0, 390.0)
page.draw_rect(arch_rect, color=(0.85, 0.90, 0.95), width=0.8)
page.insert_image(arch_rect, filename=arch_path, keep_proportion=True)

doc.save('ppt/test_p4.pdf')
