import pymupdf

doc = pymupdf.open('ppt/Round1_Template submission.pdf')
page = doc[2]

WHITE = (1.0, 1.0, 1.0)

# Redact placeholder text in Solution Overview
page.add_redact_annot(pymupdf.Rect(40.0, 110.0, 690.0, 205.0), fill=WHITE)
# Redact the entire old gray placeholder box so no weird gray edges show!
page.add_redact_annot(pymupdf.Rect(35.0, 275.0, 685.0, 375.0), fill=WHITE)
page.apply_redactions()

html = '''
<div style='font-family: sans-serif; font-size: 8.2pt; color: #1e293b; line-height: 1.35;'>
  <p style='margin: 0 0 4pt 0;'><b>• TriageGuard CDSS:</b> Live, sub-13ms clinical decision support system that classifies ED acuity (ESI Levels 1–5) and routes to 12 specialties.</p>
  <p style='margin: 0 0 4pt 0;'><b>• 5-Tier Swiss-Cheese Hierarchy:</b> 13 Deterministic Vitals Rules &rarr; 10 Multilingual Red Flags &rarr; Mondrian Conformal Screen &rarr; Safety Abstention Layer &rarr; 20x Cost-Optimal Bayes Assignment.</p>
  <p style='margin: 0 0 4pt 0;'><b>• Key Features &amp; Verified Results:</b> 92.8% Emergency Recall (96.4% on vague complaints) on 4,500 test cases; native transliterated Telugu/Hindi NLP; 81.8% anti-gaming defense.</p>
  <p style='margin: 0;'><b>• Unique Safety Philosophy:</b> Unconstrained ML never holds a solitary life-or-death vote. Fails safely by abstaining 17% of corrupted/uncertain cases to senior triage nurses.</p>
</div>
'''
page.insert_htmlbox(pymupdf.Rect(43.2, 110.0, 680.0, 205.0), html)

# Insert real UI screenshot
img_path = 'results/ui_screens/case1_true_emergency_forced_by_rule_1440.png'
img_rect = pymupdf.Rect(110.0, 212.0, 610.0, 395.0)
page.draw_rect(img_rect, color=(0.85, 0.90, 0.95), width=0.8)
page.insert_image(img_rect, filename=img_path, keep_proportion=True)

doc.save('ppt/test_p3_clean.pdf')
