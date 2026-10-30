import pymupdf

doc = pymupdf.open('ppt/Round1_Template submission.pdf')
page = doc[2]

WHITE = (1.0, 1.0, 1.0)
GRAY = (0.961, 0.961, 0.961)

# Redact placeholder text in Solution Overview
page.add_redact_annot(pymupdf.Rect(40.0, 110.0, 690.0, 205.0), fill=WHITE)
# Redact the text inside the gray box
page.add_redact_annot(pymupdf.Rect(180.0, 310.0, 540.0, 340.0), fill=GRAY)
page.apply_redactions()

html = '''
<div style='font-family: sans-serif; font-size: 8.2pt; color: #1e293b; line-height: 1.35;'>
  <p style='margin: 0 0 4pt 0;'><b>• TriageGuard CDSS:</b> End-to-end clinical decision support system classifying ED acuity (ESI Levels 1–5) and routing to 12 specialties in &lt;13ms.</p>
  <p style='margin: 0 0 4pt 0;'><b>• 5-Tier Swiss-Cheese Hierarchy:</b> 13 Deterministic Vitals Rules &rarr; 10 Multilingual Red Flags &rarr; Mondrian Conformal Screen &rarr; Safety Abstention Layer &rarr; 20x Cost-Optimal Bayes Assignment.</p>
  <p style='margin: 0 0 4pt 0;'><b>• Key Features &amp; Verified Results:</b> 92.8% Emergency Recall (96.4% on vague complaints) on 4,500 test cases; native transliterated Telugu/Hindi NLP; 81.8% anti-gaming defense.</p>
  <p style='margin: 0;'><b>• Unique Safety Philosophy:</b> Unconstrained ML never holds a solitary life-or-death vote. Fails safely by abstaining 17% of corrupted/uncertain cases to senior triage nurses.</p>
</div>
'''
page.insert_htmlbox(pymupdf.Rect(43.2, 110.0, 680.0, 205.0), html)

# Insert real UI screenshot into image box
img_path = 'results/ui_screens/case1_true_emergency_forced_by_rule_1440.png'
# The template has a gray box at Rect(36.0, 280.5, 684.0, 367.5).
# If we place it from y=210 to y=390:
# Let's see:
img_rect = pymupdf.Rect(95.0, 212.0, 625.0, 395.0)
page.draw_rect(img_rect, color=(0.8, 0.85, 0.9), width=1.0)
page.insert_image(img_rect, filename=img_path, keep_proportion=True)

doc.save('ppt/test_p3.pdf')
