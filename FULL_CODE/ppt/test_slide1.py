import pymupdf

doc = pymupdf.open('ppt/Round1_Template submission.pdf')
page = doc[0]

# Background blue on slide 1
BG_BLUE = (0.122, 0.278, 0.533)
WHITE = (1.0, 1.0, 1.0)
SKY_BLUE = (0.22, 0.74, 0.97)

# Search placeholders
team_rects = page.search_for("[Enter Team Name]")
leader_rects = page.search_for("[Enter Leader Name]")
track_rects = page.search_for("[Select Track]")

for r in team_rects + leader_rects + track_rects:
    page.add_redact_annot(r, fill=BG_BLUE)
page.apply_redactions()

# Insert new texts
# Team Name
page.insert_text((79.3, 246.0), "Ronin Labs", fontname="hebo", fontsize=15, color=WHITE)

# Team Leader
page.insert_text((403.6, 246.0), "Anudeep Gedela R S", fontname="hebo", fontsize=15, color=WHITE)
# Subtext for members
page.insert_text((403.6, 260.0), "Anga Yashwanth, Dileep Kumar Nagothi, Pavan Deepak Seeramreddi", fontname="helv", fontsize=8.5, color=(0.8, 0.9, 1.0))

# Track
track_text = "H1. OPD Triage Assistant for an Overloaded Government Hospital"
# Use insert_textbox to wrap cleanly
track_rect = pymupdf.Rect(79.3, 295.0, 380.0, 360.0)
page.insert_textbox(track_rect, track_text, fontname="hebo", fontsize=11, color=SKY_BLUE)

# Add project subtitle under AI Frontier Challenge
page.insert_text((105.0, 185.0), "TriageGuard: Clinical Decision Support System for Emergency Triage", fontname="hebo", fontsize=13, color=WHITE)

doc.save("ppt/test_output_p1.pdf")
print("Saved ppt/test_output_p1.pdf")
