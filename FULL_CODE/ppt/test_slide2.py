import pymupdf

doc = pymupdf.open('ppt/Round1_Template submission.pdf')
page = doc[1]

WHITE = (1.0, 1.0, 1.0)
NAVY = (0.122, 0.278, 0.533)
DARK_GRAY = (0.15, 0.18, 0.22)
MUTED = (0.35, 0.40, 0.45)

# Redact the two placeholder areas
page.add_redact_annot(pymupdf.Rect(40.0, 110.0, 690.0, 230.0), fill=WHITE)
page.add_redact_annot(pymupdf.Rect(40.0, 260.0, 690.0, 398.0), fill=WHITE)
page.apply_redactions()

# In PyMuPDF, we can write paragraphs or lines with exact styling
def write_bullet_section(page, rect, items, font_size=8.5, line_lead=11.5):
    y = rect.y0
    for title, desc in items:
        # Draw bullet and bold title
        bullet_str = f"•  {title}: "
        page.insert_text((rect.x0 + 8.0, y), bullet_str, fontname="hebo", fontsize=font_size, color=NAVY)
        # We can measure title width
        tl = pymupdf.get_text_length(bullet_str, fontname="hebo", fontsize=font_size)
        
        # Now wrap the rest of the text
        # Or simpler: use insert_textbox for the whole item
        item_text = f"•  {title}: {desc}"
        item_rect = pymupdf.Rect(rect.x0 + 8.0, y - 8.0, rect.x1, y + 32.0)
        # Let's test insert_textbox
        # PyMuPDF insert_textbox supports text wrapping automatically!
        y += 36.0

# Let's test insert_html or styled textbox
doc.save("ppt/test_output_p2.pdf")
print("Saved ppt/test_output_p2.pdf")
