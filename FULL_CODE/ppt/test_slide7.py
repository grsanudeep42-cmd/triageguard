import pymupdf

doc = pymupdf.open('ppt/Round1_Template submission.pdf')
p7 = doc[6]
WHITE = (1.0, 1.0, 1.0)
NAVY = (0.122, 0.278, 0.533)
ACCENT_BLUE = (0.0, 0.643, 0.937)

# Redact the old placeholder text and the original cyan play icon inside the video frame
p7.add_redact_annot(pymupdf.Rect(108.0, 115.0, 613.0, 362.0), fill=WHITE)
p7.apply_redactions()

# Draw clean player border
player_rect = pymupdf.Rect(108.0, 115.0, 612.0, 340.0)
p7.draw_rect(player_rect, color=ACCENT_BLUE, width=1.5)

# Insert real 16:9 screenshot inside the player frame
img_path = "results/ui_screens/waiting_room_1440.png"
img_rect = pymupdf.Rect(109.5, 116.5, 610.5, 338.5)
p7.insert_image(img_rect, filename=img_path, keep_proportion=False)

# Add clickable links below the player box
html7 = """
<div style='font-family: sans-serif; font-size: 8.5pt; color: #1e293b; text-align: center; line-height: 1.4;'>
  <p style='margin: 0;'>
    <b style='color: #0369a1;'>YouTube Walkthrough (2m 30s):</b> <a href='https://youtu.be/yqNZq92zks0' style='color:#2563eb; text-decoration: underline;'>https://youtu.be/yqNZq92zks0</a> &nbsp;&nbsp;|&nbsp;&nbsp;
    <b style='color: #0369a1;'>GitHub Repo:</b> <a href='https://github.com/grsanudeep42-cmd/triageguard' style='color:#2563eb; text-decoration: underline;'>github.com/grsanudeep42-cmd/triageguard</a>
  </p>
  <p style='margin: 2pt 0 0 0;'>
    <b style='color: #0369a1;'>Live Production Deployment:</b> <a href='https://web-production-bb683.up.railway.app' style='color:#2563eb; text-decoration: underline;'>https://web-production-bb683.up.railway.app</a> &nbsp;&nbsp;|&nbsp;&nbsp;
    <b style='color: #0369a1;'>Swagger Docs:</b> <a href='https://web-production-bb683.up.railway.app/docs' style='color:#2563eb; text-decoration: underline;'>/docs</a>
  </p>
</div>
"""
p7.insert_htmlbox(pymupdf.Rect(43.2, 348.0, 680.0, 395.0), html7)

# Clickable link annotations
p7.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(100.0, 348.0, 370.0, 362.0), "uri": "https://youtu.be/yqNZq92zks0"})
p7.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(380.0, 348.0, 620.0, 362.0), "uri": "https://github.com/grsanudeep42-cmd/triageguard"})
p7.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(100.0, 364.0, 480.0, 380.0), "uri": "https://web-production-bb683.up.railway.app"})
p7.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(490.0, 364.0, 620.0, 380.0), "uri": "https://web-production-bb683.up.railway.app/docs"})

doc.save('ppt/test_p7.pdf')
