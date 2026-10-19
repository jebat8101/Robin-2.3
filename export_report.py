#!/usr/bin/env python3
"""Export a saved Robin investigation to report formats.

Reads an investigation_*.json from investigations/ and writes two deliverables
into reports/:

  osint-<slug>-<date>.json   structured report (schema robin.investigation.report/1)
  osint-<slug>-<date>.pdf    print-ready investigation report (cover + body)

Usage:
  python3 export_report.py investigations/investigation_20261015_041517.json
  python3 export_report.py <file> [-o reports/]
"""
import argparse
import html
import io
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak, HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"

# ── Palette (auto-generated: pdf skill palette.cascade, cold / minimal) ──
ACCENT = colors.HexColor("#d05d37")
HEADER_FILL = colors.HexColor("#304955")
BORDER = colors.HexColor("#b8c9d2")
CARD_BG = colors.HexColor("#eaeced")
TEXT_PRIMARY = colors.HexColor("#212324")
TEXT_MUTED = colors.HexColor("#787f82")

COVER_PALETTE = {
    "primary": "#304955", "secondary": "#4b5d65",
    "text": "#212324", "muted": "#787f82", "bg": "#ffffff",
}

# ── Local fonts (Liberation = Times-metric serif; DejaVu for mono) ──
_FONTS = [
    ("LibSerif", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"),
    ("LibSerif-Bold", "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"),
    ("LibSerif-Italic", "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"),
    ("LibSerif-BoldItalic", "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf"),
    ("LibSans", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    ("LibSans-Bold", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ("DejaVuMono", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    ("DejaVuMono-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
]
SERIF, SANS, MONO = "LibSerif", "LibSans", "DejaVuMono"

_EMOJI_RE = re.compile(
    "[🀀-🫿🇦-🇿☀-➿⬀-⯿️]+"
)


def register_fonts():
    for name, path in _FONTS:
        pdfmetrics.registerFont(TTFont(name, path))
    registerFontFamily(SERIF, normal=SERIF, bold="LibSerif-Bold",
                       italic="LibSerif-Italic", boldItalic="LibSerif-BoldItalic")
    registerFontFamily(SANS, normal=SANS, bold="LibSans-Bold")
    registerFontFamily(MONO, normal=MONO, bold="DejaVuMono-Bold")


def esc(text):
    """Strip glyphs the local fonts lack, then XML-escape for Paragraph()."""
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return html.escape(text, quote=False)


def rich(text):
    """Escape XML, then allow **bold** markup."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc(text))


def pdf_text(text):
    """Strip emoji (ReportLab renders them as tofu) for PDF output."""
    return _EMOJI_RE.sub("", text).strip()


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def onion_host(link):
    m = re.match(r"https?://([^/]+)/?", link)
    return m.group(1) if m else link


def display_title(text):
    """Strip wrapping quotes the pipeline stores around free-text queries."""
    return text.strip().strip("\"'\u2018\u2019\u201c\u201d").strip()


def parse_sections(summary):
    """Split the summary into its numbered sections with nested bullets."""
    sections, cur = [], None
    for line in summary.splitlines():
        m = re.match(r"^(\d{1,2})\.\s+(.+)$", line)
        if m and not line[:1].isspace():
            body = m.group(2)
            title, inline = (body.split(":", 1) + [""])[:2] if ":" in body else (body, "")
            cur = {"number": int(m.group(1)), "title": title.strip(),
                   "inline": inline.strip(), "lines": []}
            sections.append(cur)
        elif cur is not None:
            cur["lines"].append(line)
    for sec in sections:
        sec["bullets"], sec["content"] = extract_bullets(sec.pop("lines"))
    return sections


def extract_bullets(lines):
    """Convert indented bullet lines to (depth, text); join prose lines."""
    raw = [(len(re.match(r"^\s*", ln).group(0)), ln.strip())
           for ln in lines if ln.strip()]
    bm = re.compile(r"^[*＊•]\s*(.*)$")
    bullet_indents = [ind for ind, t in raw if bm.match(t)]
    base = min(bullet_indents) if bullet_indents else 0
    bullets, prose = [], []
    for ind, text in raw:
        m = bm.match(text)
        if m:
            item = {"depth": max(0, min(3, round((ind - base) / 4))),
                    "text": m.group(1).strip()}
            if item["text"].startswith("http"):
                item["kind"] = "link"
            bullets.append(item)
        elif text.startswith("http"):
            bullets.append({"depth": max(0, round((ind - base) / 4)),
                            "text": text, "kind": "link"})
        else:
            prose.append(text)
    return bullets, " ".join(prose)


def build_report(inv, source_file):
    summary = inv.get("summary") or ""
    sources = inv.get("sources") or []
    sections = parse_sections(summary) if summary else []
    hosts = {onion_host(s.get("link", "")) for s in sources if s.get("link")}
    return {
        "schema": "robin.investigation.report/1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generator": "robin/export_report.py",
        "source_file": source_file,
        "investigation": {
            "timestamp": inv.get("timestamp", ""),
            "query": inv.get("query", ""),
            "refined_query": inv.get("refined_query", ""),
            "model": inv.get("model", ""),
            "preset": inv.get("preset", ""),
        },
        "statistics": {
            "sources_total": len(sources),
            "unique_onion_hosts": len(hosts),
            "summary_words": len(summary.split()),
            "summary_characters": len(summary),
            "sections": len(sections),
        },
        "sections": sections,
        "summary_text": summary,
        "sources": [{"index": i, "title": s.get("title", ""),
                     "link": s.get("link", ""), "host": onion_host(s.get("link", ""))}
                    for i, s in enumerate(sources, 1)],
    }


def report_json_bytes(inv, source_file=""):
    """Structured report JSON as bytes (same schema as the CLI output)."""
    return json.dumps(build_report(inv, source_file), indent=2,
                      ensure_ascii=False).encode("utf-8")


def report_pdf_bytes(inv, source_file=""):
    """Full report PDF (cover + body) as bytes; body-only if the cover engine
    is unavailable on this host."""
    report = build_report(inv, source_file)
    register_fonts()
    body, cover, out = io.BytesIO(), io.BytesIO(), io.BytesIO()
    build_body_pdf(report, body)
    try:
        build_cover_pdf(report, cover)
        merge_cover(cover, body, out, report)
    except Exception as e:  # cover engine unavailable → body-only report
        print("warning: cover skipped (%s)" % e, file=sys.stderr)
        return body.getvalue()
    return out.getvalue()


# ── PDF body ────────────────────────────────────────────────────────
def build_styles():
    return {
        "kicker": ParagraphStyle("kicker", fontName=SANS, fontSize=9.5, leading=13,
                                 textColor=TEXT_MUTED),
        "h1": ParagraphStyle("h1", fontName=SERIF, fontSize=19, leading=23,
                             textColor=TEXT_PRIMARY, spaceBefore=2, spaceAfter=2),
        "h2": ParagraphStyle("h2", fontName=SERIF, fontSize=15, leading=19,
                             textColor=TEXT_PRIMARY, spaceBefore=14, spaceAfter=6),
        "body": ParagraphStyle("body", fontName=SERIF, fontSize=10.5, leading=16.5,
                               textColor=TEXT_PRIMARY, alignment=TA_JUSTIFY,
                               spaceAfter=8),
        "bullet": ParagraphStyle("bullet", fontName=SERIF, fontSize=10.5, leading=16,
                                 textColor=TEXT_PRIMARY, alignment=TA_LEFT),
        "link": ParagraphStyle("link", fontName=MONO, fontSize=8, leading=12.5,
                               textColor=TEXT_PRIMARY),
        "meta_label": ParagraphStyle("meta_label", fontName=SANS, fontSize=9,
                                     leading=13, textColor=TEXT_MUTED),
        "meta_value": ParagraphStyle("meta_value", fontName=SERIF, fontSize=10.5,
                                     leading=14, textColor=TEXT_PRIMARY),
        "stat": ParagraphStyle("stat", fontName=SANS, fontSize=22, leading=24,
                               textColor=ACCENT, alignment=TA_CENTER),
        "stat_label": ParagraphStyle("stat_label", fontName=SANS, fontSize=8.5,
                                     leading=11, textColor=TEXT_MUTED, alignment=TA_CENTER),
        "cell": ParagraphStyle("cell", fontName=SERIF, fontSize=9.5, leading=13,
                               textColor=TEXT_PRIMARY),
        "cell_mono": ParagraphStyle("cell_mono", fontName=MONO, fontSize=7.5,
                                    leading=11, textColor=TEXT_PRIMARY),
        "cell_head": ParagraphStyle("cell_head", fontName=SANS, fontSize=9,
                                    leading=12, textColor=colors.white),
        "caption": ParagraphStyle("caption", fontName=SANS, fontSize=8.5, leading=12,
                                  textColor=TEXT_MUTED, spaceBefore=4, spaceAfter=10),
        "foot": ParagraphStyle("foot", fontName=SANS, fontSize=7.5, leading=10,
                               textColor=TEXT_MUTED),
    }


def stat_row(styles, stats):
    values = [str(stats["sources_total"]), str(stats["unique_onion_hosts"]),
              str(stats["summary_words"]), str(stats["sections"])]
    labels = ["CAPTURED SOURCES", "UNIQUE ONION HOSTS", "SUMMARY WORDS", "REPORT SECTIONS"]
    avail = A4[0] - 1.8 * inch
    gap, n = 10, 4
    colw = (avail - gap * (n - 1)) / n
    data = [[Paragraph(v, styles["stat"]) for v in values],
            [Paragraph(l, styles["stat_label"]) for l in labels]]
    t = Table(data, colWidths=[colw] * n, hAlign="CENTER")
    style = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, 0), 10),
             ("BOTTOMPADDING", (0, 1), (-1, 1), 10)]
    for i in range(n):
        style += [("BACKGROUND", (i, 0), (i, 1), CARD_BG),
                  ("BOX", (i, 0), (i, 1), 0.8, BORDER)]
    t.setStyle(TableStyle(style))
    return t


def meta_table(styles, report):
    inv = report["investigation"]
    run_ts = inv["timestamp"]
    try:
        run_ts = datetime.fromisoformat(run_ts).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        pass
    rows = [("Query", display_title(inv["query"])),
            ("Refined query", display_title(inv["refined_query"])),
            ("Model", inv["model"]), ("Domain preset", pdf_text(inv["preset"])),
            ("Investigation run", run_ts), ("Source file", report["source_file"])]
    data = [[Paragraph(esc(k), styles["meta_label"]), Paragraph(rich(v), styles["meta_value"])]
            for k, v in rows]
    avail = A4[0] - 1.8 * inch
    t = Table(data, colWidths=[130, avail - 130], hAlign="CENTER")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (0, -1), 10),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, BORDER),
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.8, BORDER),
    ]))
    return t


def source_table(styles, sources):
    avail = A4[0] - 1.8 * inch
    colw = [26, 0.36 * (avail - 26), avail - 26 - 0.36 * (avail - 26)]
    head = [Paragraph("<b>#</b>", styles["cell_head"]),
            Paragraph("<b>Title</b>", styles["cell_head"]),
            Paragraph("<b>Onion host</b>", styles["cell_head"])]
    data = [head]
    for s in sources:
        data.append([Paragraph(str(s["index"]), styles["cell"]),
                     Paragraph(rich(s["title"]), styles["cell"]),
                     Paragraph(esc(s["host"]), styles["cell_mono"])])
    t = Table(data, colWidths=colw, hAlign="CENTER", repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), HEADER_FILL),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("LEFTPADDING", (0, 0), (-1, -1), 8),
             ("RIGHTPADDING", (0, 0), (-1, -1), 8),
             ("TOPPADDING", (0, 0), (-1, -1), 5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
             ("GRID", (0, 0), (-1, -1), 0.4, BORDER)]
    for r in range(1, len(data)):
        if r % 2 == 1:
            style.append(("BACKGROUND", (0, r), (-1, r), CARD_BG))
    t.setStyle(TableStyle(style))
    return t


def section_flow(styles, sec):
    avail = A4[0] - 1.8 * inch
    heading = [Paragraph(rich(f"{sec['number']}. {sec['title']}"), styles["h2"]),
               HRFlowable(width="100%", thickness=0.8, color=ACCENT,
                          spaceBefore=0, spaceAfter=6)]
    body = []
    if sec["inline"]:
        body.append(Paragraph(rich(sec["inline"]), styles["body"]))
    if sec["content"]:
        body.append(Paragraph(rich(sec["content"]), styles["body"]))

    def bullet_el(b):
        if b.get("kind") == "link":
            st = ParagraphStyle("lnk", parent=styles["link"], leftIndent=18,
                                bulletIndent=4, bulletFontName=MONO, spaceAfter=4)
            return Paragraph(esc(b["text"]), st, bulletText="\u2022")
        ind = 14 + 16 * b["depth"]
        mark = "\u2022" if b["depth"] == 0 else "\u2013"
        st = ParagraphStyle("b%d" % b["depth"], parent=styles["bullet"],
                            leftIndent=ind, bulletIndent=ind - 10,
                            bulletFontName=SERIF, spaceAfter=3)
        return Paragraph(rich(b["text"]), st, bulletText=mark)

    # Group a bullet with its nested children so headers never orphan at page bottom
    i = 0
    while i < len(sec["bullets"]):
        b = sec["bullets"][i]
        group = [bullet_el(b)]
        j = i + 1
        if b["depth"] == 0:
            while j < len(sec["bullets"]) and sec["bullets"][j]["depth"] > b["depth"]:
                group.append(bullet_el(sec["bullets"][j]))
                j += 1
        if len(group) > 1 and sum(e.wrap(avail, A4[1])[1] for e in group) <= A4[1] * 0.33:
            body.append(KeepTogether(group))
        else:
            body.extend(group)
        i = j

    flow = [CondPageBreak(A4[1] * 0.15)]
    if body:
        first = body[0]
        # keep heading with its first block, not a whole tall section
        if isinstance(first, KeepTogether):
            flat = list(first._content)
            fh = sum(e.wrap(avail, A4[1])[1] for e in flat)
        else:
            flat, fh = [first], first.wrap(avail, A4[1])[1]
        if fh <= A4[1] * 0.33:
            flow.append(KeepTogether(heading + flat))
            flow.extend(body[1:])
            return flow
    return flow + heading + body


def page_footer(canvas, doc, stem):
    canvas.saveState()
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(0.9 * inch, 0.72 * inch, A4[0] - 0.9 * inch, 0.72 * inch)
    canvas.setFont(SANS, 7.5)
    canvas.setFillColor(TEXT_MUTED)
    canvas.drawString(0.9 * inch, 0.56 * inch,
                      "Robin · AI-powered dark web OSINT tool · investigation record")
    canvas.drawRightString(A4[0] - 0.9 * inch, 0.56 * inch,
                           "Source: %s · Page %d" % (stem, doc.page))
    canvas.restoreState()


def build_body_pdf(report, dest):
    styles = build_styles()
    inv, stats = report["investigation"], report["statistics"]
    doc = SimpleDocTemplate(
        dest if hasattr(dest, "write") else str(dest), pagesize=A4,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
        topMargin=0.8 * inch, bottomMargin=0.95 * inch,
        title="Investigation Report — %s" % display_title(inv["query"]),
        author="Robin", creator="Robin export_report.py",
        subject="Dark web OSINT investigation report")
    story = [
        Paragraph("DARK WEB OSINT · INVESTIGATION REPORT", styles["kicker"]),
        Spacer(1, 4),
        Paragraph(rich(display_title(inv["query"]).title()), styles["h1"]),
        HRFlowable(width="100%", thickness=1.4, color=ACCENT, spaceBefore=2, spaceAfter=4),
        Paragraph(esc("Interrogating 16 dark-web engines over Tor · refined query “%s”" %
                      inv["refined_query"]), styles["kicker"]),
        Spacer(1, 14),
        stat_row(styles, stats),
        Spacer(1, 16),
        meta_table(styles, report),
        Spacer(1, 6),
        Paragraph("Machine-refined dark-web sweep filtered down to the sources below; findings are "
                  "the model’s synthesis and every claim stays linked to its captured sources.",
                  styles["caption"]),
        Spacer(1, 4),
    ]
    for sec in report["sections"]:
        story.extend(section_flow(styles, sec))
    if report["sources"]:
        story.append(CondPageBreak(A4[1] * 0.15))
        story.append(Paragraph("Appendix · Source Register", styles["h2"]))
        story.append(HRFlowable(width="100%", thickness=0.8, color=ACCENT,
                                spaceBefore=0, spaceAfter=8))
        story.append(source_table(styles, report["sources"]))
        story.append(Paragraph("All %d sources captured during the investigation run, in retrieval order."
                               % len(report["sources"]), styles["caption"]))

    def on_page(canvas, d):
        page_footer(canvas, d, report["source_file"])
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def build_cover_pdf(report, dest):
    """Render the cover with the pdf skill's cover engine (template 01 HUD)."""
    skill_dir = os.environ.get(
        "PDF_SKILL_DIR",
        os.path.expanduser(
            "~/.zcode/cli/plugins/cache/zcode-plugins-official/pdf/0.1.7/skills/pdf"))
    sys.path.insert(0, os.path.join(skill_dir, "scripts"))
    from cover_render import render_cover  # noqa: E402
    inv, stats = report["investigation"], report["statistics"]
    run = inv["timestamp"][:16].replace("T", " ")
    content = {
        "kicker": "DARK WEB OSINT · INVESTIGATION REPORT",
        "hero": pdf_text(display_title(inv["query"]).title()),
        "summary": ("An LLM-refined dark-web sweep across 16 search engines over Tor, filtered to "
                    "%d captured sources and synthesized into a structured threat-intelligence "
                    "report: leak chatter, darknet market listings, identified threat actors, and "
                    "prioritized next steps." % stats["sources_total"]),
        "meta": "Run %s · Model %s · Domain: %s" %
                (run, inv["model"], pdf_text(inv["preset"])),
    }
    render_cover("01", content, dest if hasattr(dest, "write") else str(dest),
                 palette=COVER_PALETTE,
                 font_overrides={"sans": ("LibSans", _FONTS[4][1], None),
                                 "serif": ("LibSerif", _FONTS[0][1], None),
                                 "latin": ("LibSerif", _FONTS[0][1], None),
                                 "kai": ("LibSerif", _FONTS[0][1], None)})


def merge_cover(cover_pdf, body_pdf, out_path, report):
    from pypdf import PdfReader, PdfWriter

    def reader(src):
        if hasattr(src, "seek"):
            src.seek(0)
            return PdfReader(src)
        return PdfReader(str(src))

    writer = PdfWriter()
    for page in reader(cover_pdf).pages:
        writer.add_page(page)
    for page in reader(body_pdf).pages:
        writer.add_page(page)
    inv = report["investigation"]
    writer.add_metadata({
        "/Title": "Investigation Report — %s" % display_title(inv["query"]),
        "/Author": "Robin",
        "/Subject": "Dark web OSINT investigation · %s · %s" % (inv["model"], inv["preset"]),
        "/Creator": "Robin export_report.py",
    })
    if hasattr(out_path, "write"):
        writer.write(out_path)
    else:
        with open(out_path, "wb") as f:
            writer.write(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export a Robin investigation to JSON + PDF report")
    ap.add_argument("investigation", help="path to investigation_*.json")
    ap.add_argument("-o", "--outdir", default=str(REPORTS_DIR),
                    help="output directory (default: robin/reports/)")
    args = ap.parse_args(argv)

    src = Path(args.investigation)
    inv = json.loads(src.read_text())
    if not inv.get("summary"):
        print("warning: investigation has an empty summary; report will be metadata-only",
              file=sys.stderr)

    report = build_report(inv, src.name)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    slug = slugify(inv.get("query") or "investigation")
    date = (inv.get("timestamp") or "")[:10] or datetime.now().strftime("%Y-%m-%d")
    stem = "osint-%s-%s" % (slug, date)
    json_path = outdir / (stem + ".json")
    pdf_path = outdir / (stem + ".pdf")

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    pdf_path.write_bytes(report_pdf_bytes(inv, src.name))

    print("json: %s" % json_path)
    print("pdf:  %s" % pdf_path)
    print("stats: %d sources · %d unique hosts · %d words · %d sections" %
          (report["statistics"]["sources_total"], report["statistics"]["unique_onion_hosts"],
           report["statistics"]["summary_words"], report["statistics"]["sections"]))


if __name__ == "__main__":
    main()
