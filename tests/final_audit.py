"""Final delivery audit: check every artefact the plan promises.

Run it from the repository root::

    python tests/final_audit.py

It verifies the deliverable checklist of the assignment plan, the shape of the
experiment matrix, the structure of the generated report and the self-containment
of the published site.  Every check reads the artefacts from disk; nothing is
re-computed, so the audit is fast and does not need a GPU.
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
ok = True


def check(label, condition, detail=""):
    global ok
    mark = "PASS" if condition else "FAIL"
    if not condition:
        ok = False
    print(f"  [{mark}] {label}{('  ' + detail) if detail else ''}")


print("== Plan deliverable checklist ==")
for rel in [
    "report/AI_Assignment3_Report.docx",
    "report/AI_Assignment3_Report.pdf",
    "docs/index.html",
    "docs/assets/AI_PROMPTS.md",
    "docs/assets/report/AI_Assignment3_Report.pdf",
    "docs/assets/report/AI_Assignment3_Report.docx",
    "AI_PROMPTS.md",
    "README.md",
    "requirements.txt",
    "results/metrics.csv",
    "results/summary.md",
    "results/ablation.csv",
    "experiments/config.json",
    "src/polca/losses.py",
    "src/polca/model.py",
    "src/polca/data.py",
    "src/polca/utils.py",
    "src/baselines.py",
    "src/metrics.py",
    "src/train.py",
    "src/evaluate.py",
    "src/make_figures.py",
    "tests/smoke_test.py",
    "tests/test_pipeline.py",
    "tests/final_audit.py",
    "report/build_report.py",
    "report/docx_to_pdf.py",
    "report/sync_site_assets.py",
    "report/update_site_results.py",
    "report/equations.py",
    "report/math_symbols.py",
    "report/math_render.py",
    "report/latex_to_mathml.py",
    "report/render_site_math.py",
    "merge_results.py",
    "regen_summary.py",
]:
    check(rel, (root / rel).exists())

print("\n== No leftover placeholders ==")
# The assignment only asks for the student's name and matric number.  A
# supervisor placeholder would be an unfilled field in the submitted
# document, so none may survive anywhere.
for rel in [
    "report/report_text.md",
    "docs/index.html",
    "AI_PROMPTS.md",
    "report/site_config.json",
    "report/build_report.py",
    "report/sync_site_assets.py",
]:
    text = (root / rel).read_text(encoding="utf-8")
    check(f"no supervisor placeholder in {rel}", "Lecturer Name" not in text)
    check(f"no supervisor key in {rel}", '"supervisor"' not in text)

# An unfilled placeholder in a delivered document is a defect in its own
# right: the reader sees a template, not the finished report.  These are the
# tokens that a half-finished edit tends to leave behind.
PLACEHOLDER_TOKENS = [
    "<owner>",
    "<repo>",
    "<github-username>",
    "<repository-name>",
    "to be confirmed",
    "PLACEHOLDER",
]
for rel in [
    "README.md",
    "docs/README.md",
    "docs/index.html",
    "report/report_text.md",
    "report/site_config.json",
    "LICENSE",
]:
    text = (root / rel).read_text(encoding="utf-8")
    for token in PLACEHOLDER_TOKENS:
        check(f"no {token!r} placeholder in {rel}", token not in text)

print("\n== Figures (plan lists 8) ==")
figs = sorted((root / "results" / "figures").glob("*.png"))
check("8 figures", len(figs) == 8, f"{len(figs)} found")

print("\n== Experiment matrix ==")
rows = list(csv.DictReader(open(root / "results" / "metrics.csv", encoding="utf-8")))
check("180 metric rows", len(rows) == 180, str(len(rows)))
check("3 datasets", len({r['dataset'] for r in rows}) == 3)
check("5 methods", len({r['method'] for r in rows}) == 5)
check("3 seeds", len({r['seed'] for r in rows}) == 3, str(sorted({r['seed'] for r in rows})))
check("4 classifiers", len({r['classifier'] for r in rows}) == 4)
abl = list(csv.DictReader(open(root / "results" / "ablation.csv", encoding="utf-8")))
check("12 ablation rows", len(abl) == 12, str(len(abl)))

print("\n== Report ==")
from docx import Document
doc = Document(str(root / "report" / "AI_Assignment3_Report.docx"))
blob = "\n".join(p.text for p in doc.paragraphs)
for t in doc.tables:
    for r in t.rows:
        blob += "\n" + " | ".join(c.text for c in r.cells)
h1 = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading 1")]
check("6 sections + references", len(h1) == 7, str(len(h1)))
check("10 tables", len(doc.tables) == 10)
check("pages url", "umjohn123.github.io/AIassgin3" in blob)
check("repo url", "github.com/umjohn123/AIassgin3" in blob)
check("summary.md in tree", "summary.md" in blob)
check("sync_site_assets in tree", "sync_site_assets.py" in blob)
pdf = (root / "report" / "AI_Assignment3_Report.pdf").read_bytes()
check("pdf header", pdf[:5] == b"%PDF-")
check("pdf size > 1MB", len(pdf) > 1024 * 1024, f"{len(pdf)//1024} KB")

# A page that carries nothing but the running header and the page number is
# a blank page: it is produced when a page break is emitted as its own empty
# paragraph and the preceding content already filled the sheet.  The report
# must not contain any.
try:
    import fitz  # type: ignore

    _pdf = fitz.open(str(root / "report" / "AI_Assignment3_Report.pdf"))
    _blank = []
    for _pno in range(_pdf.page_count):
        _page = _pdf[_pno]
        _body = "\n".join(
            _line
            for _line in _page.get_text().splitlines()
            if _line.strip()
            and not _line.startswith("CISC3024 AI Assignment")
            and not _line.startswith("Page ")
        ).strip()
        if not _body and not _page.get_images():
            _blank.append(_pno + 1)
    _pdf.close()
    check("pdf has no blank page", not _blank, f"blank pages: {_blank}" if _blank else "")
except ImportError:  # pragma: no cover - PyMuPDF is optional
    print("  [SKIP] blank-page check (PyMuPDF not installed)")

print("\n== Author attribution ==")
import json
config = json.loads((root / "report" / "site_config.json").read_text(encoding="utf-8"))
author = str(config.get("author") or "").strip()
site_html = (root / "docs" / "index.html").read_text(encoding="utf-8")
check("author in site_config.json", bool(author), author)
check(
    "no author placeholder left",
    "Student Name" not in blob and "Student Name" not in site_html,
)
if author:
    check("author on the report title page", author in blob)
    check("author in the site header", author in site_html)
    check(
        "author in the markdown dump",
        author in (root / "report" / "report_text.md").read_text(encoding="utf-8"),
    )

print("\n== Site ==")
html = (root / "docs" / "index.html").read_text(encoding="utf-8")
body = re.search(r"BEGIN RESULTS TABLE(.*?)END RESULTS TABLE", html, re.S).group(1)
check("live table rows", len(re.findall(r"<tr>", body)) == 16)
check("no placeholders", ("&mdash;" not in body) and ("—" not in body))
for rel in sorted(set(re.findall(r'(?:src|href)="(assets/[^"]+)"', html))):
    check(f"asset {rel}", (root / "docs" / rel).exists())
check(
    "prompt log mirror in sync",
    (root / "docs" / "assets" / "AI_PROMPTS.md").read_bytes()
    == (root / "AI_PROMPTS.md").read_bytes(),
)

print("\n== Site mathematics ==")
# The site publishes the same mathematics as the report, as native MathML.
# MathML is markup rather than script, so the page stays self-contained.
sys.path.insert(0, str(root))
from report.equations import INLINE, SITE_EQUATIONS, SITE_METRICS  # noqa: E402
from report.latex_to_mathml import render_inline_mathml, render_mathml  # noqa: E402

display_count = html.count('<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">')
inline_count = html.count('<math xmlns="http://www.w3.org/1998/Math/MathML">')
expected_display = len(SITE_EQUATIONS) + len(SITE_METRICS)
check(
    f"{expected_display} display equations",
    display_count == expected_display,
    f"{display_count} found",
)
check("inline symbols typeset", inline_count >= len(INLINE), f"{inline_count} found")
check("no un-typeset math placeholder", 'data-math="' not in html)
check("no ASCII formula leftovers", "sum<sub>" not in html and "&#770;" not in html)
# Every equation of the shared source has to be present verbatim.
_missing = [
    title for title, latex in SITE_EQUATIONS + SITE_METRICS
    if render_mathml(latex) not in html
]
check("every shared equation is on the page", not _missing, str(_missing))
# And the inline symbols too.
_missing_inline = [
    name for name, latex in INLINE.items()
    if render_inline_mathml(latex) not in html
]
check("every inline symbol is on the page", not _missing_inline, str(_missing_inline))
# The page must not depend on a script or an external stylesheet for its
# mathematics: the whole point of MathML is that the browser draws it.
check("no external maths dependency", "mathjax" not in html.lower() and "katex" not in html.lower())

print("\n== GitHub Pages publishing source ==")
# GitHub Pages publishes exactly one folder.  The project page lives in
# ``docs/``, so that folder must be the publishing source; the check makes
# sure the recorded value and the documentation agree, because a mismatch
# here is what makes the live site fall back to the repository README.
config = json.loads((root / "report" / "site_config.json").read_text(encoding="utf-8"))
check(
    "pages_source is main /docs",
    str(config.get("pages_source", "")).strip() == "main /docs",
    str(config.get("pages_source")),
)
check(".nojekyll in the repository root", (root / ".nojekyll").exists())
check(".nojekyll in docs/", (root / "docs" / ".nojekyll").exists())
root_index = root / "index.html"
check("root redirect page exists", root_index.exists())
if root_index.exists():
    redirect = root_index.read_text(encoding="utf-8")
    check("root redirect points at docs/", 'href="docs/"' in redirect)
# Every file the published site references must exist inside the published
# folder, or the link returns a 404 on the live site.
published = root / "docs"
for rel in sorted(set(re.findall(r'(?:src|href)="(assets/[^"]+)"', html))):
    check(f"published asset {rel}", (published / rel).exists())
check(
    "site has an entry file",
    (published / "index.html").exists(),
)
check(
    "license file present",
    (root / "LICENSE").exists(),
    str(config.get("licence", "")),
)

print("\n== Git ==")
st = subprocess.run(
    ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=str(root)
)
check("working tree clean", st.stdout.strip() == "", st.stdout.strip()[:120])
log = subprocess.run(
    ["git", "log", "--oneline", "-5"], capture_output=True, text=True, cwd=str(root)
)
print(log.stdout)

print("\nRESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
