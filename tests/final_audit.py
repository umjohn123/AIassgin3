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
    "merge_results.py",
    "regen_summary.py",
]:
    check(rel, (root / rel).exists())

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
