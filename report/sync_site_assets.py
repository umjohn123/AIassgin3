"""Mirror the artefacts that the GitHub Pages site links to into ``docs/``.

GitHub Pages publishes **only** the content of the selected folder (here
``docs/``), so a page that lives in ``docs/index.html`` cannot link to
``../results/figures/...`` or ``../report/...`` -- those paths simply do not
exist on the published site.

This script copies the assets the site references into ``docs/``:

    docs/assets/figures/*.png   <- results/figures/*.png
    docs/assets/report/*.pdf     <- report/AI_Assignment3_Report.pdf
    docs/assets/report/*.docx    <- report/AI_Assignment3_Report.docx
    docs/assets/AI_PROMPTS.md    <- AI_PROMPTS.md

and rewrites the references in ``docs/index.html`` accordingly.  The copies
inside ``docs/assets`` are committed, so the site stays self-contained and
renders identically online and offline.

Usage::

    python report/sync_site_assets.py
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: ``(source, destination-directory)`` pairs, relative to the repository root.
ASSET_COPIES = (
    ("results/figures", "docs/assets/figures"),
    ("report/AI_Assignment3_Report.pdf", "docs/assets/report"),
    ("report/AI_Assignment3_Report.docx", "docs/assets/report"),
    ("AI_PROMPTS.md", "docs/assets"),
)

#: ``(pattern, replacement)`` applied to ``docs/index.html``.
LINK_REWRITES = (
    (r"\.\./results/figures/", "assets/figures/"),
    (r"\.\./report/", "assets/report/"),
    (r"\.\./AI_PROMPTS\.md", "assets/AI_PROMPTS.md"),
)

#: Placeholders in the site that carry the author and the supervisor.  They
#: are filled from ``report/site_config.json`` so that the page, the report
#: title page and the configuration file can never disagree.
AUTHOR_PATTERNS = (
    r"&lt;Student Name / Matric No\.&gt;",
    r"<Student Name / Matric No\.>",
)


def apply_site_metadata(site_path: Path, repo_root: Path) -> bool:
    """Write the author and the supervisor into the site from the config."""
    config_path = repo_root / "report" / "site_config.json"
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False

    author = str(config.get("author") or "").strip()
    supervisor = str(config.get("supervisor") or "").strip()
    if not author:
        return False

    html = site_path.read_text(encoding="utf-8")
    original = html
    for pattern in AUTHOR_PATTERNS:
        html = re.sub(pattern, author, html)
    if supervisor and not supervisor.startswith("<"):
        html = re.sub(r"Supervisor: &lt;[^&]*&gt;", f"Supervisor: {supervisor}", html)
    if html == original:
        return False
    site_path.write_text(html, encoding="utf-8")
    return True


def copy_assets(repo_root: Path) -> List[str]:
    """Copy every asset the site needs into ``docs/assets``."""
    copied: List[str] = []
    for source, destination in ASSET_COPIES:
        src = repo_root / source
        dst = repo_root / destination
        if src.is_dir():
            if not src.exists():
                print(f"  ! missing directory {src}")
                continue
            dst.mkdir(parents=True, exist_ok=True)
            for item in sorted(src.glob("*.png")):
                target = dst / item.name
                shutil.copy2(item, target)
                copied.append(str(target.relative_to(repo_root)))
        elif src.exists():
            dst.mkdir(parents=True, exist_ok=True)
            target = dst / src.name
            shutil.copy2(src, target)
            copied.append(str(target.relative_to(repo_root)))
        else:
            print(f"  ! missing file {src}")
    return copied


def rewrite_links(site_path: Path) -> bool:
    """Rewrite the ``../`` links of the site so that they resolve on Pages."""
    html = site_path.read_text(encoding="utf-8")
    original = html
    for pattern, replacement in LINK_REWRITES:
        html = re.sub(pattern, replacement, html)
    if html == original:
        return False
    site_path.write_text(html, encoding="utf-8")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Mirror the report figures and documents into docs/ for GitHub Pages.",
    )
    parser.add_argument("--repo-root", default=str(_REPO_ROOT))
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root)
    print(f"Mirroring the site assets into {repo_root / 'docs' / 'assets'}")
    copied = copy_assets(repo_root)
    for path in copied:
        print(f"  {path}")

    site = repo_root / "docs" / "index.html"
    if rewrite_links(site):
        print(f"  rewrote the links in {site}")
    else:
        print("  the links in the site were already up to date")
    if apply_site_metadata(site, repo_root):
        print(f"  wrote the author and the supervisor into {site}")
    else:
        print("  the author and the supervisor in the site were already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
