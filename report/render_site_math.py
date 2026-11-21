"""Typeset the mathematics of the project site as native MathML.

The site publishes the same equations as the Word report.  The report
typesets them with Word's own maths engine; the site typesets them with the
browser's, by embedding **MathML**, which every current browser renders
natively - with real fraction bars, real summation limits, real radicals and
a proper maths font.

Three things make this safe for the project:

*One source of truth.*  The equations come from ``report/equations.py`` and
the symbols from ``report/math_symbols.py`` - the same modules the Word
report reads.  The document and the page therefore cannot drift apart.

*Self-containment.*  MathML is markup, not a script, so the page still needs
no JavaScript, no CDN and no web font.  It renders identically online and
offline, exactly as before.

*Reproducibility.*  The HTML is generated, not hand-edited: the display
equations live between the ``<!-- BEGIN EQUATIONS -->`` /
``<!-- END EQUATIONS -->`` markers, and every inline symbol is written as
``<code data-math="NAME">fallback</code>``.  This script fills both in from
the LaTeX source, so the committed page always shows the real equations and
can always be regenerated.

Usage::

    python report/render_site_math.py
    python report/render_site_math.py --site docs/index.html
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from report.equations import INLINE, SITE_EQUATIONS, SITE_METRICS  # noqa: E402
from report.latex_to_mathml import (  # noqa: E402
    render_inline_mathml,
    render_mathml,
)

#: The marker pair that delimits the display equations of the algorithm
#: section.
EQUATION_MARKERS = ("<!-- BEGIN EQUATIONS -->", "<!-- END EQUATIONS -->")

#: The marker pair that delimits the display equations of the metrics section.
METRIC_MARKERS = ("<!-- BEGIN METRIC EQUATIONS -->", "<!-- END METRIC EQUATIONS -->")

#: The pattern of an inline symbol placeholder.
INLINE_PATTERN = re.compile(
    r'<code\s+data-math="(?P<name>[A-Za-z0-9_]+)"\s*>.*?</code>',
    re.DOTALL,
)


def _equation_block(
    title: str,
    latex: str,
    number: int,
    indent: str = "        ",
) -> str:
    """One display equation, with its number and its caption.

    The number is placed in the right margin by the page's own grid, the
    way a printed paper does it, rather than being tucked into the
    caption.
    """
    math = render_mathml(latex)
    return (
        f'{indent}<figure class="equation">\n'
        f"{indent}  <figcaption>{title}</figcaption>\n"
        f'{indent}  <div class="math-display">\n'
        f"{indent}    {math}\n"
        f"{indent}  </div>\n"
        f'{indent}  <span class="equation-number">({number})</span>\n'
        f"{indent}</figure>"
    )


def build_equations(
    pairs: Tuple[Tuple[str, str], ...],
    first_number: int = 1,
    indent: str = "        ",
) -> str:
    """Build the HTML of a block of display equations.

    The equations are numbered continuously across the whole page, so the
    algorithm block starts at ``first_number`` and the metrics block
    continues where the algorithm block stopped.
    """
    blocks = [
        _equation_block(title, latex, first_number + offset, indent)
        for offset, (title, latex) in enumerate(pairs)
    ]
    return "\n".join(blocks)


def replace_block(html: str, markers: Tuple[str, str], body: str) -> str:
    """Replace the region between two markers."""
    begin, end = markers
    pattern = re.compile(
        re.escape(begin) + r".*?" + re.escape(end),
        re.DOTALL,
    )
    if not pattern.search(html):
        raise SystemExit(
            f"the markers {begin} / {end} were not found in the site; "
            "the site has to be updated before the equations can be injected"
        )
    return pattern.sub(lambda _m: f"{begin}\n{body}\n      {end}", html)


def replace_inline(html: str, symbols: Dict[str, str]) -> str:
    """Replace every inline symbol placeholder by its MathML."""

    def _swap(match: "re.Match[str]") -> str:
        name = match.group("name")
        latex = symbols.get(name)
        if latex is None:
            print(f"  ! unknown inline symbol {name!r} - left as plain text")
            return match.group(0)
        return render_inline_mathml(latex)

    return INLINE_PATTERN.sub(_swap, html)


def render_site(site_path: Path) -> bool:
    """Rewrite the mathematics of the site.  Returns True when it changed."""
    html = site_path.read_text(encoding="utf-8")
    original = html

    html = replace_block(
        html, EQUATION_MARKERS, build_equations(SITE_EQUATIONS)
    )
    # The equations are numbered continuously across the page, so the
    # metrics block continues the numbering of the algorithm block.
    html = replace_block(
        html,
        METRIC_MARKERS,
        build_equations(SITE_METRICS, first_number=len(SITE_EQUATIONS) + 1),
    )
    html = replace_inline(html, INLINE)

    if html == original:
        print("  the mathematics of the site was already up to date")
        return False

    site_path.write_text(html, encoding="utf-8")
    print(f"  typeset {len(SITE_EQUATIONS) + len(SITE_METRICS)} display equations "
          f"and the inline symbols into {site_path}")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Typeset the mathematics of docs/index.html as native MathML.",
    )
    parser.add_argument("--site", default=str(_REPO_ROOT / "docs" / "index.html"))
    args = parser.parse_args(argv)

    render_site(Path(args.site))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
