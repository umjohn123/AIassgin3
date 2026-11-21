"""Render the report's LaTeX subset as native MathML.

The project site has to publish the same mathematics as the Word report, but
a browser cannot render OMML and the report's LaTeX source is not HTML.
This module converts the LaTeX subset understood by ``report/math_render.py``
into **MathML**, which every current browser renders natively - with real
fraction bars, real summation limits, real radicals and a proper maths font.

Two properties matter as much as the typography:

*Self-containment.*  MathML is markup, not a script, so the page needs no
JavaScript, no CDN and no web font.  The site keeps rendering offline and
renders identically everywhere, exactly as it did before.

*One source of truth.*  The equations come from ``report/equations.py`` and
the symbols from ``report/math_symbols.py``, the same modules the Word
report reads, so the document and the page cannot disagree about what a
formula is.

The converter is deliberately small, mirroring the LaTeX subset:

``L_{polca}``       subscript on the preceding symbol
``x^{2}``           superscript on the preceding symbol
``\\frac{a}{b}``     fraction with a bar
``\\sum_{i<j}{...}`` summation with limits and a body
``\\hat{z}``         accent - ``\\hat``, ``\\bar``, ``\\tilde``
``\\mathrm{mean}``   upright text (function names)
``\\text{...}``      upright text (prose inside an equation)
``\\mathbb{E}``     blackboard-bold characters
``\\left( ... \\right)``  delimiters that grow with their content
``\\alpha`` ...      Greek letters
``\\,`` ``\\;`` ``\\quad``   spacing
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from math_symbols import (  # noqa: E402
    ACCENT_LETTERS,
    BIG_OPERATORS,
    BLACKBOARD,
    GREEK,
    SPACE_WIDTHS,
    SYMBOLS,
    TEXT_COMMANDS,
)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _run(content: str, upright: bool = False) -> str:
    """A run of characters, in upright or italic type.

    A single character becomes an ``<mi>`` so that the browser italicises it
    as a variable; a longer run becomes an ``<mtext>``, which is set upright.
    Function names are therefore always upright, as they are in the report.
    """
    if not content:
        return ""
    variant = ' mathvariant="normal"' if upright else ""
    if len(content) == 1:
        return f"<mi{variant}>{_escape(content)}</mi>"
    return f"<mtext{variant}>{_escape(content)}</mtext>"


class _Parser:
    """Recursive-descent parser over the supported LaTeX subset."""

    def __init__(self, source: str) -> None:
        self.s = source
        self.i = 0

    # -- helpers ---------------------------------------------------------- #
    def _peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def _expect(self, char: str) -> None:
        if self._peek() == char:
            self.i += 1

    def _read_name(self) -> str:
        start = self.i
        while self.i < len(self.s) and self.s[self.i].isalpha():
            self.i += 1
        return self.s[start:self.i]

    # -- grammar ---------------------------------------------------------- #
    def parse(self) -> str:
        return self._sequence()

    def _sequence(self, stop: str = "}") -> str:
        out: List[str] = []
        while self.i < len(self.s):
            if self.s[self.i] in stop:
                break
            out.append(self._term())
        return "".join(out)

    def _term(self) -> str:
        base = self._base()
        while self._peek() in ("_", "^"):
            which = self._peek()
            self.i += 1
            argument = self._group()
            if which == "_":
                base = f"<msub>{base}{argument}</msub>"
            else:
                base = f"<msup>{base}{argument}</msup>"
        return base

    def _group(self) -> str:
        if self._peek() == "{":
            self.i += 1
            content = self._sequence("}")
            self._expect("}")
            return content
        return self._term()

    def _base(self) -> str:
        char = self._peek()
        if not char:
            return ""

        if char == "\\":
            return self._command()

        if char == "{":
            self.i += 1
            content = self._sequence("}")
            self._expect("}")
            return f"<mrow>{content}</mrow>"

        if char in "([{":
            return self._delimiter(char)

        if char == " ":
            # A literal space in the source is only there for readability;
            # the spacing commands carry the real spacing.
            self.i += 1
            return ""

        if char.isalpha():
            return _run(self._read_name())

        if char.isdigit():
            start = self.i
            while self._peek().isdigit() or self._peek() == ".":
                self.i += 1
            return f"<mn>{_escape(self.s[start:self.i])}</mn>"

        self.i += 1
        return f"<mo>{_escape(char)}</mo>"

    def _delimiter(self, opening: str, closing: Optional[str] = None) -> str:
        if closing is None:
            closing = {"(": ")", "[": "]"}[opening]
        self.i += 1
        content = self._sequence(closing)
        self._expect(closing)
        return (
            f'<mrow><mo stretchy="true">{_escape(opening)}</mo>'
            f"<mrow>{content}</mrow>"
            f'<mo stretchy="true">{_escape(closing)}</mo></mrow>'
        )

    def _command(self) -> str:
        self.i += 1  # consume the backslash
        name = self._read_name()
        if not name:
            symbol = self._peek()
            if symbol in SPACE_WIDTHS:
                self.i += 1
                return self._space(SPACE_WIDTHS[symbol])
            return "<mo>\\</mo>"

        if name in SPACE_WIDTHS:
            return self._space(SPACE_WIDTHS[name])

        if name in GREEK:
            return f"<mi>{_escape(GREEK[name])}</mi>"

        if name in SYMBOLS:
            return f"<mo>{_escape(SYMBOLS[name])}</mo>"

        if name == "frac":
            numerator = self._group()
            denominator = self._group()
            return f"<mfrac>{numerator}{denominator}</mfrac>"

        if name == "sqrt":
            body = self._group()
            return f"<msqrt>{body}</msqrt>"

        if name in BIG_OPERATORS:
            return self._big_operator(BIG_OPERATORS[name])

        if name in ACCENT_LETTERS:
            body = self._group()
            return (
                f'<mover accent="true">{body}'
                f'<mo stretchy="false">{_escape(ACCENT_LETTERS[name])}</mo></mover>'
            )

        if name == "mathbb":
            body = self._group()
            return self._blackboard(body)

        if name in TEXT_COMMANDS:
            if self._peek() == "{":
                self.i += 1
                start = self.i
                while self.i < len(self.s) and self.s[self.i] != "}":
                    self.i += 1
                literal = self.s[start:self.i]
                self._expect("}")
            else:
                literal = name
            return _run(literal, upright=True)

        if name == "left":
            return self._left_right()

        if name == "right":
            return ""

        # An unknown command is shown as the command name itself.
        return _run(name)

    @staticmethod
    def _space(width: str) -> str:
        if not width:
            return ""
        return f'<mspace width="{width}"/>'

    def _blackboard(self, body: str) -> str:
        letters = re.findall(r"<mi[^>]*>(.*?)</mi>", body, re.S)
        raw = "".join(letters)
        mapped = "".join(BLACKBOARD.get(c, c) for c in raw)
        return f"<mi>{_escape(mapped)}</mi>"

    def _left_right(self) -> str:
        opening = self._peek()
        if opening in "([{":
            self.i += 1
        else:
            opening = "."
        depth = 0
        start = self.i
        while self.i < len(self.s):
            if self.s.startswith("\\left", self.i):
                depth += 1
                self.i += 5
                continue
            if self.s.startswith("\\right", self.i):
                if depth == 0:
                    break
                depth -= 1
                self.i += 6
                continue
            if self.s[self.i] == "{":
                depth += 1
            elif self.s[self.i] == "}":
                depth -= 1
            self.i += 1
        content = self.s[start:self.i]
        if self.s.startswith("\\right", self.i):
            self.i += 6
            closing = self._peek()
            if closing in ")]}":
                self.i += 1
            else:
                closing = "."
        else:
            closing = {"(": ")", "[": "]"}.get(opening, ".")

        inner = _Parser(content).parse()
        return (
            f'<mrow><mo stretchy="true">{_escape(opening)}</mo>'
            f"<mrow>{inner}</mrow>"
            f'<mo stretchy="true">{_escape(closing)}</mo></mrow>'
        )

    def _big_operator(self, symbol: str) -> str:
        """A large operator with optional limits above and below it."""
        lower = upper = ""
        while self._peek() in ("_", "^"):
            which = self._peek()
            self.i += 1
            argument = self._group()
            if which == "_":
                lower = argument
            else:
                upper = argument

        if self._peek() == "{":
            body = self._group()
        else:
            body = self._term()

        op = f'<mo stretchy="true" movablelimits="false">{_escape(symbol)}</mo>'
        if lower and upper:
            head = f"<munderover>{op}{lower}{upper}</munderover>"
        elif lower:
            head = f"<munder>{op}{lower}</munder>"
        elif upper:
            head = f"<mover>{op}{upper}</mover>"
        else:
            head = op
        return f"{head}{body}"


def render_mathml(source: str) -> str:
    """Return the MathML for a LaTeX expression, set as a display equation."""
    inner = _Parser(source).parse()
    return (
        '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">'
        f"{inner}</math>"
    )


def render_inline_mathml(source: str) -> str:
    """Return the MathML for an expression inside a running paragraph."""
    inner = _Parser(source).parse()
    return f'<math xmlns="http://www.w3.org/1998/Math/MathML">{inner}</math>'
