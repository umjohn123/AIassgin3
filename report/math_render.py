"""Render a small LaTeX-like subset as native Office Math (OMML).

python-docx cannot typeset mathematics, and writing the five loss equations of
the paper as monospaced ASCII (``L_ort = ( 2 / ( n * (n - 1) ) ) * SUM_{i < j} S_ij^2``)
makes the report read like a transcript rather than a paper.  This module
converts a deliberately small subset of LaTeX into *real* Word equations, so the
result is typeset by Word's own maths engine, can be edited in Word like any
other equation, and exports to PDF with the same quality.

Supported syntax
----------------
``L_{polca}``       subscript on the preceding symbol
``x^{2}``           superscript on the preceding symbol
``\\frac{a}{b}``     fraction with a bar
``\\sum_{i<j}{...}`` summation, limits below and above; the body is braced
``\\hat{z}``         accent - ``\\hat``, ``\\bar``, ``\\tilde``
``\\mathrm{mean}``   upright text (function names)
``\\text{...}``      upright text (prose inside an equation)
``\\mathbb{E}``     blackboard-bold characters (expectation)
``\\left( ... \\right)``  delimiters that grow with their content
``\\alpha`` ...      Greek letters
``\\,`` ``\\quad`` ``\\qquad``  spacing
``( )`` ``[ ]``     delimiters
Everything else is copied literally.

The parser is deliberately small rather than general: it has to render a
handful of equations faithfully, and a general LaTeX engine would be a far
larger and much harder to verify dependency than the problem warrants.

Two details matter more than they look.

*The size of the runs.*  A run without an explicit ``w:sz`` inherits the size
of the surrounding paragraph, which makes an equation the same size as body
text and leaves no room for the machinery of the equation to breathe.  Every
run therefore carries an explicit half-point size.

*Where the limits of a big operator go.*  A summation inside a fraction has
no room beneath it for the limits, so LaTeX puts them as a subscript and a
superscript beside the operator.  The same is true in this renderer: the
limits of a ``\\sum`` move from under-and-over to beside-and-above whenever
the operator is nested inside a fraction.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from math_symbols import (  # noqa: E402
    ACCENTS,
    BLACKBOARD,
    BIG_OPERATORS,
    GREEK,
    SPACES,
    SYMBOLS,
    TEXT_COMMANDS,
)

MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

#: The font Word itself uses for equations.
MATH_FONT = "Cambria Math"

#: The size of a display equation, in points.  Display equations are set
#: slightly larger than the body text so that they read as display material.
DISPLAY_SIZE = 12.5

#: The size of an inline equation, in points.  An inline equation must not
#: stand out from the prose that carries it.
INLINE_SIZE = 12.0

_CTRL_PR = (
    f'<m:ctrlPr><w:rPr><w:rFonts w:ascii="{MATH_FONT}" w:hAnsi="{MATH_FONT}"/>'
    f"<w:i/></w:rPr></m:ctrlPr>"
)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _run(text: str, upright: bool = False, size: float = DISPLAY_SIZE) -> str:
    """A single math run.  ``upright`` switches off the automatic italics."""
    style = '<m:rPr><m:sty m:val="p"/></m:rPr>' if upright else ""
    half = int(round(size * 2))
    return (
        f"<m:r>{style}"
        f'<w:rPr><w:rFonts w:ascii="{MATH_FONT}" w:hAnsi="{MATH_FONT}"/>'
        f'<w:sz w:val="{half}"/><w:szCs w:val="{half}"/></w:rPr>'
        f'<m:t xml:space="preserve">{_escape(text)}</m:t></m:r>'
    )


class _Parser:
    """Recursive-descent parser over the supported LaTeX subset."""

    def __init__(self, source: str, size: float = DISPLAY_SIZE) -> None:
        self.s = source
        self.i = 0
        self.size = size
        #: Nesting depth inside a fraction, where big operators have to put
        #: their limits beside them rather than beneath them.
        self.in_fraction = 0

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
        """Parse a run of terms, stopping at any of the characters in ``stop``.

        A braced group holds ordinary mathematics, so it stops only at ``}``;
        a parenthesised or bracketed group stops at its own closing character
        as well, which is what keeps ``\\left[ ( z_i - mu_i )^{2} \\right]``
        from swallowing the closing bracket.
        """
        out: List[str] = []
        while self.i < len(self.s):
            if self.s[self.i] in stop:
                break
            out.append(self._term())
        return "".join(out)

    def _term(self) -> str:
        """A base expression followed by any number of sub/superscripts."""
        base = self._base()
        while self._peek() in ("_", "^"):
            which = self._peek()
            self.i += 1
            argument = self._group()
            if which == "_":
                base = f"<m:sSub><m:e>{base}</m:e><m:sub>{argument}</m:sub></m:sSub>"
            else:
                base = f"<m:sSup><m:e>{base}</m:e><m:sup>{argument}</m:sup></m:sSup>"
        return base

    def _group(self) -> str:
        """Either a braced group or a single term."""
        if self._peek() == "{":
            self.i += 1
            content = self._sequence("}")
            self._expect("}")
            return content
        return self._term()

    def _base(self) -> str:
        char = self._peek()
        if not char:
            return _run("", size=self.size)

        if char == "\\":
            return self._command()

        if char == "{":
            self.i += 1
            content = self._sequence("}")
            self._expect("}")
            return content

        if char in "([{":
            return self._delimiter(char)

        if char.isalpha():
            return _run(self._read_name(), size=self.size)

        if char.isdigit():
            start = self.i
            while self._peek().isdigit() or self._peek() == ".":
                self.i += 1
            return _run(self.s[start:self.i], size=self.size)

        self.i += 1
        return _run(char, size=self.size)

    def _delimiter(self, opening: str, closing: Optional[str] = None) -> str:
        if closing is None:
            closing = {"(": ")", "[": "]"}[opening]
        self.i += 1
        content = self._sequence(closing)
        self._expect(closing)
        return (
            f"<m:d><m:dPr>"
            f'<m:begChr m:val="{opening}"/><m:endChr m:val="{closing}"/>'
            f"{_CTRL_PR}</m:dPr>"
            f"<m:e>{content}</m:e></m:d>"
        )

    def _command(self) -> str:
        self.i += 1  # consume the backslash
        name = self._read_name()
        if not name:
            # A punctuation-only command such as ``\!``, ``\,`` or ``\;``.
            symbol = self._peek()
            if symbol in SPACES:
                self.i += 1
                return _run(SPACES[symbol], size=self.size)
            return _run("\\", size=self.size)

        if name in SPACES:
            return _run(SPACES[name], size=self.size)

        if name in GREEK:
            return _run(GREEK[name], size=self.size)

        if name in SYMBOLS:
            return _run(SYMBOLS[name], size=self.size)

        if name == "frac":
            # A big operator inside a fraction has no room for limits placed
            # beneath it, so the limits move beside it.  The flag therefore
            # has to be set *before* the two arguments are parsed.
            self.in_fraction += 1
            try:
                numerator = self._group()
                denominator = self._group()
            finally:
                self.in_fraction -= 1
            return (
                f'<m:f><m:fPr><m:type m:val="bar"/>{_CTRL_PR}</m:fPr>'
                f"<m:num>{numerator}</m:num><m:den>{denominator}</m:den></m:f>"
            )

        if name == "sqrt":
            body = self._group()
            return (
                f"<m:rad><m:radPr><m:degHide m:val=\"1\"/>{_CTRL_PR}</m:radPr>"
                f"<m:deg/><m:e>{body}</m:e></m:rad>"
            )

        if name in BIG_OPERATORS:
            return self._big_operator(BIG_OPERATORS[name])

        if name in ACCENTS:
            accent = ACCENTS[name]
            body = self._group()
            return (
                f'<m:acc><m:accPr><m:chr m:val="{accent}"/>{_CTRL_PR}</m:accPr>'
                f"<m:e>{body}</m:e></m:acc>"
            )

        if name == "mathbb":
            body = self._group()
            return self._blackboard(body)

        if name in TEXT_COMMANDS:
            # Upright text: the braces contain plain prose, not mathematics.
            if self._peek() == "{":
                self.i += 1
                start = self.i
                while self.i < len(self.s) and self.s[self.i] != "}":
                    self.i += 1
                literal = self.s[start:self.i]
                self._expect("}")
            else:
                # The command name itself is the text, e.g. ``\log``.
                literal = name
            return _run(literal, upright=True, size=self.size)

        if name == "left":
            return self._left_right()

        if name == "right":
            # A stray ``\right`` without a matching ``\left``; skip it.
            return ""

        # An unknown command is shown as the command name itself.
        return _run(name, size=self.size)

    def _blackboard(self, body: str) -> str:
        """Turn a plain letter group into its blackboard-bold form."""
        letters = re.findall(r"<m:t[^>]*>(.*?)</m:t>", body, re.S)
        raw = "".join(letters)
        mapped = "".join(BLACKBOARD.get(c, c) for c in raw)
        return _run(mapped, size=self.size)

    def _left_right(self) -> str:
        """``\\left( ... \\right)`` - a delimiter that grows with its content."""
        opening = self._peek()
        if opening in "([{":
            self.i += 1
        else:
            opening = "."
        # Find the matching ``\right`` at this nesting level.
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

        inner = _Parser(content, self.size).parse()
        return (
            f"<m:d><m:dPr>"
            f'<m:begChr m:val="{opening}"/><m:endChr m:val="{closing}"/>'
            f"{_CTRL_PR}</m:dPr>"
            f"<m:e>{inner}</m:e></m:d>"
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

        if self.in_fraction:
            # No room beneath the operator: the limits go beside it, as a
            # subscript and a superscript, the way LaTeX sets them inside a
            # fraction.
            return (
                f"<m:sSub><m:e>{_run(symbol, size=self.size)}</m:e>"
                f"<m:sub>{lower}</m:sub></m:sSub>"
            ) if not upper else (
                f"<m:sSubSup><m:e>{_run(symbol, size=self.size)}</m:e>"
                f"<m:sub>{lower}</m:sub><m:sup>{upper}</m:sup></m:sSubSup>"
            )

        sup_hidden = "" if upper else '<m:supHide m:val="1"/>'
        return (
            f"<m:nary><m:naryPr>"
            f'<m:chr m:val="{symbol}"/><m:limLoc m:val="undOvr"/>'
            f'<m:subHide m:val="0"/>{sup_hidden}{_CTRL_PR}</m:naryPr>'
            f"<m:sub>{lower}</m:sub><m:sup>{upper}</m:sup>"
            f"<m:e>{body}</m:e></m:nary>"
        )


def render_equation(source: str, size: float = DISPLAY_SIZE) -> str:
    """Return the ``m:oMath`` XML for a display equation.

    The result is *inline* OMML rather than a block-level ``m:oMathPara``,
    because a block-level equation cannot share a paragraph with the equation
    number that a paper would put at the right margin.  The caller centres
    the equation by placing it in a paragraph with a centred tab stop.
    """
    inner = _Parser(source, size).parse()
    return (
        f'<m:oMath xmlns:m="{MATH_NS}" xmlns:w="{WORD_NS}">'
        f"{inner}</m:oMath>"
    )


def render_inline(source: str, size: float = INLINE_SIZE) -> str:
    """Return the ``m:oMath`` XML for an equation inside a running paragraph."""
    inner = _Parser(source, size).parse()
    return (
        f'<m:oMath xmlns:m="{MATH_NS}" xmlns:w="{WORD_NS}">'
        f"{inner}</m:oMath>"
    )
