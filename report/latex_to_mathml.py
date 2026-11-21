"""Render the report's LaTeX subset as native MathML.

The project site has to publish the same mathematics as the Word report, but
a browser cannot render OMML and the report's LaTeX source is not HTML.
This module converts the LaTeX subset understood by ``report/math_render.py``
into **MathML**, which every current browser renders natively - with real
fraction bars, real summation limits, real radicals and a proper maths font.

Two properties matter as much as the typography:
    10|
*Self-containment.*  MathML is markup, not a script, so the page needs no
JavaScript, no CDN and no web font.  The site keeps rendering offline and
renders identically everywhere, exactly as it did before.

*One source of truth.*  The equations come from ``report/equations.py`` and
the symbols from ``report/math_symbols.py``, the same modules the Word
report reads, so the document and the page cannot disagree about what a
formula is.

    20|The converter is deliberately small, mirroring the LaTeX subset:

``L_{polca}``       subscript on the preceding symbol
``x^{2}``           superscript on the preceding symbol
``\\frac{a}{b}``     fraction with a bar
``\\sum_{i<j}{...}`` summation with limits and a body
``\\hat{z}``         accent - ``\\hat``, ``\\bar``, ``\\tilde``
``\\mathrm{mean}``   upright text (function names)
``\\text{...}``      upright text (prose inside an equation)
``\\mathbb{E}``     blackboard-bold characters
    30|``\\left( ... \\right)``  delimiters that grow with their content
``\\alpha`` ...      Greek letters
``\\,`` ``\\;`` ``\\quad``   spacing

Every element the converter emits carries the **exact number of children
MathML Core requires**.  That is not pedantry: ``<mfrac>`` with three
children instead of two, or ``<munder>`` with nine children instead of two,
is not merely untidy markup - the browser abandons the intended layout and
flattens the limits of a summation into a row of symbols beside the sign.
Every argument is therefore wrapped in an explicit ``<mrow>``.
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

#: The namespace every element of the document belongs to.
MATHML_NS = "http://www.w3.org/1998/Math/MathML"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _row(content: str) -> str:
    """Wrap a piece of markup in an ``<mrow>``.

    MathML Core fixes the arity of its elements: an ``<mfrac>`` takes
    exactly two children, an ``<munder>`` exactly two, an ``<msup>``
    exactly two.  An element with the wrong number of children is not
    rendered as intended, so every multi-part argument is wrapped here
    rather than at each call site.
    """
    if not content:
        return ""
    return f"<mrow>{content}</mrow>"


def _run(content: str, upright: bool = False) -> str:
    """A run of characters, in upright or italic type.

    In mathematics a name made of several letters is a product of
    variables, so each letter becomes its own ``<mi>`` and the browser
    italicises it: the subscript of ``S`` is the product ``i`` ``j``, not
    the word "ij".  Prose is set upright, and reaches the converter through
    ``\\mathrm`` or ``\\text`` rather than through this path.
    """
    if not content:
        return ""
    if upright:
        return f'<mtext mathvariant="normal">{_escape(content)}</mtext>'
    if len(content) == 1:
        return f"<mi>{_escape(content)}</mi>"
    return "".join(f"<mi>{_escape(char)}</mi>" for char in content)


class _Parser:
    """Recursive-descent parser over the supported LaTeX subset."""

    def __init__(self, source: str) -> None:
        self.s = source
        self.i = 0
        #: Whether the next operator is binary or prefix.  LaTeX sets
        #: ``-`` in ``a - b`` with space on both sides but ``-2`` tightly,
        #: and the difference between the two is visible.  The flag is
        #: raised at the start of a group and lowered once a term has been
        #: read, so an operator in the first position of a group is a
        #: prefix operator and everything else is binary.
        self.at_start = True
        #: The nesting depth inside a fraction.  A big operator that sits
        #: in a fraction has no room for its limits beneath it, so they
        #: move beside it.
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
        out: List[str] = []
        while self.i < len(self.s):
            if self.s[self.i] in stop:
                break
            out.append(self._term())
            # A completed term means the next operator is binary.
            self.at_start = False
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
        """The argument of a script, a fraction or an operator.

        The result is always a **single** element, so that ``<msub>`` and
        ``<msup>`` have exactly the two children they require.
        """
        if self._peek() == "{":
            self.i += 1
            was_start = self.at_start
            self.at_start = True
            content = self._sequence("}")
            self.at_start = was_start
            self._expect("}")
            return _row(content)
        return self._term()

    def _base(self) -> str:
        char = self._peek()
        if not char:
            return ""

        if char == "\\":
            return self._command()

        if char == "{":
            self.i += 1
            was_start = self.at_start
            self.at_start = True
            content = self._sequence("}")
            self.at_start = was_start
            self._expect("}")
            return _row(content)

        if char in "([{":
            return self._delimiter(char)

        if char == " ":
            # A literal space in the source is only there for readability;
            # the spacing commands carry the real spacing.
            self.i += 1
            return ""

        if char.isalpha():
            self.at_start = False
            return _run(self._read_name())

        if char.isdigit():
            start = self.i
            while self._peek().isdigit() or self._peek() == ".":
                self.i += 1
            self.at_start = False
            return f"<mn>{_escape(self.s[start:self.i])}</mn>"

        self.i += 1
        # A sign in the first position of a group is a prefix operator: it
        # is set tight against the number that follows it, as in the
        # exponent ``10^{-2}``.  Everywhere else it is a binary operator
        # and is set with space on both sides.
        if char in "+-" and self.at_start:
            self.at_start = False
            return f'<mo lspace="0em" rspace="0.05em">{_escape(char)}</mo>'
        self.at_start = False
        # A comma or a full stop is punctuation: it is followed by a thin
        # space and preceded by none.  The relation and the binary
        # operators get the wider space on both sides.
        if char in ",;":
            return f'<mo lspace="0em" rspace="0.167em">{_escape(char)}</mo>'
        if char == ".":
            return f'<mo lspace="0em" rspace="0.167em">{_escape(char)}</mo>'
        return f'<mo lspace="0.28em" rspace="0.28em">{_escape(char)}</mo>'

    def _delimiter(self, opening: str, closing: Optional[str] = None) -> str:
        if closing is None:
            closing = {"(": ")", "[": "]"}[opening]
        self.i += 1
        content = self._sequence(closing)
        self._expect(closing)
        # The bracketed group is a complete term.
        self.at_start = False
        return (
            f'<mrow><mo stretchy="true" fence="true">{_escape(opening)}</mo>'
            f"{_row(content)}"
            f'<mo stretchy="true" fence="true">{_escape(closing)}</mo></mrow>'
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
            # Lower-case Greek letters are variables and are set in italic;
            # capital Greek letters (Sigma, Delta, Omega, Phi) stand for
            # matrices and sets, and are therefore upright.
            upright = GREEK[name].isupper()
            variant = ' mathvariant="normal"' if upright else ' mathvariant="italic"'
            self.at_start = False
            return f"<mi{variant}>{_escape(GREEK[name])}</mi>"

        if name in SYMBOLS:
            self.at_start = False
            return f'<mo lspace="0.28em" rspace="0.28em">{_escape(SYMBOLS[name])}</mo>'

        if name == "frac":
            # A big operator inside a fraction has no room for limits
            # placed beneath it, so the limits move beside it.  The flag
            # therefore has to be set *before* the two arguments are
            # parsed.
            self.in_fraction += 1
            try:
                numerator = self._group()
                denominator = self._group()
            finally:
                self.in_fraction -= 1
            self.at_start = False
            return f"<mfrac>{numerator}{denominator}</mfrac>"

        if name == "sqrt":
            body = self._group()
            self.at_start = False
            return f"<msqrt>{body}</msqrt>"

        if name in BIG_OPERATORS:
            return self._big_operator(BIG_OPERATORS[name])

        if name in ACCENT_LETTERS:
            body = self._group()
            self.at_start = False
            return (
                f'<mover accent="true">{body}'
                f'<mo stretchy="false">{_escape(ACCENT_LETTERS[name])}</mo></mover>'
            )

        if name == "mathbb":
            body = self._group()
            self.at_start = False
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
            self.at_start = False
            return _run(literal, upright=True)

        if name == "left":
            return self._left_right()

        if name == "right":
            return ""

        # An unknown command is shown as the command name itself.
        self.at_start = False
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
        self.at_start = False
        return (
            f'<mrow><mo stretchy="true" fence="true">{_escape(opening)}</mo>'
            f"{_row(inner)}"
            f'<mo stretchy="true" fence="true">{_escape(closing)}</mo></mrow>'
        )

    def _big_operator(self, symbol: str) -> str:
        """A large operator with optional limits above and below it.

        The limits are stacked, not placed to the side.  MathML decides that
        on its own, but only if the operator is a *large* operator: the
        ``largeop`` attribute is what tells the browser that the sign may
        grow and that its limits belong above and below it.

        There is one exception.  A big operator inside a fraction has no
        room beneath it, so LaTeX moves its limits beside it, as a
        subscript and a superscript.  The same rule is applied here, and
        it is the same rule the Word renderer follows, so the document and
        the page set the fraction the same way.
        """
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
            # No room beneath the operator: the limits go beside it.
            # A script element takes a base and its scripts, and every one
            # of them has to be an element, so the sign becomes an <mo>.
            sign = f"<mo>{_escape(symbol)}</mo>"
            if lower and upper:
                head = f"<msubsup>{sign}{lower}{upper}</msubsup>"
            elif lower:
                head = f"<msub>{sign}{lower}</msub>"
            elif upper:
                head = f"<msup>{sign}{upper}</msup>"
            else:
                head = sign
            self.at_start = False
            return f"{head}{body}"

        op = (
            f'<mo stretchy="true" largeop="true" movablelimits="false">'
            f"{_escape(symbol)}</mo>"
        )
        if lower and upper:
            head = f"<munderover>{op}{lower}{upper}</munderover>"
        elif lower:
            head = f"<munder>{op}{lower}</munder>"
        elif upper:
            head = f"<mover>{op}{upper}</mover>"
        else:
            head = op
        self.at_start = False
        return f"{head}{body}"


def render_mathml(source: str) -> str:
    """Return the MathML for a LaTeX expression, set as a display equation."""
    inner = _Parser(source).parse()
    return f'<math xmlns="{MATHML_NS}" display="block">{inner}</math>'


def render_inline_mathml(source: str) -> str:
    """Return the MathML for an expression inside a running paragraph."""
    inner = _Parser(source).parse()
    return f'<math xmlns="{MATHML_NS}">{inner}</math>'
