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

The parser is deliberately small rather than general: it has to render five
equations faithfully, and a general LaTeX engine would be a far larger and
much harder to verify dependency than the problem warrants.
"""

from __future__ import annotations

from typing import List, Optional

MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

#: The font Word itself uses for equations.
MATH_FONT = "Cambria Math"

_CTRL_PR = (
    f'<m:ctrlPr><w:rPr><w:rFonts w:ascii="{MATH_FONT}" w:hAnsi="{MATH_FONT}"/>'
    f"<w:i/></w:rPr></m:ctrlPr>"
)

_GREEK = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "varepsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Sigma": "Σ",
    "Delta": "Δ",
    "Omega": "Ω",
    "Phi": "Φ",
}

#: Relations and other named symbols.
_SYMBOLS = {
    "leq": "≤",
    "le": "≤",
    "geq": "≥",
    "ge": "≥",
    "neq": "≠",
    "ne": "≠",
    "approx": "≈",
    "equiv": "≡",
    "propto": "∝",
    "cdot": "⋅",
    "times": "×",
    "div": "÷",
    "pm": "±",
    "mp": "∓",
    "to": "→",
    "rightarrow": "→",
    "leftarrow": "←",
    "Rightarrow": "⇒",
    "in": "∈",
    "notin": "∉",
    "subset": "⊂",
    "forall": "∀",
    "exists": "∃",
    "partial": "∂",
    "nabla": "∇",
    "infty": "∞",
    "ldots": "…",
    "dots": "…",
    "prime": "′",
    "times": "×",
    "lVert": "‖",
    "rVert": "‖",
    "ll": "≪",
    "gg": "≫",
    "ast": "∗",
    "star": "⋆",
    "oplus": "⊕",
    "otimes": "⊗",
    "circ": "∘",
    "bullet": "∙",
    "degree": "°",
}

_ACCENTS = {
    "hat": "̂",  # combining circumflex
    "bar": "̄",  # combining macron
    "tilde": "̃",  # combining tilde
    "dot": "̇",  # combining dot above
}

#: LaTeX spacing commands -> the space that is actually emitted.
#:
#: ``\;`` is a thick space, ``\:`` a medium one, ``\!`` a negative
#: thin space (which pulls the following symbol closer to the preceding
#: one) and the named commands are the usual word-sized spaces.
_SPACES = {
    ",": "\u2009",   # thin space
    " ": "\u0020",   # ordinary space
    ":": "\u2005",   # medium space
    ";": "\u2005",   # thick space
    "!": "",          # negative thin space
    "quad": "\u2003",
    "qquad": "\u2003\u2003",
}

#: BlackBoard-bold characters, used for the expectation symbol.
_BLACKBOARD = {
    "E": "𝔼",
    "R": "ℝ",
    "C": "ℂ",
    "N": "ℕ",
    "Z": "ℤ",
    "Q": "ℚ",
}

#: Characters that Word renders as operators rather than ordinary symbols.
_OPERATORS = set("+-=<>±×÷≤≥≠")


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _run(text: str, upright: bool = False) -> str:
    """A single math run.  ``upright`` switches off the automatic italics."""
    style = '<m:rPr><m:sty m:val="p"/></m:rPr>' if upright else ""
    return (
        f"<m:r>{style}"
        f'<w:rPr><w:rFonts w:ascii="{MATH_FONT}" w:hAnsi="{MATH_FONT}"/></w:rPr>'
        f'<m:t xml:space="preserve">{_escape(text)}</m:t></m:r>'
    )


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
            return _run("")

        if char == "\\":
            return self._command()

        if char == "{":
            self.i += 1
            content = self._sequence()
            self._expect("}")
            return content

        if char in "([{":
            return self._delimiter(char)

        if char.isalpha():
            return _run(self._read_name())

        if char.isdigit():
            start = self.i
            while self._peek().isdigit() or self._peek() == ".":
                self.i += 1
            return _run(self.s[start:self.i])

        self.i += 1
        return _run(char)

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
            if symbol in _SPACES:
                self.i += 1
                return _run(_SPACES[symbol])
            return _run("\\")
        if name in _SPACES:
            return _run(_SPACES[name])

        if name in _GREEK:
            return _run(_GREEK[name])

        if name in _SYMBOLS:
            return _run(_SYMBOLS[name])

        if name == "frac":
            numerator = self._group()
            denominator = self._group()
            return (
                f'<m:f><m:fPr><m:type m:val="bar"/></m:fPr>'
                f"<m:num>{numerator}</m:num><m:den>{denominator}</m:den></m:f>"
            )

        if name == "sum":
            return self._sum()

        if name in _ACCENTS:
            accent = _ACCENTS[name]
            body = self._group()
            return (
                f'<m:acc><m:accPr><m:chr m:val="{accent}"/>{_CTRL_PR}</m:accPr>'
                f"<m:e>{body}</m:e></m:acc>"
            )

        if name == "mathbb":
            body = self._group()
            return self._blackboard(body)

        if name in ("mathrm", "text", "operatorname", "log", "ln", "exp", "sin",
                    "cos", "tan", "max", "min", "arg", "det", "dim", "ker"):
            # Upright text: the braces contain plain prose, not mathematics.
            if self._peek() == "{":
                self.i += 1
                start = self.i
                while self.i < len(self.s) and self.s[self.i] != "}":
                    self.i += 1
                literal = self.s[start:self.i]
                self._expect("}")
            else:
                literal = self._read_name()
            return _run(literal, upright=True)

        if name == "left":
            return self._left_right()

        if name == "right":
            # A stray ``\right`` without a matching ``\left``; skip it.
            return ""

        # An unknown command is shown as the command name itself.
        return _run(name)

    def _blackboard(self, body: str) -> str:
        """Turn a plain letter group into its blackboard-bold form."""
        import re as _re

        letters = _re.findall(r"<m:t[^>]*>(.*?)</m:t>", body, _re.S)
        raw = "".join(letters)
        mapped = "".join(_BLACKBOARD.get(c, c) for c in raw)
        return _run(mapped)

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

        inner = _Parser(content).parse()
        return (
            f"<m:d><m:dPr>"
            f'<m:begChr m:val="{opening}"/><m:endChr m:val="{closing}"/>'
            f"{_CTRL_PR}</m:dPr>"
            f"<m:e>{inner}</m:e></m:d>"
        )

    def _sum(self) -> str:
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

        sup_hidden = "" if upper else '<m:supHide m:val="1"/>'
        return (
            f"<m:nary><m:naryPr>"
            f'<m:chr m:val="∑"/><m:limLoc m:val="undOvr"/>'
            f'<m:subHide m:val="0"/>{sup_hidden}{_CTRL_PR}</m:naryPr>'
            f"<m:sub>{lower}</m:sub><m:sup>{upper}</m:sup>"
            f"<m:e>{body}</m:e></m:nary>"
        )


def render_equation(source: str, centered: bool = True) -> str:
    """Return the ``m:oMathPara`` XML for a display equation."""
    inner = _Parser(source).parse()
    justification = '<m:jc m:val="center"/>' if centered else ""
    return (
        f'<m:oMathPara xmlns:m="{MATH_NS}" xmlns:w="{WORD_NS}">'
        f"<m:oMathParaPr>{justification}</m:oMathParaPr>"
        f"<m:oMath>{inner}</m:oMath>"
        f"</m:oMathPara>"
    )


def render_inline(source: str) -> str:
    """Return the ``m:oMath`` XML for an equation inside a running paragraph."""
    inner = _Parser(source).parse()
    return (
        f'<m:oMath xmlns:m="{MATH_NS}" xmlns:w="{WORD_NS}">'
        f"{inner}</m:oMath>"
    )
