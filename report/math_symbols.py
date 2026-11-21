"""The symbol tables shared by the two mathematics renderers.

The Word report typesets its equations with Word's own maths engine
(``report/math_render.py``, which produces OMML) and the project site
publishes the same equations as native MathML
(``report/latex_to_mathml.py``).  Both parsers accept the same LaTeX subset,
so they must agree on what every symbol means.

Keeping the tables here rather than in each parser is what makes that
agreement durable: a symbol can only be added or corrected in one place, and
the document and the page cannot drift apart.
"""

from __future__ import annotations

#: The Greek letters.
GREEK = {
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
SYMBOLS = {
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

#: The accents, as combining marks.  Word places the mark directly above the
#: base character, so a combining mark is the right character here.
ACCENTS = {
    "hat": "̂",  # combining circumflex
    "bar": "̄",  # combining macron
    "tilde": "̃",  # combining tilde
    "dot": "̇",  # combining dot above
    "check": "̌",  # combining caron
    "breve": "̆",  # combining breve
}

#: The same accents as standalone modifier letters.  MathML draws the accent
#: itself and therefore needs a character that stands on its own rather than a
#: combining mark that would be drawn twice.
ACCENT_LETTERS = {
    "hat": "ˆ",
    "bar": "¯",
    "tilde": "˜",
    "dot": "˙",
    "check": "ˇ",
    "breve": "˘",
}

#: LaTeX spacing commands -> the space that is actually emitted.
#:
#: ``\,`` is a thin space, ``\:`` a medium one, ``\!`` a negative thin space
#: (which pulls the following symbol closer to the preceding one) and the
#: named commands are the usual word-sized spaces.
SPACES = {
    ",": " ",
    " ": " ",
    ":": " ",
    ";": " ",
    "!": "",
    "quad": " ",
    "qquad": "  ",
}

#: The same spacing commands, as the widths MathML wants them in.
SPACE_WIDTHS = {
    ",": "0.167em",
    " ": "0.25em",
    ":": "0.222em",
    ";": "0.278em",
    "!": "",
    "quad": "1em",
    "qquad": "2em",
}

#: BlackBoard-bold characters, used for the expectation symbol.
BLACKBOARD = {
    "E": "𝔼",
    "R": "ℝ",
    "C": "ℂ",
    "N": "ℕ",
    "Z": "ℤ",
    "Q": "ℚ",
}

#: Commands whose argument is upright text rather than mathematics.
TEXT_COMMANDS = (
    "mathrm", "text", "operatorname", "log", "ln", "exp", "sin", "cos",
    "tan", "max", "min", "arg", "det", "dim", "ker",
)

#: The operators that take their limits under and over themselves.
BIG_OPERATORS = {
    "sum": "∑",
    "prod": "∏",
    "int": "∫",
}
