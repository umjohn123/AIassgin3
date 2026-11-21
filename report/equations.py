"""The equations of the report, in one place.

The report typesets its mathematics with Word's own maths engine
(``report/math_render.py``); the project site publishes the same mathematics
as native MathML (``report/latex_to_mathml.py``).  Both converters read the
LaTeX source from *this* module, so the Word document and the web page cannot
drift apart: there is exactly one definition of every equation in the project.

The LaTeX subset is the one understood by ``report/math_render.py``:

    L_{polca}       subscript on the preceding symbol
    x^{2}           superscript on the preceding symbol
    \\frac{a}{b}     fraction with a bar
    \\sum_{i<j}{...} summation with limits and a body
    \\hat{z}         accent - \\hat, \\bar, \\tilde
    \\mathrm{mean}   upright text (function names)
    \\text{...}      upright text (prose inside an equation)
    \\mathbb{E}     blackboard-bold characters
    \\left( ... \\right)  delimiters
    \\alpha ...      Greek letters
    \\, \\; \\quad   spacing
"""

from __future__ import annotations

from typing import Dict, Tuple

# --------------------------------------------------------------------------- #
# The objective
# --------------------------------------------------------------------------- #

#: The composite objective of POLCA Net.
COMPOSITE = (
    r"L_{\mathrm{polca}} \;=\; L_{\mathrm{rec}} \;+\; L_{\mathrm{class}} "
    r"\;+\; \alpha\, L_{\mathrm{ort}} \;+\; \beta\, L_{\mathrm{com}} "
    r"\;+\; \gamma\, L_{\mathrm{var}}"
)

#: The reconstruction term.
RECONSTRUCTION = (
    r"L_{\mathrm{rec}} \;=\; \mathrm{mean}\!\left( (\hat{x} - x)^{2} \right)"
)

#: The orthogonality term, with the Gram matrix it penalises.
ORTHOGONALITY = (
    r"L_{\mathrm{ort}} \;=\; \frac{2}{n\,(n-1)} "
    r"\sum_{1 \leq i < j \leq n} { S_{ij}^{2} }"
    r"\;,\qquad S = \hat{Z}^{T}\hat{Z}"
)

#: The variance term.
VARIANCE = (
    r"L_{\mathrm{var}} \;=\; \sum_{i=0}^{n-1} "
    r"{ \mathbb{E}\!\left[ ( z_{i} - \mu_{i} )^{2} \right] }"
)

#: The centre-of-mass term, which imposes the variance ordering.
CENTRE_OF_MASS = (
    r"L_{\mathrm{com}} \;=\; \frac{ \sum_{i=0}^{n-1} "
    r"{ i^{1+\varepsilon}\, \mathbb{E}\!\left[ ( z_{i} - \mu_{i} )^{2} \right] } }"
    r"{ n\, L_{\mathrm{var}} }"
    r"\;,\qquad \varepsilon = 0.25"
)

#: The optional supervised term.
CLASSIFICATION = (
    r"L_{\mathrm{class}} \;=\; \mathrm{CrossEntropy}\!\left("
    r"\mathrm{logits}(z),\, y \right)"
)

# --------------------------------------------------------------------------- #
# The metrics
# --------------------------------------------------------------------------- #

#: Root mean squared error and its normalised form.
RMSE = (
    r"\mathrm{RMSE} \;=\; \sqrt{ \frac{1}{M} \sum_{k=1}^{M}"
    r" ( x_{k} - \hat{x}_{k} )^{2} }"
    r"\;,\qquad \mathrm{NRMSE} \;=\; \frac{ \mathrm{RMSE} }"
    r"{ \max(x) - \min(x) }"
)

#: Peak signal-to-noise ratio.
PSNR = (
    r"\mathrm{PSNR} \;=\; 10 \log_{10}\!\left("
    r"\frac{ R^{2} }{ \mathrm{MSE} } \right)"
    r"\;,\qquad R = 1 \text{ for data in the unit interval}"
)

# --------------------------------------------------------------------------- #
# The hyper-parameters
# --------------------------------------------------------------------------- #

#: The hyper-parameters of the run, as a display equation.  The values are the
#: ones in ``experiments/config.json``; the equation is written here so that
#: the page and the report state them the same way.
HYPERPARAMETERS = (
    r"\alpha = 10^{-2}\,,\qquad \beta = 10^{-2}\,,\qquad \gamma = 10^{-7}\,,\qquad"
    r"\varepsilon = 0.25"
    r"\;,\qquad \mathrm{batch} = 64\,,\qquad \mathrm{updates} = 20\,000"
    r"\;,\qquad \mathrm{seed} = 5"
)

# --------------------------------------------------------------------------- #
# What the project site publishes
# --------------------------------------------------------------------------- #

#: The display equations of the "The Algorithm" section of the site, in the
#: order in which they appear, each with the caption that introduces it.
#:
#: The captions are not the equation numbers of the report: the report
#: numbers its equations continuously across six sections, whereas the site
#: groups them by topic, and a number that meant one thing in one place and
#: another elsewhere would be worse than no number at all.
SITE_EQUATIONS: Tuple[Tuple[str, str], ...] = (
    ("Composite objective", COMPOSITE),
    ("Reconstruction term", RECONSTRUCTION),
    ("Orthogonality penalty", ORTHOGONALITY),
    ("Variance regulariser", VARIANCE),
    ("Centre-of-mass term (variance ordering)", CENTRE_OF_MASS),
    ("Optional supervised term", CLASSIFICATION),
    ("Hyper-parameters of the reported run", HYPERPARAMETERS),
)

#: The display equations of the "Experiments" section of the site.
SITE_METRICS: Tuple[Tuple[str, str], ...] = (
    ("Reconstruction error", RMSE),
    ("Peak signal-to-noise ratio", PSNR),
)

#: The inline symbols the site refers to.  The key is the name the HTML uses
#: (``data-math="L_rec"``); the value is the LaTeX that is typeset in its
#: place.  Keeping them here means the site and the report always spell a
#: symbol the same way.
INLINE: Dict[str, str] = {
    "L_polca": r"L_{\mathrm{polca}}",
    "L_rec": r"L_{\mathrm{rec}}",
    "L_class": r"L_{\mathrm{class}}",
    "L_ort": r"L_{\mathrm{ort}}",
    "L_com": r"L_{\mathrm{com}}",
    "L_var": r"L_{\mathrm{var}}",
    "S_ij": r"S_{ij}",
    "Z_hat": r"\hat{Z}",
    "z_hat": r"\hat{z}",
    "mu_i": r"\mu_{i}",
    "E": r"\mathbb{E}",
    "sum_ij": r"\sum_{i<j}",
    "eps": r"\varepsilon",
    "alpha": r"\alpha",
    "beta": r"\beta",
    "gamma": r"\gamma",
}
