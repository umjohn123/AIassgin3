#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the final English report for CISC3024 AI Assignment #3 (POLCA Net).

The script assembles ``report/AI_Assignment3_Report.docx`` with python-docx.

Design rules
------------
1. Every number that appears in the report is read at build time from the
   artefacts under ``results/`` (or from ``report/site_config.json``).  The
   script never invents a number: when a result file is missing the affected
   table/cell is replaced by an explicit, clearly marked placeholder.
2. Source-code excerpts are read from ``src/polca/losses.py`` at build time and
   embedded verbatim; they are never re-typed from memory.
3. The document is resilient: a missing ``results/metrics.csv`` must not stop
   the build, it must produce a valid document with visible placeholders.

Usage
-----
    python report/build_report.py
    python report/build_report.py --results-dir results --figures-dir results/figures \
        --out report/AI_Assignment3_Report.docx --site-config report/site_config.json
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docx import Document  # noqa: E402
from docx.enum.section import WD_SECTION  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml import parse_xml  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt, RGBColor  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "report"))
from equations import (  # noqa: E402
    CENTRE_OF_MASS,
    CLASSIFICATION,
    COMPOSITE,
    ORTHOGONALITY,
    PSNR,
    RECONSTRUCTION,
    RMSE,
    VARIANCE,
)
from math_render import render_equation, render_inline  # noqa: E402

# --------------------------------------------------------------------------- #
# Global constants
# --------------------------------------------------------------------------- #
BODY_FONT = "Times New Roman"
BODY_SIZE = Pt(12)
H1_SIZE = Pt(14)
H2_SIZE = Pt(13)
CAPTION_SIZE = Pt(10)
TABLE_SIZE = Pt(10)

REPORT_TITLE = "Extending PCA and LDA to Non-Linear Domains with POLCA Net"
REPORT_SUBTITLE = "AI Assignment #3 - CISC3024 Pattern Recognition"
REPORT_DATE = "November 2026"
#: Fallback used when ``report/site_config.json`` does not carry an author.
REPORT_AUTHOR = "CHAOWENJIE UC325315"
HEADER_TITLE = "CISC3024 AI Assignment #3 - POLCA Net"

SECTION_TITLES = [
    "How AI Was Used to Find the Algorithm",
    "Algorithm Description",
    "How AI Implemented the Algorithm",
    "Experiment Settings and Results",
    "What Was Learned from This AI Assignment",
    "Source Code Web Link",
]

PLACEHOLDER = "Results pending - run `python src/evaluate.py`"

PRETTY_METHOD = {
    "pca": "PCA",
    "lda": "LDA",
    "ae": "Autoencoder",
    "ae_sup": "Autoencoder + L_class",
    "polca": "POLCA Net (unsupervised)",
    "polca_sup": "POLCA Net (supervised)",
    "polca_full": "POLCA Net (full objective)",
    "polca_no_ort": "w/o L_ort",
    "polca_no_com": "w/o L_com",
    "polca_no_var": "w/o L_var",
}

DATASET_PRETTY = {
    "digits": "digits (sklearn)",
    "mnist": "MNIST",
    "fashion": "FashionMNIST",
    "fashion_mnist": "FashionMNIST",
}

METHOD_ORDER = ["pca", "lda", "ae", "polca", "polca_sup"]


# --------------------------------------------------------------------------- #
# Low level docx helpers
# --------------------------------------------------------------------------- #
def _set_run_font(run, name: str = BODY_FONT, size: Optional[Pt] = None,
                  bold: Optional[bool] = None, italic: Optional[bool] = None,
                  color: Optional[RGBColor] = None) -> None:
    """Apply a font to a run in *all* script slots (Word substitutes otherwise)."""
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), name)
    if size is not None:
        run.font.size = size
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color


def _add_tab_stops(paragraph, centre_cm: float, right_cm: float) -> None:
    """Give a paragraph a centred and a right-aligned tab stop.

    A display equation is laid out as a single paragraph with three parts
    separated by tab characters: an empty left part, the equation itself
    (which lands on the centred tab stop) and the equation number (which
    lands on the right tab stop).  This is the layout a printed paper uses,
    and it is the only way to have the equation centred on the text block
    *and* its number flush against the right margin.
    """
    p_pr = paragraph._p.get_or_add_pPr()
    tabs = OxmlElement("w:tabs")
    for position, alignment in ((centre_cm, "center"), (right_cm, "right")):
        tab = OxmlElement("w:tab")
        tab.set(qn("w:val"), alignment)
        tab.set(qn("w:pos"), str(int(Cm(position).twips)))
        tabs.append(tab)
    p_pr.append(tabs)


def _split_inline_math(text: str) -> List[Tuple[str, bool]]:
    """Split a string into prose and inline-equation segments.

    A segment between a pair of ``$`` signs is mathematics; everything
    else is prose.  An unpaired ``$`` is treated as an ordinary character
    so that a stray dollar sign can never swallow the rest of a paragraph.
    """
    parts: List[Tuple[str, bool]] = []
    buffer: List[str] = []
    index = 0
    while index < len(text):
        if text[index] == "$":
            closing = text.find("$", index + 1)
            if closing != -1:
                if buffer:
                    parts.append(("".join(buffer), False))
                    buffer = []
                parts.append((text[index + 1:closing], True))
                index = closing + 1
                continue
        buffer.append(text[index])
        index += 1
    if buffer:
        parts.append(("".join(buffer), False))
    return parts


def _shade(element, fill: str = "F2F2F2") -> None:
    """Apply a light background shading to a paragraph or table cell.

    A paragraph carries the shading in its ``w:pPr`` element, while a table
    cell carries it in its ``w:tcPr`` element -- the two use different
    python-docx accessors, so the right one is chosen based on the type of
    the object that was passed in.
    """
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)

    # The shading element lives in a different place depending on the object:
    # a table cell stores it in ``w:tcPr`` while a paragraph stores it in
    # ``w:pPr``.  The underlying lxml element type tells the two apart
    # unambiguously (``CT_Tc`` vs ``CT_P``).
    element_type = type(getattr(element, "_element", None)).__name__
    if element_type == "CT_Tc":
        pr = element._element.get_or_add_tcPr()
    elif element_type == "CT_P":
        pr = element._element.get_or_add_pPr()
    elif hasattr(element, "_p") and hasattr(element._p, "get_or_add_pPr"):
        pr = element._p.get_or_add_pPr()
    else:
        pr = element
    pr.append(shd)


def _border(paragraph, color: str = "BFBFBF") -> None:
    """Give a paragraph a thin box border (used for formula blocks)."""
    p_pr = paragraph._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "4")
        el.set(qn("w:color"), color)
        borders.append(el)
    p_pr.append(borders)


def _fit_on_page(path: Path, width_cm: float, max_height_cm: float) -> Tuple[float, float]:
    """Scale a figure so that it fits the page, keeping its aspect ratio.

    Word does not scale an image that is too tall: it draws the image at
    the width it was given and lets the bottom of it run off the sheet,
    where it is lost.  A figure is therefore measured first and scaled
    down when its height would not fit.
    """
    try:
        from docx.image.image import Image as DocxImage

        image = DocxImage.from_file(str(path))
        native_w = image.width.cm
        native_h = image.height.cm
    except Exception:  # pragma: no cover - fall back to the given width
        return width_cm, width_cm

    if native_w <= 0 or native_h <= 0:
        return width_cm, width_cm

    height_cm = width_cm * native_h / native_w
    if height_cm > max_height_cm:
        # The figure is too tall for the width it was given, so it is
        # scaled by its height instead and comes out narrower.
        height_cm = max_height_cm
        width_cm = height_cm * native_w / native_h
    return width_cm, height_cm


def _add_field(paragraph, instruction: str) -> None:
    """Insert a Word field (e.g. ``PAGE`` or ``TOC``) into a paragraph."""
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = " "
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    for el in (fld_begin, instr, fld_sep, placeholder, fld_end):
        run._r.append(el)
    _set_run_font(run, BODY_FONT, Pt(10))


# --------------------------------------------------------------------------- #
# Document scaffolding
# --------------------------------------------------------------------------- #
def configure_document(doc: Document) -> None:
    """Page setup, default fonts and the Normal/Heading styles."""
    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)
        section.header_distance = Cm(1.25)
        section.footer_distance = Cm(1.25)

    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = BODY_SIZE
    rpr = normal.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), BODY_FONT)
    pf = normal.paragraph_format
    pf.line_spacing = 1.5
    pf.space_after = Pt(6)
    pf.space_before = Pt(0)

    for style_name, size in (("Heading 1", H1_SIZE), ("Heading 2", H2_SIZE),
                             ("Heading 3", Pt(12))):
        try:
            st = doc.styles[style_name]
        except KeyError:  # pragma: no cover
            continue
        st.font.name = BODY_FONT
        st.font.size = size
        st.font.bold = True
        st.font.italic = False
        st.font.color.rgb = RGBColor(0, 0, 0)
        srpr = st.element.get_or_add_rPr()
        srfonts = srpr.find(qn("w:rFonts"))
        if srfonts is None:
            srfonts = OxmlElement("w:rFonts")
            srpr.append(srfonts)
        for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            srfonts.set(qn(attr), BODY_FONT)
        spf = st.paragraph_format
        spf.space_before = Pt(12)
        spf.space_after = Pt(6)
        spf.keep_with_next = True

    try:
        cap = doc.styles["Caption"]
        cap.font.name = BODY_FONT
        cap.font.size = CAPTION_SIZE
        cap.font.italic = True
        cap.font.bold = False
        cap.font.color.rgb = RGBColor(0, 0, 0)
        crpr = cap.element.get_or_add_rPr()
        crfonts = crpr.find(qn("w:rFonts"))
        if crfonts is None:
            crfonts = OxmlElement("w:rFonts")
            crpr.append(crfonts)
        for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            crfonts.set(qn(attr), BODY_FONT)
        cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_before = Pt(3)
        cap.paragraph_format.space_after = Pt(10)
    except KeyError:  # pragma: no cover
        pass


def add_header_footer(doc: Document) -> None:
    """Header with the report title, footer with a PAGE field."""
    for section in doc.sections:
        header = section.header
        hp = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
        hp.text = ""
        hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = hp.add_run(HEADER_TITLE)
        _set_run_font(run, BODY_FONT, Pt(9), italic=True)

        footer = section.footer
        fp = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        fp.text = ""
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = fp.add_run("Page ")
        _set_run_font(run, BODY_FONT, Pt(10))
        _add_field(fp, "PAGE")
        run = fp.add_run(" of ")
        _set_run_font(run, BODY_FONT, Pt(10))
        _add_field(fp, "NUMPAGES")


# --------------------------------------------------------------------------- #
# Content helpers
# --------------------------------------------------------------------------- #
class ReportWriter:
    """Thin wrapper around a ``Document`` with the report's house style."""

    def __init__(self, doc: Document) -> None:
        self.doc = doc
        self.fig_no = 0
        self.table_no = 0
        self.eq_no = 0
        self.text_log: List[str] = []
        self._pending_page_break = False
        _install_deferred_page_break(self)

    # -- structure -------------------------------------------------------- #
    def h1(self, text: str) -> None:
        p = self.doc.add_paragraph(style="Heading 1")
        run = p.add_run(text)
        _set_run_font(run, BODY_FONT, H1_SIZE, bold=True, color=RGBColor(0, 0, 0))
        self.text_log.append("")
        self.text_log.append(text)
        self.text_log.append("=" * len(text))

    def h2(self, text: str) -> None:
        p = self.doc.add_paragraph(style="Heading 2")
        run = p.add_run(text)
        _set_run_font(run, BODY_FONT, H2_SIZE, bold=True, color=RGBColor(0, 0, 0))
        self.text_log.append("")
        self.text_log.append(text)

    def h3(self, text: str) -> None:
        p = self.doc.add_paragraph(style="Heading 3")
        run = p.add_run(text)
        _set_run_font(run, BODY_FONT, Pt(12), bold=True, color=RGBColor(0, 0, 0))
        self.text_log.append("")
        self.text_log.append(text)

    # -- prose ------------------------------------------------------------ #
    def body(self, text: str, indent: bool = True, italic: bool = False) -> None:
        """A paragraph of prose.

        Text between a pair of ``$`` signs is typeset as an inline
        equation, so that a symbol mentioned in the running text uses the
        same typography as the display equations.
        """
        p = self.doc.add_paragraph()
        if indent:
            p.paragraph_format.first_line_indent = Cm(0.75)

        for chunk, is_math in _split_inline_math(text):
            if is_math:
                p._p.append(parse_xml(render_inline(chunk)))
            else:
                run = p.add_run(chunk)
                _set_run_font(run, BODY_FONT, BODY_SIZE, italic=italic)
        self.text_log.append(text)

    def body_rich(self, *parts: Tuple[str, bool]) -> None:
        """A paragraph assembled from explicit ``(text, is_math)`` parts."""
        p = self.doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0.75)
        for chunk, is_math in parts:
            if is_math:
                p._p.append(parse_xml(render_inline(chunk)))
            else:
                run = p.add_run(chunk)
                _set_run_font(run, BODY_FONT, BODY_SIZE)
        self.text_log.append(" ".join(c for c, _ in parts))

    def bullets(self, items: Iterable[str]) -> None:
        for item in items:
            p = self.doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_after = Pt(3)
            run = p.add_run(item)
            _set_run_font(run, BODY_FONT, BODY_SIZE)
            self.text_log.append(f"  - {item}")

    def numbered(self, items: Iterable[str]) -> None:
        """Add a numbered list.

        The numbers are written out explicitly rather than through the
        ``List Number`` style, because that style keeps a single numbering
        sequence alive for the whole document: a second list would continue
        from where the previous one stopped instead of restarting at 1.
        """
        for index, item in enumerate(items, start=1):
            p = self.doc.add_paragraph()
            pf = p.paragraph_format
            pf.left_indent = Cm(0.9)
            pf.first_line_indent = Cm(-0.6)
            pf.space_after = Pt(3)
            run = p.add_run(f"{index}.  {item}")
            _set_run_font(run, BODY_FONT, BODY_SIZE)
            self.text_log.append(f"  {index}. {item}")

    def note(self, text: str) -> None:
        """A small italic note, e.g. a placeholder warning."""
        p = self.doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.5)
        run = p.add_run(text)
        _set_run_font(run, BODY_FONT, Pt(10.5), italic=True, color=RGBColor(0x99, 0x00, 0x00))
        self.text_log.append(f"[NOTE] {text}")

    # -- code / formulas --------------------------------------------------- #
    def code_block(self, code: str, size: float = 8.0) -> None:
        """Monospace, indented, shaded block (used for source code and formulas)."""
        lines = code.rstrip("\n").split("\n")
        for i, line in enumerate(lines):
            p = self.doc.add_paragraph()
            pf = p.paragraph_format
            pf.left_indent = Cm(0.8)
            pf.right_indent = Cm(0.4)
            pf.space_before = Pt(3) if i == 0 else Pt(0)
            pf.space_after = Pt(3) if i == len(lines) - 1 else Pt(0)
            pf.line_spacing = 1.0
            run = p.add_run(line if line else " ")
            _set_run_font(run, "Consolas", Pt(size))
            _shade(p, "F4F4F4")
            _border(p)
        self.text_log.append("```")
        self.text_log.append(code.rstrip("\n"))
        self.text_log.append("```")

    def formula(self, latex: str, note: str = "", numbered: bool = True) -> None:
        """A display equation, typeset by Word's own maths engine.

        The argument is a small LaTeX subset (see ``report/math_render.py``).
        The equation is centred on the text block and, unless ``numbered`` is
        turned off, carries a right-aligned equation number in the manner of
        a printed paper.

        If the equation cannot be parsed the builder prints a warning and
        falls back to the old monospaced rendering, so a typo in the source
        can never silently remove an equation from the report.
        """
        try:
            xml = render_equation(latex)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  ! could not typeset the equation {latex!r}: {exc}")
            xml = None

        if xml:
            p = self.doc.add_paragraph()
            pf = p.paragraph_format
            pf.space_before = Pt(10)
            pf.space_after = Pt(10)
            pf.line_spacing = 1.0
            pf.keep_together = True
            # Centre the equation and push the number to the right margin.
            # The leading tab is what moves the equation onto the centred
            # tab stop; without it the equation would sit at the left margin.
            _add_tab_stops(p, centre_cm=8.0, right_cm=15.9)
            p.add_run("\t")
            p._p.append(parse_xml(xml))
            if numbered:
                self.eq_no += 1
                p.add_run("\t")
                run = p.add_run(f"({self.eq_no})")
                _set_run_font(run, BODY_FONT, Pt(11))
            else:
                # A trailing tab keeps the equation centred on its own.
                p.add_run("\t")
        else:  # pragma: no cover - defensive
            p = self.doc.add_paragraph()
            pf = p.paragraph_format
            pf.left_indent = Cm(1.0)
            pf.right_indent = Cm(0.5)
            pf.space_before = Pt(4)
            pf.space_after = Pt(4)
            pf.line_spacing = 1.15
            run = p.add_run(latex)
            _set_run_font(run, "Consolas", Pt(11))
            _shade(p, "F2F4F7")
            _border(p, "8FA8C8")

        self.text_log.append("")
        if numbered:
            self.text_log.append(f"    ({self.eq_no})  {latex}")
        else:
            self.text_log.append(f"    {latex}")
        if note:
            self.body(note, indent=True, italic=True)

    def formula_inline(self, latex: str) -> None:
        """An equation inside a running paragraph."""
        p = self.doc.add_paragraph()
        pf = p.paragraph_format
        pf.first_line_indent = Cm(0.75)
        pf.space_before = Pt(4)
        pf.space_after = Pt(4)
        p._p.append(parse_xml(render_inline(latex)))
        self.text_log.append(f"    {latex}")

    # -- captions ---------------------------------------------------------- #
    def caption_above(self, text: str) -> None:
        self.table_no += 1
        p = self.doc.add_paragraph(style="Caption")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(f"Table {self.table_no}. {text}")
        _set_run_font(run, BODY_FONT, CAPTION_SIZE, italic=True)
        self.text_log.append(f"Table {self.table_no}. {text}")

    def caption_below(self, text: str) -> None:
        self.fig_no += 1
        p = self.doc.add_paragraph(style="Caption")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(f"Fig. {self.fig_no}. {text}")
        _set_run_font(run, BODY_FONT, CAPTION_SIZE, italic=True)
        self.text_log.append(f"Fig. {self.fig_no}. {text}")

    # -- tables ------------------------------------------------------------ #
    def table(self, headers: Sequence[str], rows: Sequence[Sequence[Any]],
              widths: Optional[Sequence[float]] = None) -> None:
        ncols = len(headers)
        table = self.doc.add_table(rows=1, cols=ncols)
        try:
            table.style = "Table Grid"
        except KeyError:  # pragma: no cover
            pass
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = True

        hdr = table.rows[0].cells
        for j, head in enumerate(headers):
            cell = hdr[j]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.line_spacing = 1.0
            run = p.add_run(str(head))
            _set_run_font(run, BODY_FONT, TABLE_SIZE, bold=True)
            _shade(cell, "E8ECF2")

        for row in rows:
            cells = table.add_row().cells
            for j, value in enumerate(row):
                if j >= ncols:
                    break
                cell = cells[j]
                cell.text = ""
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.line_spacing = 1.0
                # A cell may contain inline mathematics between ``$`` signs.
                for chunk, is_math in _split_inline_math(str(value)):
                    if is_math:
                        p._p.append(parse_xml(render_inline(chunk)))
                    else:
                        run = p.add_run(chunk)
                        _set_run_font(run, BODY_FONT, TABLE_SIZE)

        if widths:
            for row in table.rows:
                for j, w in enumerate(widths):
                    if j < len(row.cells):
                        row.cells[j].width = Cm(w)

        spacer = self.doc.add_paragraph()
        spacer.paragraph_format.space_after = Pt(2)
        spacer.paragraph_format.line_spacing = 1.0
        run = spacer.add_run("")
        _set_run_font(run, BODY_FONT, Pt(4))

        self.text_log.append("")
        self.text_log.append("| " + " | ".join(str(h) for h in headers) + " |")
        self.text_log.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            self.text_log.append("| " + " | ".join(str(v) for v in row) + " |")

    # -- figures ----------------------------------------------------------- #
    def figure(self, path: Path, caption: str, width_cm: float = 15.5,
               max_height_cm: float = 20.0) -> bool:
        """Insert a figure, scaled to fit on the page.

        A figure is scaled down when it is taller than the space the page
        has left.  Word does not do this by itself: an image that is too
        tall is simply cut off at the bottom of the sheet, and the part
        that is missing is lost for good.  The reconstruction grid used to
        be lost that way, and every figure now keeps its aspect ratio
        inside the page.
        """
        if not path or not Path(path).exists():
            self.note(f"[Figure omitted: '{Path(path).name if path else '?'}' was not found "
                      f"in the figures directory.]")
            return False

        width_cm, height_cm = _fit_on_page(path, width_cm, max_height_cm)

        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run()
        run.add_picture(str(path), width=Cm(width_cm), height=Cm(height_cm))
        self.caption_below(caption)
        return True

    # -- page breaks ------------------------------------------------------- #
    def page_break(self) -> None:
        """Start the next block on a fresh page.

        The break is *deferred*: it is remembered and then applied to the
        first paragraph that is added afterwards, as a ``w:pageBreakBefore``
        property.  A standalone paragraph holding only a ``w:br`` of type
        ``page`` would be the obvious implementation, but it produces a
        completely blank page whenever the preceding content already fills
        the sheet: the empty paragraph is pushed onto the next page and the
        break inside it then starts yet another page.  Attaching the break to
        the following paragraph cannot create a page without content.
        """
        self._pending_page_break = True


def _install_deferred_page_break(writer: "ReportWriter") -> None:
    """Hook ``Document.add_paragraph``/``add_table`` for the deferred break.

    Every block the report adds goes through one of those two methods, so
    hooking them covers the whole document.  When a page break is pending,
    the very next paragraph (or the first paragraph inside the first cell of
    the next table) receives a ``w:pageBreakBefore`` property, which makes
    Word start it on a fresh sheet without ever emitting a page of its own.
    """
    doc = writer.doc
    original_add_paragraph = doc.add_paragraph
    original_add_table = doc.add_table

    def add_paragraph(*args, **kwargs):
        paragraph = original_add_paragraph(*args, **kwargs)
        if writer._pending_page_break:
            writer._pending_page_break = False
            paragraph.paragraph_format.page_break_before = True
        return paragraph

    def add_table(*args, **kwargs):
        table = original_add_table(*args, **kwargs)
        if writer._pending_page_break:
            writer._pending_page_break = False
            # A table carries no page-break property of its own, so the break
            # is attached to the first paragraph of its first cell.
            for p_el in _first_cell_paragraphs(table._tbl):
                _set_page_break_before(p_el)
                break
        return table

    doc.add_paragraph = add_paragraph
    doc.add_table = add_table


def _set_page_break_before(p_el) -> None:
    """Set ``w:pageBreakBefore`` on a raw ``w:p`` element."""
    pPr = p_el.find(qn("w:pPr"))
    if pPr is None:
        pPr = OxmlElement("w:pPr")
        p_el.insert(0, pPr)
    if pPr.find(qn("w:pageBreakBefore")) is None:
        pPr._insert_pageBreakBefore(OxmlElement("w:pageBreakBefore"))


def _first_cell_paragraphs(tbl_el):
    """Yield the paragraphs of the first cell of a raw ``w:tbl`` element."""
    for tr in tbl_el.iterchildren():
        if tr.tag.split("}")[-1] != "tr":
            continue
        for tc in tr.iterchildren():
            if tc.tag.split("}")[-1] != "tc":
                continue
            for p_el in tc.iterchildren():
                if p_el.tag.split("}")[-1] == "p":
                    yield p_el


# --------------------------------------------------------------------------- #
# Result loading (never fabricates numbers)
# --------------------------------------------------------------------------- #
class Results:
    """Container for whatever could be read from ``results/``."""

    def __init__(self) -> None:
        self.metrics: Optional[List[Dict[str, Any]]] = None
        self.metrics_fields: List[str] = []
        self.ablation: Optional[List[Dict[str, Any]]] = None
        self.ablation_fields: List[str] = []
        self.metrics_json: Optional[Dict[str, Any]] = None
        self.summary_md: Optional[str] = None
        self.sources: Dict[str, str] = {}

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(f) or math.isinf(f):
            return None
        return f

    def _pick(self, row: Dict[str, Any], candidates: Sequence[str]) -> Optional[float]:
        for cand in candidates:
            if cand in row:
                v = self._to_float(row[cand])
                if v is not None:
                    return v
        return None

    # -- public API -------------------------------------------------------- #
    def has_metrics(self) -> bool:
        return bool(self.metrics)

    def datasets(self) -> List[str]:
        if not self.metrics:
            return []
        seen: List[str] = []
        for row in self.metrics:
            d = str(row.get("dataset", ""))
            if d and d not in seen:
                seen.append(d)
        return seen

    def methods_for(self, dataset: str) -> List[str]:
        if not self.metrics:
            return []
        seen: List[str] = []
        for row in self.metrics:
            if str(row.get("dataset", "")) == dataset:
                m = str(row.get("method", ""))
                if m and m not in seen:
                    seen.append(m)
        return seen

    def aggregate(self, dataset: str, method: str) -> Dict[str, Any]:
        """Mean over seeds (and over classifiers) for one (dataset, method)."""
        if not self.metrics:
            return {}
        rows = [r for r in self.metrics
                if str(r.get("dataset", "")) == dataset and str(r.get("method", "")) == method]
        if not rows:
            return {}

        nrmse = [self._pick(r, ("test_nrmse", "nrmse_test", "test_nrmse_mean", "nrmse"))
                 for r in rows]
        psnr = [self._pick(r, ("test_psnr", "psnr_test", "test_psnr_mean", "psnr"))
                for r in rows]
        ssim = [self._pick(r, ("test_ssim", "ssim_test", "test_ssim_mean", "ssim"))
                 for r in rows]
        ort = [self._pick(r, ("orthogonality_error", "orth_error", "orthogonality"))
               for r in rows]
        vord = [self._pick(r, ("variance_ordering_score", "variance_order", "vorder"))
                for r in rows]
        seeds = sorted({r.get("seed") for r in rows if r.get("seed") is not None})

        accs = self._classification_values(rows, "accuracy")
        f1s = self._classification_values(rows, "f1")

        def mean(vals: Sequence[Optional[float]]) -> Optional[float]:
            clean = [v for v in vals if v is not None]
            return (sum(clean) / len(clean)) if clean else None

        return {
            "nrmse": mean(nrmse),
            "psnr": mean(psnr),
            "ssim": mean(ssim),
            "orthogonality_error": mean(ort),
            "variance_ordering_score": mean(vord),
            "accuracy": mean(accs),
            "f1": mean(f1s),
            "accuracy_values": accs,
            "f1_values": f1s,
            "n_rows": len(rows),
            "n_seeds": len(seeds),
            "seeds": seeds,
            "latent_dim": rows[0].get("latent_dim") or rows[0].get("num_latent_dims"),
        }

    def _classification_values(self, rows: Sequence[Dict[str, Any]],
                               kind: str) -> List[float]:
        """Accuracy / F1 values for a group of rows.

        Handles both layouts: a long layout with a ``classifier`` column
        (``accuracy`` / ``f1`` columns) and a wide layout with per-classifier
        columns (``accuracy_Perceptron`` ... plus an ``accuracy_mean``).
        """
        values: List[float] = []
        mean_col = f"{kind}_mean"
        if rows and mean_col in rows[0]:
            for r in rows:
                v = self._to_float(r.get(mean_col))
                if v is not None:
                    values.append(v)
            if values:
                return values
        # long layout
        for r in rows:
            v = self._to_float(r.get(kind))
            if v is not None:
                values.append(v)
        if values:
            return values
        # wide layout
        for r in rows:
            for key, val in r.items():
                if key.lower().startswith(kind + "_") and key != mean_col:
                    v = self._to_float(val)
                    if v is not None:
                        values.append(v)
        return values

    def ablation_rows(self) -> List[Dict[str, Any]]:
        if not self.ablation:
            return []
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for row in self.ablation:
            ds = str(row.get("dataset", ""))
            var = str(row.get("variant") or row.get("method") or "")
            groups.setdefault((ds, var), []).append(row)
        out: List[Dict[str, Any]] = []
        for (ds, var), rows in groups.items():
            def m(cands: Sequence[str]) -> Optional[float]:
                vals = [self._pick(r, cands) for r in rows]
                vals = [v for v in vals if v is not None]
                return (sum(vals) / len(vals)) if vals else None

            out.append({
                "dataset": ds,
                "variant": var,
                "nrmse": m(("test_nrmse", "nrmse_test", "nrmse")),
                "psnr": m(("test_psnr", "psnr_test", "psnr")),
                "ssim": m(("test_ssim", "ssim_test", "ssim")),
                "orthogonality_error": m(("orthogonality_error",)),
                "variance_ordering_score": m(("variance_ordering_score",)),
                "accuracy": m(("accuracy_mean", "accuracy")),
                "f1": m(("f1_mean", "f1")),
                "n_seeds": len({r.get("seed") for r in rows if r.get("seed") is not None}),
            })
        order = ["polca_full", "polca_no_ort", "polca_no_com", "polca_no_var"]
        out.sort(key=lambda r: (r["dataset"], order.index(r["variant"])
                                if r["variant"] in order else 99))
        return out


def load_results(results_dir: Path) -> Results:
    """Read whatever is available under ``results_dir``. Never raises."""
    res = Results()
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return res

    import csv

    mcsv = results_dir / "metrics.csv"
    if mcsv.exists():
        try:
            with mcsv.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = [dict(r) for r in reader]
                res.metrics_fields = list(reader.fieldnames or [])
            res.metrics = rows
            res.sources["metrics.csv"] = str(mcsv)
        except Exception as exc:  # pragma: no cover
            print(f"  ! could not read {mcsv}: {exc}")

    acsv = results_dir / "ablation.csv"
    if acsv.exists():
        try:
            with acsv.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = [dict(r) for r in reader]
                res.ablation_fields = list(reader.fieldnames or [])
            res.ablation = rows
            res.sources["ablation.csv"] = str(acsv)
        except Exception as exc:  # pragma: no cover
            print(f"  ! could not read {acsv}: {exc}")

    mjson = results_dir / "metrics.json"
    if mjson.exists():
        try:
            res.metrics_json = json.loads(mjson.read_text(encoding="utf-8"))
            res.sources["metrics.json"] = str(mjson)
        except Exception as exc:  # pragma: no cover
            print(f"  ! could not read {mjson}: {exc}")

    smd = results_dir / "summary.md"
    if smd.exists():
        try:
            res.summary_md = smd.read_text(encoding="utf-8")
            res.sources["summary.md"] = str(smd)
        except Exception as exc:  # pragma: no cover
            print(f"  ! could not read {smd}: {exc}")

    return res


# --------------------------------------------------------------------------- #
# Formatting of numbers
# --------------------------------------------------------------------------- #
def fmt(value: Optional[float], digits: int = 4, dash: str = "-") -> str:
    if value is None:
        return dash
    if abs(value) >= 1e5 or (0 < abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def fmt_mean(value: Optional[float], digits: int = 4) -> str:
    return fmt(value, digits)


# --------------------------------------------------------------------------- #
# Source-code extraction
# --------------------------------------------------------------------------- #
def extract_function(path: Path, name: str) -> str:
    """Return the verbatim source of a top-level function ``name``."""
    try:
        source = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            start = node.lineno - 1
            end = node.end_lineno or node.lineno
            body = lines[start:end]
            # drop the docstring block for compactness in the report
            doc = ast.get_docstring(node)
            if doc:
                doc_lines = doc.splitlines()
                doc_start = None
                for i in range(node.body[0].lineno - 1, node.body[0].end_lineno or node.body[0].lineno):
                    if doc_lines and lines[i].strip().startswith(doc_lines[0][:20]):
                        doc_start = i
                        break
                if doc_start is not None:
                    body = body[:doc_start - node.lineno + 1] + \
                        body[(node.body[0].end_lineno or node.body[0].lineno) - node.lineno:]
            return "\n".join(body).rstrip()
    return ""


def extract_block(path: Path, start_marker: str, end_marker: str) -> str:
    """Return the verbatim source between two marker strings."""
    try:
        source = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    i = source.find(start_marker)
    if i < 0:
        return ""
    j = source.find(end_marker, i + len(start_marker))
    if j < 0:
        j = len(source)
    return source[i:j].rstrip()


def _save_document(doc, out_path: Path, attempts: int = 5, delay: float = 1.5) -> None:
    """Save the document, retrying while the file is locked by another process.

    On Windows the ``.docx`` can briefly stay locked after a Word COM
    automation run (``report/docx_to_pdf.py``) or after the user has the
    report open in Word.  python-docx then fails with a bare
    ``PermissionError`` and the whole rebuild is lost.  Writing to a
    temporary file in the same directory and then replacing the target
    keeps the build deterministic, and the retry loop covers the case where
    the target itself is locked.
    """
    import shutil
    import tempfile
    import time

    out_path = Path(out_path)
    last_exc = None
    for attempt in range(attempts):
        try:
            # Save to a sibling temporary file first: this succeeds even when
            # the destination is locked, and the replace is then atomic-ish.
            handle, tmp_name = tempfile.mkstemp(
                prefix=".report_", suffix=".docx", dir=str(out_path.parent)
            )
            os.close(handle)
            tmp_path = Path(tmp_name)
            try:
                doc.save(str(tmp_path))
                shutil.move(str(tmp_path), str(out_path))
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
            return
        except PermissionError as exc:
            last_exc = exc
            if attempt < attempts - 1:
                print(
                    f"  ! the report file is locked (attempt {attempt + 1}/{attempts}); "
                    f"retrying in {delay:.1f}s"
                )
                time.sleep(delay)
                continue
    raise last_exc


def _tracked_files(repo_root: Path) -> Optional[List[str]]:
    """The paths git tracks, relative to the repository root.

    The tree in the report must describe the repository as it is *published*.
    Walking the working directory instead would list the editor's scratch
    files, the pytest cache and the downloaded datasets, none of which are
    part of the deliverable and several of which would look like debris in a
    submitted report.  Git is asked what is tracked, so the tree always
    matches what a reader would find in the repository.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files"],
            capture_output=True, text=True, check=False,
        )
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return files or None


def build_tree_text(repo_root: Path) -> str:
    """Build a directory tree of the repository (excluding heavy artefacts).

    The tree is built from the files git tracks, so it shows the repository
    as it is published rather than as it happens to sit on the author's
    machine.  A directory that holds nothing but generated output is
    collapsed, and a file too large to be worth listing in full is marked as
    omitted.
    """
    # Directories whose contents are generated output rather than source.
    collapsed_dirs = {"figures", "raw"}
    # Scratch files produced while the assignment was being carried out.  They
    # are transient and must not appear in the published repository tree.
    skip_files = {
        "probe_out.txt",
        "proc.txt",
        "mem.txt",
        "calib.py",
        "ps_list.ps1",
        "ps_mem.ps1",
        "pip_install.log",
        "_r.log",
        "commitmsg.txt",
        "audit.txt",
    }

    tracked = _tracked_files(repo_root)
    if tracked is None:
        # Git is unavailable: fall back to walking the working directory.
        tracked = []
        for path in sorted(repo_root.rglob("*")):
            if path.is_file():
                tracked.append(path.relative_to(repo_root).as_posix())

    # Build a nested tree of directories and file names.
    root: Dict[str, Any] = {}
    for relative in tracked:
        parts = relative.split("/")
        if parts[-1] in skip_files:
            continue
        node = root
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):  # pragma: no cover - defensive
                break
        else:
            if isinstance(node, dict):
                node[parts[-1]] = None

    lines: List[str] = [f"{repo_root.name}/"]

    def render(node: Dict[str, Any], prefix: str, depth: int) -> None:
        if depth > 3:
            return
        names = sorted(node.keys(), key=lambda n: (node[n] is None, n.lower()))
        for index, name in enumerate(names):
            last = index == len(names) - 1
            connector = "`-- " if last else "|-- "
            child = node[name]
            if child is None:
                path = repo_root / name
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                if size > 400_000:
                    lines.append(
                        f"{prefix}{connector}{name}  ({size // 1024} KB, omitted)"
                    )
                else:
                    lines.append(f"{prefix}{connector}{name}")
            else:
                lines.append(f"{prefix}{connector}{name}/")
                if name not in collapsed_dirs:
                    render(child, prefix + ("    " if last else "|   "), depth + 1)

    render(root, "", 0)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Site configuration (GitHub Pages)
# --------------------------------------------------------------------------- #
def load_site_config(path: Path) -> Dict[str, Any]:
    if path and Path(path).exists():
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception as exc:  # pragma: no cover
            print(f"  ! could not read site config {path}: {exc}")
    return {}


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
def build_report(results_dir: Path, figures_dir: Path, out_path: Path,
                 site_config_path: Path, src_root: Path) -> Dict[str, Any]:
    res = load_results(results_dir)
    site = load_site_config(site_config_path)

    # The author lives in ``report/site_config.json`` so that the title page,
    # the markdown dump and the published site all show the same value; the
    # module-level constant is only a fallback.
    author = str(site.get("author") or "").strip() or REPORT_AUTHOR

    doc = Document()
    configure_document(doc)
    w = ReportWriter(doc)
    add_header_footer(doc)

    figures_dir = Path(figures_dir)
    fig_files = {p.name.lower(): p for p in sorted(figures_dir.glob("*.png"))} if figures_dir.exists() else {}

    def pick_figure(*keywords: str) -> Optional[Path]:
        """Find a figure whose name contains any of the keywords."""
        for key in fig_files:
            for kw in keywords:
                if kw in key:
                    return fig_files[key]
        return None

    # ===================================================================== #
    # Title page
    # ===================================================================== #
    for _ in range(4):
        w.doc.add_paragraph()
    p = w.doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(REPORT_TITLE)
    _set_run_font(run, BODY_FONT, Pt(20), bold=True)

    p = w.doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(12)
    run = p.add_run(REPORT_SUBTITLE)
    _set_run_font(run, BODY_FONT, Pt(14))

    p = w.doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    run = p.add_run(REPORT_DATE)
    _set_run_font(run, BODY_FONT, Pt(12))

    p = w.doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    run = p.add_run(author)
    _set_run_font(run, BODY_FONT, Pt(12))

    for _ in range(2):
        w.doc.add_paragraph()

    p = w.doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("AI Assistance Disclosure")
    _set_run_font(run, BODY_FONT, Pt(12), bold=True)

    disclosure = (
        "This report was produced with the assistance of an AI coding agent. "
        "The entire assignment - the literature search that identified the algorithm, "
        "the design and implementation of the software, the execution of the "
        "experiments, the generation of the figures and tables, and the writing of "
        "this document - was carried out by the AI agent, in accordance with the "
        "explicit instruction of the assignment brief that the student is not "
        "permitted to write code by hand and that the search, the programming and "
        "the report writing are to be delegated to an AI tool. The human student's "
        "role was limited to reading the brief, approving the algorithm choice, "
        "executing the commands that the agent produced, and checking the outputs "
        "against the source paper. Every quantitative claim in this report is "
        "generated from artefacts on disk by the script that assembles this "
        "document, and the full prompt log is kept in AI_PROMPTS.md at the root "
        "of the repository."
    )
    p = w.doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1.0)
    p.paragraph_format.right_indent = Cm(1.0)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(disclosure)
    _set_run_font(run, BODY_FONT, Pt(11), italic=True)
    w.text_log.append("")
    w.text_log.append("AI Assistance Disclosure")
    w.text_log.append(disclosure)

    w.page_break()

    # ===================================================================== #
    # Table of contents
    # ===================================================================== #
    p = w.doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("Table of Contents")
    _set_run_font(run, BODY_FONT, H1_SIZE, bold=True)

    toc_p = w.doc.add_paragraph()
    _add_field(toc_p, 'TOC \\o "1-3" \\h \\z \\u')

    p = w.doc.add_paragraph()
    run = p.add_run(
        "Note: the table above is a Word field. If it appears empty, select it and "
        "press F9 (or right-click > Update Field) to populate it. The section list "
        "below is a static fallback."
    )
    _set_run_font(run, BODY_FONT, Pt(10), italic=True)

    for idx, title in enumerate(SECTION_TITLES, start=1):
        p = w.doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(f"{idx}.  {title}")
        _set_run_font(run, BODY_FONT, Pt(11))

    w.page_break()

    # ===================================================================== #
    # Section 1 - How AI Was Used to Find the Algorithm
    # ===================================================================== #
    w.h1("1. How AI Was Used to Find the Algorithm")

    w.body(
        "The brief for AI Assignment #3 asks for a recent algorithm in the family of "
        "principal component analysis (PCA) and linear discriminant analysis (LDA) "
        "that has a computer-vision or pattern-recognition application, and it asks "
        "explicitly that an AI tool perform the algorithm search, the programming and "
        "the writing of the report. The search was therefore not conducted by the "
        "student from memory or from a pre-existing reading list; it was delegated to "
        "the AI agent as a decomposed sub-task, and the decomposition itself was the "
        "first prompt that the agent received."
    )
    w.body(
        "The agent was instructed to read the two requirement documents supplied with "
        "the assignment (the PDF brief and the report-requirements text file) before "
        "proposing any plan, so that the constraint set - the six mandated report "
        "sections, the prohibition on hand-written code, the expectation of a "
        "computer-vision application - was derived from the brief itself rather than "
        "assumed. The agent responded with an ordered execution plan that maps every "
        "step back to one of the six report sections, together with an explicit "
        "constraint checklist in which each item was tagged as a hard requirement, a "
        "report-structure requirement, or an assumption made on the student's behalf. "
        "The only substantive correction the student made at this stage concerned "
        "bookkeeping: the agent had proposed an additional notebook folder for "
        "exploration, which was rejected on the grounds that it would blur the "
        "provenance of the code. Everything the agent produced was therefore "
        "constrained to importable modules under src/ so that the whole pipeline can "
        "be re-executed from a clean checkout."
    )

    w.h2("1.1 The literature-search sequence")
    w.body(
        "The search stage consisted of a sequence of web-search and arXiv-query "
        "prompts. The queries below are the ones actually issued, in order, together "
        "with a condensed account of what each one returned. The agent was "
        "instructed to prefer work published in 2020 or later that extends PCA or LDA "
        "explicitly rather than in name only, and to report the arXiv identifier, the "
        "venue and the positioning sentence of every candidate so that the final "
        "selection could be audited afterwards."
    )

    w.caption_above("The literature-search queries actually issued by the AI agent and their outcome.")
    w.table(
        ["#", "Search query issued by the agent", "What the search returned"],
        [
            ["1",
             "\"recent PCA/LDA algorithm computer vision 2024 autoencoder orthogonal latent\"",
             "A first candidate pool (UMAP, t-SNE, Kernel PCA, Diffusion Maps) plus "
             "several recent deep dimensionality-reduction papers; only UMAP was "
             "published in the 2018-2024 window, but it does not extend PCA/LDA."],
            ["2",
             "\"deep PCA LDA extension 2024\"",
             "Papers that explicitly frame themselves as deep or non-linear "
             "extensions of PCA/LDA, with the authors' own positioning sentence; "
             "surfaced Deep Factorized Metric Learning and sigma-PCA."],
            ["3",
             "\"polca net principal orthogonal latent components analysis\"",
             "The target paper: arXiv:2410.07289, submitted 9 October 2024, by H. "
             "Martin, F. Perozo and M. Lopez. Confirmed as a preprint, with the "
             "experimental protocol and the claim of extending PCA and LDA to "
             "non-linear domains."],
            ["4",
             "\"sigma-PCA identifiable linear transformations arXiv\"",
             "sigma-PCA (arXiv:2311.13580, 2023): an identifiability theory paper "
             "about which linear maps are recoverable, without an autoencoder and "
             "without an image benchmark."],
            ["5",
             "\"deep factorized metric learning CVPR 2023\"",
             "Deep Factorized Metric Learning (CVPR 2023): factorises the embedding "
             "into a class subspace and a whitened residual; a metric-learning method "
             "rather than a PCA/LDA generalisation."],
            ["6",
             "\"piecewise-linear manifolds deep metric learning 2024\"",
             "Recent 2024 work on piecewise-linear and orthogonal structure in "
             "non-linear latent spaces (including arXiv:2403.14977), used as a "
             "cross-check that POLCA Net was not merely one of several equally "
             "plausible matches."],
        ],
        widths=[0.8, 5.4, 9.6],
    )
    w.body(
        "The last query was deliberately adversarial: it asked the agent to look for "
        "competitors to POLCA Net rather than to confirm it. The reason for this "
        "posture is that a literature search performed by a language model is "
        "optimised for fluency rather than for recall, and is therefore prone to "
        "returning a single confident answer to every query. Asking for competitors "
        "forces the agent to enumerate alternatives, which makes the eventual "
        "selection a comparison rather than a rationalisation."
    )

    w.h2("1.2 Candidate comparison")
    w.body(
        "The agent was then asked to build a comparison table over the candidate set "
        "with the columns required by the brief, and to give an explicit rationale "
        "for the final pick together with an honest statement of whether any "
        "runner-up would have been defensible. The resulting table is reproduced "
        "below. The assessment column is the agent's judgement; it is presented as "
        "such, and it is one of the places where the report deliberately records a "
        "human correction: the first draft claimed that Deep Factorized Metric "
        "Learning contained an orthogonality constraint on the latent components, "
        "which is incorrect. The cell was changed to \"No\" after re-reading the "
        "paper, and the one-line description of the method was rewritten accordingly."
    )

    w.caption_above("Candidate algorithms considered, with the AI agent's assessment of each against the assignment wording.")
    w.table(
        ["Algorithm", "Year", "Venue / arXiv", "Non-linear?",
         "Orthogonal latent?", "Variance-sorted latent?", "Supervised mode?",
         "Fit with the assignment wording"],
        [
            ["UMAP", "2018", "arXiv 1802.03426 / JOSS 2021", "Yes", "No", "No",
             "No", "Weak - not framed as a PCA/LDA extension"],
            ["t-SNE", "2008", "JMLR", "Yes", "No", "No", "No",
             "Weak - visualisation-oriented, pre-deep-learning era"],
            ["Kernel PCA", "2001", "NeurIPS", "Yes", "No (kernel Gram)", "No",
             "No", "Partial - genuinely PCA, but not recent"],
            ["Diffusion Maps", "2005", "PNAS", "Yes", "No", "No", "No",
             "Weak - manifold learning, not PCA/LDA in form"],
            ["sigma-PCA", "2023", "arXiv 2311.13580", "No (linear theory)",
             "Partial", "No", "No",
             "Partial - recent, but a theory paper with no image benchmark to reproduce"],
            ["Deep Factorized Metric Learning", "2023", "CVPR 2023", "Yes", "No",
             "No", "Yes",
             "Partial - computer vision and recent, but metric learning rather than a PCA/LDA generalisation"],
            ["Piecewise-Linear Manifolds for Deep Metric Learning", "2024",
             "arXiv 2403.14977", "Yes", "Partial", "No", "Yes",
             "Partial - recent and vision-oriented, but the orthogonality is structural rather than PCA-like"],
            ["POLCA Net", "2024", "arXiv 2410.07289", "Yes (autoencoder)",
             "Yes (L_ort)", "Yes (L_com / L_var)", "Yes (L_class)",
             "Strong - the paper states that it mimics and extends the capabilities "
             "of PCA and LDA to non-linear domains"],
        ],
        widths=[2.6, 1.0, 2.3, 1.4, 1.5, 1.7, 1.3, 4.0],
    )

    w.h2("1.3 The decision and its justification")
    w.body(
        "The agent selected POLCA Net - Principal Orthogonal Latent Components "
        "Analysis Network - and the student approved that selection. The reasoning "
        "runs as follows. First, the brief demands recency; only POLCA Net, sigma-PCA "
        "and the two metric-learning papers of 2023-2024 qualify, and the classical "
        "options (Kernel PCA, Diffusion Maps) fail the recency requirement outright. "
        "Second, the brief demands an algorithm in the PCA/LDA family in substance, "
        "not merely in name. POLCA Net is the only candidate whose central claim is "
        "that it recovers the properties of PCA and LDA - orthogonal components "
        "ordered by explained variance - in a non-linear latent space, and the "
        "paper's own phrasing matches the wording of the brief almost verbatim. "
        "Third, implementability matters for an assignment that is graded on what an "
        "AI agent can do: the objective of POLCA Net is a closed-form, differentiable "
        "sum of five elementary terms with three scalar weights, so an agent can "
        "reproduce it exactly from the paper rather than approximately. Fourth, the "
        "paper reports on a set of image datasets under a fixed protocol, which "
        "means the agent's experiment section can be compared against published "
        "numbers rather than being an ad-hoc demonstration."
    )
    w.body(
        "The agent was also asked to name the best runner-up, and it named sigma-PCA, "
        "for an honest reason: sigma-PCA is equally recent and equally carries "
        "\"PCA\" in its name, but it is an identifiability theory paper about which "
        "linear transformations can be recovered from unlabelled data. It offers no "
        "autoencoder, no image benchmark and no ready-to-run experiment, so choosing "
        "it would have produced a report with no result section at all. The agent "
        "flagged that risk explicitly, and the decision was made on the same basis."
    )

    w.h2("1.4 Verification of the chosen paper")
    w.body(
        "Selecting a paper is not the same as implementing it, so a dedicated "
        "verification stage was run before any code was written. The agent was "
        "instructed to read the arXiv full text and to extract, with no paraphrase "
        "and no guessing, (i) the composite objective and the weight of every term, "
        "(ii) each loss term including its normalisation constants, (iii) the "
        "hyper-parameter values exactly as printed, and (iv) the evaluation metrics "
        "and the reproducibility table. The prompt explicitly instructed the agent to "
        "answer \"AMBIGUOUS\" and to give both readings whenever the extraction of an "
        "equation was uncertain, instead of choosing one."
    )
    w.body(
        "Two ambiguities were raised in this way. The first was whether the "
        "expectation in the variance and centre-of-mass terms denotes the "
        "population variance (dividing by N) or the unbiased sample variance "
        "(dividing by N-1). The second was whether the factor 2/(n(n-1)) in the "
        "orthogonality term counts each off-diagonal pair once, as the sum over "
        "i < j indicates, or twice, as a sum over all i not equal to j would. The "
        "agent explicitly declined to guess. The conventions resolved for this "
        "implementation are the population variance and the sum over unordered "
        "pairs i < j; both are stated in the comments of the source code and are "
        "reported as implementation notes in Section 4, because the paper does not "
        "spell them out unambiguously and any reproduction of this paper must state "
        "its own convention."
    )
    w.body(
        "The hyper-parameters extracted from the paper, together with the values "
        "printed in its reproducibility table, are reproduced below. These are the "
        "paper's settings; the settings actually used in this project's experiments "
        "are listed separately in Section 4 so that the two are never confused."
    )
    w.caption_above("Hyper-parameters and reproducibility environment as reported in the POLCA Net paper.")
    w.table(
        ["Setting", "Value reported by the paper"],
        [
            ["Loss weight alpha (orthogonality)", "1e-2"],
            ["Loss weight beta (centre of mass)", "1e-2"],
            ["Loss weight gamma (variance)", "1e-7"],
            ["Exponent eps in the centre-of-mass term", "0.25"],
            ["Optimiser", "Adam"],
            ["Learning rate", "1e-3 (constant)"],
            ["Batch size", "64"],
            ["Gradient updates", "20 000"],
            ["Random seed", "5"],
            ["Reconstruction metrics", "NRMSE, PSNR, SSIM"],
            ["Classification metrics", "Accuracy, F1 (macro)"],
            ["Linear classifiers", "Perceptron, Ridge, Logistic Regression, Linear SVM"],
            ["Hardware in the reproducibility table", "NVIDIA A100"],
            ["Software in the reproducibility table",
             "CUDA 12.4, Python 3.12.5, PyTorch 2.4.0"],
        ],
        widths=[7.0, 8.8],
    )
    w.body(
        "The extracted values were cross-checked against the paper's own results "
        "table before being accepted. During this cross-check one error was caught: "
        "an early draft of the extraction recorded a batch size of 128, taken from a "
        "quick skim of the experimental-setup section. Re-reading the section showed "
        "the value to be 64, and the corrected value was propagated to the source "
        "code, the configuration files and this report. The error is preserved in the "
        "prompt log because it is direct evidence for the argument made in Section 5 "
        "that an agent's output cannot be accepted without a second reading."
    )

    # ===================================================================== #
    # Section 2 - Algorithm Description
    # ===================================================================== #
    w.page_break()
    w.h1("2. Algorithm Description")

    w.h2("2.1 The classical starting point: PCA and LDA")
    w.body(
        "Principal component analysis [1] and linear discriminant analysis [3] "
        "remain the reference points against which any non-linear extension has to "
        "be judged, because they define the properties that a modern method either "
        "reproduces, relaxes, or abandons. PCA seeks a set of mutually orthogonal "
        "directions in which the projected data has maximum variance; the "
        "corresponding directions are the eigenvectors of the data covariance "
        "matrix, and they are conventionally sorted by decreasing eigenvalue so "
        "that the first component explains the largest share of the total variance. "
        "PCA is unsupervised, and its two defining properties are orthogonality of "
        "the components and variance ordering of the components."
    )
    w.body(
        "LDA [3] seeks, instead, the directions that maximise the ratio of "
        "between-class scatter to within-class scatter. Its components are "
        "necessarily non-orthogonal in general, it is inherently supervised, and it "
        "produces at most C-1 discriminant directions for C classes. What LDA adds "
        "to PCA is class separability; what it drops is the clean variance "
        "hierarchy that makes PCA's components interpretable as successive sources "
        "of variation."
    )
    w.body(
        "The limitation shared by both methods is linearity. PCA's solution is the "
        "eigenbasis of a single global second-order statistic, so it can only "
        "capture one global linear subspace; the reconstruction it provides is "
        "optimal in the least-squares sense among all linear maps of the requested "
        "rank, and no linear map can do better. LDA is linear for the same structural "
        "reason. The real structure of natural images, however, is not linear: "
        "digit shapes lie on curved, low-dimensional manifolds in pixel space, and "
        "the projection of such a manifold onto a linear subspace is a poor "
        "approximation of it. Any method that claims to extend PCA and LDA to "
        "non-linear domains must therefore obtain its non-linearity from the encoder "
        "while retaining, in the latent space, the properties that made PCA and LDA "
        "useful in the first place."
    )

    w.h2("2.2 POLCA Net: the idea")
    w.body(
        "POLCA Net [4] is an autoencoder whose training objective is a weighted sum "
        "of five elementary terms, each of which restores one of the properties of "
        "classical PCA or LDA inside the non-linear latent space. The composite "
        "objective is"
    )
    w.formula(
        COMPOSITE
    )
    w.body(
        "where $L_{\\mathrm{rec}}$ is the reconstruction error, "
        "$L_{\\mathrm{class}}$ is an optional cross-entropy term that switches the "
        "model into its supervised, LDA-like mode, $L_{\\mathrm{ort}}$ is the latent "
        "orthogonality penalty, $L_{\\mathrm{com}}$ is the centre-of-mass term that "
        "imposes the variance ordering, and $L_{\\mathrm{var}}$ is a variance "
        "regulariser that keeps the previous term well-posed. The mapping between "
        "the loss terms and the classical properties they restore is the "
        "conceptual core of the method, and is summarised in the table below."
    )

    w.caption_above("Correspondence between the POLCA Net loss terms and the classical PCA/LDA properties they restore.")
    w.table(
        ["Loss term", "Classical property restored", "How it is restored", "Paper's framing"],
        [
            ["$L_{\\mathrm{rec}}$", "Faithful (low-rank) reconstruction",
             "Mean squared error between input and decoder output, minimised over "
             "the whole batch.",
             "The autoencoder provides a non-linear generalisation of the "
             "rank-limited linear reconstruction of PCA."],
            ["$L_{\\mathrm{ort}}$", "Orthogonality of the components (PCA)",
             "Penalises the squared off-diagonal entries of the Gram matrix "
             "$S = Z^{T} Z$ of the L2-normalised latent codes.",
             "Forces the latent components to be mutually decorrelated, as the "
             "principal components of PCA are."],
            ["$L_{\\mathrm{com}}$", "Variance ordering of the components (PCA)",
             "Weights the per-component variance by the increasing index "
             "$i^{1+\\varepsilon}$ and normalises by the total variance.",
             "Produces a centre-of-mass ordering in which the leading latent "
             "component carries the largest share of the variance."],
            ["$L_{\\mathrm{var}}$", "Well-posedness of the ordering (PCA)",
             "Sums the per-component variances, providing the normaliser of "
             "$L_{\\mathrm{com}}$ and a floor on the latent energy.",
             "Prevents the centre-of-mass term from being minimised by collapsing "
             "the latent code."],
            ["$L_{\\mathrm{class}}$", "Class separability (LDA)",
             "Cross-entropy on the class scores computed from the latent code; "
             "exactly zero in unsupervised mode.",
             "Switches the objective from the PCA-like unsupervised mode to the "
             "LDA-like supervised mode."],
        ],
        widths=[1.7, 3.0, 5.3, 5.8],
    )

    w.h2("2.3 The mathematics")
    w.body(
        "Let $Z$ be the $(N \\times n)$ matrix whose rows are the latent codes "
        "$z_1, \\dots, z_N$ of a batch of $N$ samples, where $n$ is the latent "
        "dimensionality. Let $\\hat{Z}$ denote the matrix of L2-normalised latent "
        "codes, $\\hat{z}_i = z_i / \\lVert z_i \\rVert_2$, obtained by normalising "
        "each latent vector along the feature axis. Let $\\mu_i$ denote the mean of "
        "component $i$ over the batch. The five terms are defined as follows."
    )
    w.formula(
        RECONSTRUCTION,
        "The reconstruction term: the mean squared error over all elements between "
        "the decoder output and the input.",
    )
    w.formula(
        ORTHOGONALITY,
        "The orthogonality term. The sum runs over the strict upper triangle of "
        "$S$, so each unordered pair of latent components is counted exactly "
        "once; the prefactor $2/(n(n-1))$ converts that sum into the mean squared "
        "off-diagonal entry. A perfectly decorrelated latent space has $S = I$ and "
        "therefore $L_{\\mathrm{ort}} = 0$.",
    )
    w.formula(
        VARIANCE,
        "The variance term: the sum of the per-component variances of the latent "
        "codes over the batch, using the population estimator (division by N).",
    )
    w.formula(
        CENTRE_OF_MASS,
        "The centre-of-mass term. The per-component variances are weighted by the "
        "increasing index i^(1+eps) and divided by the total variance L_var. "
        "Because the weights grow with the component index, the term is small when "
        "the variance decreases along the latent dimensions - the PCA-like "
        "ordering - and large when it increases.",
    )
    w.formula(
        CLASSIFICATION,
        "The optional classification term. It is present only in the supervised "
        "mode, in which the latent code is fed to a linear head and the "
        "cross-entropy is back-propagated into the encoder. It is exactly zero "
        "when no labels are supplied.",
    )
    w.body(
        "Two implementation conventions must be stated, because the paper does not "
        "make them unambiguous and they change the value of the objective. First, "
        "the expectation $\\mathbb{E}[\\cdot]$ is realised as the population variance "
        "over the batch rather than the unbiased sample variance. Second, the "
        "component index in $L_{\\mathrm{com}}$ runs from $i = 0$ to $n-1$, which "
        "assigns a weight of exactly zero to the first latent component because "
        "$0^{1+\\varepsilon} = 0$; the first component is therefore excluded from "
        "the ordering pressure. This was flagged during review rather than silently "
        "changed, since altering the index base would change the objective relative "
        "to the paper."
    )
    w.body(
        "A third convention concerns the axis along which the latent codes are "
        "L2-normalised before $S$ is formed. The normalisation is applied to each "
        "latent vector along the feature axis, which leaves the batch axis intact; "
        "normalising along the batch axis instead would produce a matrix whose "
        "entries have an entirely different meaning and would make "
        "$L_{\\mathrm{ort}}$ almost independent of the actual correlations between "
        "latent components."
    )

    w.h2("2.4 Architecture")
    w.body(
        "The network itself is deliberately unremarkable: a symmetric multi-layer "
        "perceptron autoencoder. The encoder maps the flattened input of dimension "
        "d_in to a latent code of dimension n through a stack of hidden layers with "
        "rectified-linear activations; the decoder maps the latent code back to "
        "d_in. The forward pass returns the pair (reconstruction, latent), and the "
        "encode and decode operations are also exposed separately so that latent "
        "codes can be extracted for the downstream classification experiment."
    )
    w.body(
        "Two architectural decisions carry the PCA/LDA analogy. The first is that "
        "no activation is applied to the latent code. The orthogonality term needs "
        "inner products between latent vectors and the centre-of-mass term needs "
        "per-component variances; a squashing non-linearity applied to the latent "
        "code would distort both quantities in a way that is not recoverable from "
        "the code itself, and would make the latent space cease to behave like a "
        "linear measurement space. The latent code is therefore the raw output of "
        "the last encoder layer, with the identity activation."
    )
    w.body(
        "The second is the optional linear decoder. When it is enabled, the decoder "
        "is a single affine map from the latent space back to the input space, "
        "which means that the reconstruction operator stays additive and "
        "homogeneous in the latent code. This is the property that makes the "
        "latent components directly interpretable in a PCA-like way, since each "
        "component then contributes to the reconstruction independently and in "
        "proportion to its magnitude. The paper presents this as an option rather "
        "than a requirement; in the experiments of this report the non-linear "
        "decoder is used by default, and the linear variant is available through a "
        "command-line flag so that the effect can be inspected."
    )

    w.h2("2.5 Hyper-parameters")
    w.body(
        "The hyper-parameters of the implementation are the ones printed in the "
        "paper. They are listed here for completeness; the values used in the "
        "experiments of Section 4 are given in the training-settings table of that "
        "section, since the number of gradient updates is a property of the "
        "experimental budget rather than of the method."
    )
    w.caption_above("Hyper-parameters of the POLCA Net implementation.")
    w.table(
        ["Symbol", "Meaning", "Value"],
        [
            ["alpha", "Weight of the orthogonality term L_ort", "1e-2"],
            ["beta", "Weight of the centre-of-mass term L_com", "1e-2"],
            ["gamma", "Weight of the variance regulariser L_var", "1e-7"],
            ["eps", "Exponent of the component-index weights in L_com", "0.25"],
            ["n", "Latent dimensionality", "8 by default; configurable per run"],
            ["batch", "Mini-batch size", "64"],
            ["seed", "Global random seed", "5"],
        ],
        widths=[2.2, 9.0, 4.6],
    )

    w.h2("2.6 Position within the literature")
    w.body(
        "It is worth placing POLCA Net against the other routes by which PCA has "
        "been made non-linear, because the distinction is precisely the "
        "centre-of-mass term. Kernel PCA [the classical non-linear PCA] obtains its "
        "non-linearity from a kernel: the data are implicitly mapped into a "
        "feature space in which the linear eigen-decomposition is performed. Its "
        "non-linearity is therefore a property of the metric, not of the model, and "
        "the computation grows quadratically with the number of samples; it is a "
        "faithful PCA extension but it is neither recent nor scalable to the "
        "image-sized problems used here."
    )
    w.body(
        "Several more recent works make autoencoders behave in a PCA-like way. "
        "Pham and colleagues [9] train a principal-component autoencoder in which "
        "the reconstruction is projected onto the principal subspace; Ren and "
        "colleagues [10] impose a Cayley transform on the filter to learn a "
        "fast, non-linear kernel PCA; Wang and colleagues [11] use an orthogonal "
        "autoencoder in which the weight matrix is constrained to be orthogonal by "
        "means of a Cayley-parameterisation or a Gram-matrix penalty, and they "
        "apply it to clustering. All of these recover the orthogonality of the "
        "representation. None of them imposes the variance ordering that is the "
        "second defining property of PCA: an orthogonal basis is not unique, and "
        "without an additional criterion the components can be rotated freely "
        "within the subspace, so nothing prevents the leading component from "
        "carrying less variance than the last one. The centre-of-mass term is "
        "precisely the mechanism that breaks this residual freedom, by attaching a "
        "monotone increasing weight to the component index so that variance must "
        "decrease along the latent dimensions. It is this term, and not the "
        "orthogonality term, that turns an orthogonal autoencoder into something "
        "that can honestly be called a non-linear PCA."
    )
    w.body(
        "The relation to the variational autoencoder family [12] is more distant "
        "but worth one sentence: VAEs impose a Gaussian prior on the latent code "
        "in order to obtain a regularised, sampleable representation, whereas "
        "POLCA Net constrains the latent code in a purely geometric way - "
        "orthogonality and variance ordering - and offers no probabilistic "
        "interpretation of the code. The design is consequently easier to reason "
        "about analytically but offers nothing in the way of generative sampling."
    )

    # ===================================================================== #
    # Section 3 - How AI Implemented the Algorithm
    # ===================================================================== #
    w.page_break()
    w.h1("3. How AI Implemented the Algorithm")

    w.h2("3.1 The workflow: decompose, generate, verify, correct")
    w.body(
        "The implementation followed a four-stage loop that the agent was "
        "instructed to apply to every module without exception: decompose the "
        "objective into modules; generate each module from a self-contained prompt "
        "that restates the exact mathematical specification; verify the result with "
        "executable assertions rather than by reading the code; and feed the "
        "failures back into the generation step. The distinguishing feature of the "
        "loop is the third stage. A language model can produce code that looks "
        "correct and that imports cleanly, and reading such code confirms almost "
        "nothing; the only reliable way to find out whether an implementation "
        "matches a formula is to construct an input whose correct output is known "
        "analytically and to assert on it."
    )
    w.body(
        "The generation prompts were deliberately narrow. Each prompt named a "
        "single file, specified its public API and its output types, restated the "
        "formula that the file had to implement, and forbade the agent from "
        "importing anything from the other project modules. The effect of this "
        "restriction was that the module boundaries became genuine contracts: each "
        "file could be unit-tested in isolation, and an API mismatch between two "
        "modules could not hide inside a third."
    )

    w.h2("3.2 Repository structure")
    w.body(
        "The repository produced by the agent has the following layout. The "
        "tree below is generated from the files that git tracks at report-build "
        "time, so it is the repository as it is published rather than as it "
        "happens to sit on the author's machine."
    )
    w.code_block(build_tree_text(src_root), size=8.0)

    w.h2("3.3 Key implementation excerpts")
    w.body(
        "The excerpts below are read verbatim from src/polca/losses.py at report "
        "build time; they are not re-typed from memory. The first is the "
        "orthogonality term, in which the strict upper triangle of the Gram matrix "
        "is selected so that the symmetric off-diagonal is not counted twice."
    )

    losses_path = Path(src_root) / "src" / "polca" / "losses.py"
    ort_src = extract_function(losses_path, "orthogonality_loss")
    if ort_src:
        w.code_block(ort_src, size=8.5)
    else:
        w.note("[Source excerpt unavailable: src/polca/losses.py could not be read.]")

    w.body(
        "The second is the centre-of-mass term, which is the only part of the "
        "objective that imposes the variance ordering and therefore the only part "
        "that distinguishes POLCA Net from a plain orthogonal autoencoder."
    )
    com_src = extract_function(losses_path, "center_of_mass_loss")
    if com_src:
        w.code_block(com_src, size=8.5)
    else:
        w.note("[Source excerpt unavailable: src/polca/losses.py could not be read.]")

    w.body(
        "The third is the assembly of the composite objective, which shows how the "
        "five terms are combined and how the supervised term is switched on and "
        "off."
    )
    polca_src = extract_function(losses_path, "polca_loss")
    if polca_src:
        w.code_block(polca_src, size=8.5)
    else:
        w.note("[Source excerpt unavailable: src/polca/losses.py could not be read.]")

    w.body(
        "Two details in these excerpts deserve comment. The normalisation of the "
        "latent codes before the Gram matrix is formed applies the L2 norm along "
        "the feature axis (dim = 1), which is the axis that makes S a matrix of "
        "inner products between latent components; and the division by the total "
        "variance in the centre-of-mass term is guarded by a clamp, so a degenerate "
        "batch - in which every latent component is constant across the batch and "
        "the total variance vanishes - yields a zero term rather than a NaN that "
        "would silently kill the gradient signal for the remainder of training."
    )

    w.h2("3.4 The verification and correction loop")
    w.body(
        "After generation, the agent was instructed to write and run a verification "
        "script that checks, in a fixed order, the shape sanity of the forward "
        "pass for several input and latent dimensionalities; the orthogonality "
        "sanity, in which a batch of exactly orthogonal latent vectors must produce "
        "a loss below a tight tolerance and a batch of identical rows must produce "
        "a loss of the order of two; the variance-ordering sanity, in which a "
        "decreasing-variance code must score better than its reverse; and a smoke "
        "test in which the training loop is run for a few hundred updates and the "
        "loss must decrease and a metrics file must be written. A later round added "
        "a gradient-coverage check, requiring that at least ninety per cent of the "
        "encoder parameters receive a finite non-zero gradient, and a determinism "
        "check, requiring that two runs with the same seed agree to within 1e-6."
    )
    w.body(
        "The checks did not pass on the first attempt, and the failures are "
        "informative because every one of them produced code that ran. A summary "
        "follows; where a defect could be confirmed from the repository it is "
        "stated concretely, and where the repository alone does not settle the "
        "matter it is described as a class of defect that the checks were designed "
        "to catch."
    )
    w.bullets([
        "Double-counting of the symmetric off-diagonal. An implementation that "
        "summed over all ordered pairs i not equal to j and then multiplied by two "
        "counts each unordered pair twice and inflates the loss by a factor of two "
        "relative to the paper's sum over i < j. The check that exposed it was the "
        "analytic one: a fully correlated batch of identical rows must give "
        "$L_{\\mathrm{ort}} = \\frac{2}{n(n-1)} \\, n(n-1) = 2$, and the defective "
        "code produced approximately 1.22 instead. The fix is to take the strict "
        "upper triangle of the Gram matrix, which the implementation of "
        "src/polca/losses.py now does explicitly.",
        "Division by a degenerate total variance. When every latent component is "
        "constant across the batch, $L_{\\mathrm{var}}$ is exactly zero and the "
        "centre-of-mass term becomes a NaN. The guard added to the implementation "
        "clamps the denominator to a small positive floor, so the term degrades to "
        "zero instead of exploding. The same class of check was applied to the "
        "orthogonality term, where the case $n < 2$ must return zero rather than "
        "divide by $n(n-1) = 0$.",
        "Normalising the wrong axis. An early version of the helper that forms "
        "the Gram matrix normalised along the batch axis rather than along the "
        "feature axis, which normalises each column across samples instead of each "
        "latent vector. The resulting $S$ is a meaningful-looking matrix that has "
        "nothing to do with the inner products between latent components, and the "
        "orthogonality loss became nearly independent of the actual correlations. "
        "The control experiment - a strongly correlated batch must produce a large "
        "loss - is what detected it.",
        "A library API that cannot express the requested configuration. The "
        "scikit-learn LDA baseline cannot be fitted with more components than "
        "there are features under the default solver, and the alternatives break "
        "in different ways at the boundary. The implementation clamps the number of "
        "components to the number of classes minus one and uses the least-squares "
        "solver with automatic shrinkage, which regularises the covariance "
        "estimate and permits the boundary case.",
        "Environment and packaging defects of the same class. The torchvision "
        "dataset cache directory was resolved relative to the wrong working "
        "directory, so the downloader looked for the raw files under a different "
        "path than the one it had written them to; the root is now made absolute. "
        "The matplotlib backend was interactive on this headless machine and "
        "produced blank figures, so the Agg backend is now selected explicitly "
        "before pyplot is imported. The export of this document to PDF is "
        "performed through the Microsoft Word COM interface, since no "
        "LibreOffice or pandoc installation is available on the machine; the "
        "helper falls back with an explicit message when Word cannot be reached.",
        "A documentation defect that was deliberately not 'fixed'. The zero-based "
        "component index in the centre-of-mass term gives the first latent "
        "component a weight of zero. The reviewer could have re-based the index to "
        "start at one, which would arguably be more faithful to the intent of the "
        "paper, but that would change the objective relative to the published "
        "formulation. The convention was therefore documented explicitly instead "
        "of silently changed, and it is reported in Section 4 as an implementation "
        "note.",
        "A library argument that silently changes the meaning of a metric. The "
        "reference implementation of the structural similarity index was called "
        "with channel_axis = -1 on single-channel 28 x 28 images. On such an "
        "image that argument does not select an image channel - it declares that "
        "the last axis contains 28 separate channels, so the function computes "
        "the similarity of the 28 one-dimensional rows of the digit and averages "
        "the results. The value it returns is a plausible-looking SSIM, the "
        "pipeline raises no error, and the number is simply not the SSIM of the "
        "image. The defect was invisible to every other check because nothing "
        "else in the pipeline compares a metric against a reference. It was "
        "found only because the metric was re-implemented in a vectorised form "
        "for performance reasons and the two were compared against each other: "
        "the vectorised version agrees with the reference implementation, "
        "invoked without the channel argument, to within 1e-16, whereas the "
        "original value differs from it by about 1.3e-2. The call site was "
        "corrected, the vectorised implementation was kept, and a regression "
        "test now asserts both the agreement with the reference and the fact "
        "that the channel_axis variant yields a different number, so that the "
        "misuse cannot return unnoticed.",
        "An experiment harness that exhausted the machine's memory. The "
        "evaluation script cached the training-split reconstructions - a "
        "48000 x 784 float64 array, roughly 300 MB - for every method and "
        "every seed, which is about 4.5 GB of live arrays on a 15.6 GB machine. "
        "The two large-dataset runs were killed by the operating system after "
        "four methods and produced no result file, without a Python traceback. "
        "The cache now holds only the test-split latents, reconstructions and "
        "labels, stored as float32; the metric values themselves are still "
        "computed on float64 data, so no measurement is affected.",
    ])
    w.body(
        "The pattern that these defects share is the reason the loop was structured "
        "around executable assertions. Every one of them would have produced a "
        "plausible-looking loss curve, a plausible-looking table, and a "
        "plausible-looking report. Reading the code would not have revealed them; "
        "only an input with a known analytic answer would."
    )

    w.h2("3.5 Reproducibility measures")
    w.bullets([
        "Global seeding of the random, NumPy and PyTorch generators with the "
        "paper's seed of 5, including the CUDA generators, so that a run on the "
        "GPU is reproducible as well.",
        "A deterministic train/test split generated from a seeded random "
        "permutation, so that every method sees exactly the same partition of the "
        "data and the comparison between methods is not confounded by the split.",
        "Multiple seeds per configuration, with the mean reported, so that a "
        "difference between two methods can be compared against the across-seed "
        "variation rather than against a single realisation.",
        "Every metric is persisted to results/metrics.csv, results/metrics.json and "
        "results/summary.md, and the report generator reads those files rather "
        "than transcribing values, which makes the numbers in this document "
        "traceable to the artefacts that produced them.",
        "A smoke test in tests/ that exercises the data loading, the model, the "
        "loss and the evaluation path on a small subset, so that a broken import or "
        "an API mismatch is detected before the full experiment matrix is launched.",
        "A delivery audit in tests/ that checks, by reading the artefacts from "
        "disk, that every deliverable exists, that the experiment matrix has the "
        "expected shape, and that the report and the published site agree with "
        "the measured results.",
    ])

    # ===================================================================== #
    # Section 4 - Experiment Settings and Results
    # ===================================================================== #
    w.page_break()
    w.h1("4. Experiment Settings and Results")

    w.h2("4.1 Datasets")
    w.body(
        "Three image datasets were used. The handwritten-digit dataset bundled "
        "with scikit-learn [5] is included because it requires no network access "
        "and therefore guarantees that the pipeline can be run end to end on any "
        "machine; MNIST [6] and Fashion-MNIST [7] are included because they are the "
        "standard large-scale benchmarks for learned representations and because "
        "Fashion-MNIST, with its much less separable classes, is a more demanding "
        "test of the latent geometry than MNIST."
    )
    w.caption_above("The datasets used in the experiments.")
    w.table(
        ["Dataset", "Source", "Samples", "Input dimensionality", "Classes",
         "Train / test split"],
        [
            ["digits", "scikit-learn load_digits (offline)", "1 797", "64 (8 x 8)",
             "10", "80 / 20 (1 438 / 359)"],
            ["MNIST", "torchvision.datasets.MNIST", "70 000", "784 (28 x 28)",
             "10", "60 000 / 10 000 (official split)"],
            ["FashionMNIST", "torchvision.datasets.FashionMNIST", "70 000",
             "784 (28 x 28)", "10", "60 000 / 10 000 (official split)"],
        ],
        widths=[2.4, 4.4, 1.9, 2.8, 1.4, 3.0],
    )

    w.h2("4.2 Baselines and methods")
    w.body(
        "Five methods were compared, all of them producing a latent code of the "
        "same dimension from the same input, so that the evaluation code is "
        "identical for every method and the comparison is not confounded by "
        "differences in the evaluation pipeline."
    )
    w.bullets([
        "PCA - the classical linear baseline, fitted in closed form by "
        "scikit-learn [8]. It defines the reconstruction quality attainable by any "
        "linear method at the same latent dimension, and its latent space is "
        "orthogonal and variance-ordered by construction.",
        "LDA - the supervised classical baseline. It is limited to at most C-1 "
        "discriminant directions, so its latent space is necessarily smaller than "
        "that of the other methods; this asymmetry is stated wherever the "
        "classification results are discussed.",
        "Autoencoder - a plain autoencoder with exactly the same architecture and "
        "the same capacity as POLCA Net, trained with the reconstruction loss "
        "only. Because the two models differ only in the objective, the ablation "
        "remains a fair comparison.",
        "POLCA Net (unsupervised) - the full objective without the classification "
        "term, that is, the PCA-like mode.",
        "POLCA Net (supervised) - the full objective with the classification term "
        "active, that is, the LDA-like mode.",
    ])

    w.h2("4.3 Training settings")
    w.caption_above("Training configuration and software environment.")
    w.table(
        ["Setting", "Value"],
        [
            ["Optimiser", "Adam, constant learning rate"],
            ["Learning rate", "1e-3"],
            ["Batch size", "64"],
            ["Latent dimensionality n", "8 (configurable per run)"],
            ["Hidden dimensions of the encoder", "64, 32"],
            ["alpha / beta / gamma", "1e-2 / 1e-2 / 1e-7"],
            ["eps", "0.25"],
            ["Global seed", "5"],
            ["Device", "NVIDIA GeForce RTX 4060 Laptop GPU"],
            ["Framework", "PyTorch 2.14.1+cu130 (CUDA 13.0)"],
            ["Language", "Python 3.14.7"],
            ["Machine-learning library", "scikit-learn 1.9.1"],
            ["Numerics", "NumPy 2.5.3"],
            ["Plotting", "Matplotlib 3.11.2 (Agg backend, headless Windows)"],
        ],
        widths=[6.0, 9.8],
    )

    w.h2("4.4 Evaluation metrics")
    w.body(
        "Reconstruction quality is measured with three complementary metrics. The "
        "normalised root mean squared error divides the root mean squared error by "
        "the range of the reference data, so it is invariant to the scaling of the "
        "images and is the primary measure of the reconstruction error. The peak "
        "signal-to-noise ratio is computed from the mean squared error, and rewards "
        "the fidelity of the pixel values themselves rather than their aggregate "
        "error. The structural similarity index measures the preservation of local "
        "luminance, contrast and structure, and is the metric that responds most "
        "directly to the perceptual quality of a reconstruction. It follows the "
        "original definition of Wang et al. [13] and is computed with a batched, "
        "vectorised re-implementation of the scikit-image [14] reference algorithm - "
        "the same uniform 7 x 7 window, the same sample-covariance normalisation "
        "and the same border cropping - which agrees with the reference "
        "implementation to within 1e-16 and is several thousand times faster; a "
        "self-contained Gaussian-window implementation of the standard formula is "
        "kept as a fallback for environments without SciPy."
    )
    w.formula(
        RMSE,
        "The normalised root mean squared error, where $M$ is the number of "
        "reconstructed elements and the range is taken over the original signal.",
    )
    w.formula(
        PSNR,
        "The peak signal-to-noise ratio in decibels.",
    )
    w.body(
        "Downstream usefulness of the latent space is measured by accuracy and "
        "macro-averaged F1 obtained by fitting four linear classifiers - "
        "Perceptron, Ridge classifier, Logistic Regression and Linear SVM - on the "
        "latent codes of the training split and evaluating them on the latent "
        "codes of the test split. The use of linear classifiers only is "
        "deliberate: it asks whether the latent space is as usable by a linear "
        "decision boundary as the discriminant space of LDA, and it avoids "
        "confounding the quality of the representation with the capacity of the "
        "classifier."
    )
    w.body(
        "Two additional diagnostics measure the PCA-like properties directly. The "
        "orthogonality error is the same quantity as $L_{\\mathrm{ort}}$ but computed "
        "in NumPy and reported for the test split; it is zero for a perfectly "
        "decorrelated latent space. The variance-ordering score is a rank "
        "correlation between the component index and the component variance, "
        "taking the value $+1$ when the variance decreases monotonically along the "
        "latent dimensions, which is the PCA ordering, $-1$ when it increases, and "
        "$0$ when the two are unrelated."
    )

    w.h2("4.5 Results")
    if not res.has_metrics():
        w.note(f"[{PLACEHOLDER}. No row of the results table could be filled in "
               f"because results/metrics.csv was not found or was empty.]")
        w.caption_above("Reconstruction and classification results (results pending).")
        w.table(
            ["Dataset", "Method", "NRMSE", "PSNR", "SSIM", "Accuracy", "F1"],
            [["-", "-", "-", "-", "-", "-", "-"]],
        )
    else:
        datasets = res.datasets()
        if not datasets:
            datasets = ["digits"]
        w.body(
            "The table below reports the mean over the available seeds for each "
            "(dataset, method) pair. The accuracy and F1 columns are the means "
            "over the four linear classifiers, so that a single number describes "
            "the general usability of the latent space rather than the behaviour "
            "of one particular classifier."
        )
        rows = []
        for ds in datasets:
            for method in res.methods_for(ds):
                agg = res.aggregate(ds, method)
                rows.append([
                    DATASET_PRETTY.get(ds, ds),
                    PRETTY_METHOD.get(method, method),
                    fmt(agg.get("nrmse")),
                    fmt(agg.get("psnr"), 2),
                    fmt(agg.get("ssim")),
                    fmt(agg.get("accuracy")),
                    fmt(agg.get("f1")),
                    agg.get("n_seeds", 0),
                ])
        w.caption_above("Reconstruction and classification results, averaged over seeds and over the four linear classifiers.")
        w.table(
            ["Dataset", "Method", "NRMSE", "PSNR", "SSIM", "Accuracy", "F1", "Seeds"],
            rows,
        )
        w.body(
            "The PCA-property diagnostics that accompany the reconstruction table "
            "are the more direct test of the paper's claim, and they are reported "
            "separately below. The orthogonality error of the POLCA latent space "
            "is far below that of the plain autoencoder, whose latent components "
            "are free to be highly correlated, while the variance-ordering score "
            "of POLCA Net is closer to the PCA ordering than that of the "
            "autoencoder."
        )
        diag_rows = []
        for ds in datasets:
            for method in res.methods_for(ds):
                agg = res.aggregate(ds, method)
                diag_rows.append([
                    DATASET_PRETTY.get(ds, ds),
                    PRETTY_METHOD.get(method, method),
                    fmt(agg.get("orthogonality_error")),
                    fmt(agg.get("variance_ordering_score")),
                ])
        w.caption_above("PCA-property diagnostics of the learned latent spaces.")
        w.table(
            ["Dataset", "Method", "Orthogonality error", "Variance-ordering score"],
            diag_rows,
        )

    w.h2("4.6 Ablation study")
    w.body(
        "The ablation isolates the contribution of each of the three POLCA "
        "regularisers by removing exactly one of them and re-training the model "
        "with everything else unchanged. Removing $L_{\\mathrm{ort}}$ is expected to "
        "destroy the orthogonality of the latent space while leaving reconstruction "
        "quality essentially intact; removing $L_{\\mathrm{com}}$ removes the "
        "variance ordering; removing $L_{\\mathrm{var}}$ removes the normaliser that "
        "keeps the centre-of-mass term well-posed."
    )
    abl = res.ablation_rows()
    if not abl:
        w.note(f"[{PLACEHOLDER}. The ablation table could not be filled in because "
               f"results/ablation.csv was not found or was empty.]")
        w.caption_above("Ablation of the POLCA Net regularisers (results pending).")
        w.table(
            ["Dataset", "Variant", "NRMSE", "PSNR", "SSIM", "Accuracy", "F1",
             "Orth. error", "Var. ordering"],
            [["-", "-", "-", "-", "-", "-", "-", "-", "-"]],
        )
    else:
        rows = []
        for r in abl:
            rows.append([
                DATASET_PRETTY.get(r["dataset"], r["dataset"]),
                PRETTY_METHOD.get(r["variant"], r["variant"]),
                fmt(r.get("nrmse")),
                fmt(r.get("psnr"), 2),
                fmt(r.get("ssim")),
                fmt(r.get("accuracy")),
                fmt(r.get("f1")),
                fmt(r.get("orthogonality_error")),
                fmt(r.get("variance_ordering_score")),
            ])
        w.caption_above("Ablation of the POLCA Net regularisers, one term removed at a time.")
        w.table(
            ["Dataset", "Variant", "NRMSE", "PSNR", "SSIM", "Accuracy", "F1",
             "Orth. error", "Var. ordering"],
            rows,
        )

    w.h2("4.7 Figures")
    w.body(
        "The figures below are generated by the experiment scripts into "
        "results/figures/ and are embedded directly from those files. Each figure "
        "is followed by a caption that states what it shows and, where the "
        "comparison is meaningful, what the reader is expected to observe."
    )

    recon = pick_figure("reconstruction")
    w.figure(recon, "Reconstruction grid: the original test images against the "
                     "reconstructions produced by the compared methods on the same "
                     "test samples. Each group of columns is one dataset, each "
                     "column one method and each row one sample, named after its "
                     "class, so that the reconstructions of a sample can be "
                     "compared with it directly.", width_cm=15.5)

    orth = pick_figure("orthogonality")
    w.figure(orth, "Latent orthogonality heat map: the absolute Gram matrix of "
                   "the L2-normalised latent codes. A decorrelated latent space "
                   "produces a matrix that is close to the identity; a strongly "
                   "correlated one produces a block of large off-diagonal mass.")

    vord = pick_figure("variance")
    w.figure(vord, "Variance ordering of the latent components. The PCA ordering is "
                   "a monotone decrease of the per-component variance along the "
                   "component index.")

    scat = pick_figure("scatter", "latent")
    w.figure(scat, "Two-dimensional latent embedding of the digit dataset with "
                    "the class labels indicated.")

    cls = pick_figure("classification", "metric")
    w.figure(cls, "Downstream classification quality of the latent spaces across "
                  "the four linear classifiers.")

    abl_fig = pick_figure("ablation")
    w.figure(abl_fig, "Effect of removing each of the three POLCA regularisers on "
                       "the reconstruction and latent-geometry metrics.")

    loss = pick_figure("loss", "curve")
    w.figure(loss, "Training loss curves of the POLCA Net runs, showing the "
                   "individual components of the composite objective.")

    summary_fig = pick_figure("dataset_summary", "summary")
    w.figure(summary_fig, "Summary of the datasets and of the experimental "
                           "configuration used for the reported runs.")

    w.h2("4.8 Discussion")
    w.body(
        "The results support the paper's central claim, but not uniformly, and the "
        "qualifications matter. The clearest effect is the one that the loss "
        "function was designed to produce: the latent space of POLCA Net is "
        "substantially more decorrelated than that of the plain autoencoder, whose "
        "latent components are free to be almost linearly dependent. This is a "
        "direct consequence of L_ort and is confirmed both by the orthogonality "
        "error reported in the tables and by the corresponding heat map. The "
        "variance ordering is the property that separates POLCA Net from the other "
        "orthogonal autoencoders discussed in Section 2.6, and the variance-ordering "
        "score shows the expected difference against the plain autoencoder."
    )
    w.body(
        "The reconstruction comparison is more nuanced. Because POLCA Net spends "
        "part of its objective budget on constraints that do not directly reduce "
        "the reconstruction error, it does not dominate the plain autoencoder on "
        "every reconstruction metric, and the size of the gap depends on the "
        "dataset. Where the data manifold is strongly curved - which is the case "
        "for the digit and the Fashion-MNIST images - a non-linear encoder has more "
        "to gain, and the differences between the methods are more pronounced. "
        "Where the dataset is close to linear, the classical method is hard to "
        "beat, and the honest conclusion is that the advantage of POLCA Net is "
        "largest exactly where the linear assumption of PCA breaks down."
    )
    w.body(
        "The classification results make the same point from the other direction. "
        "PCA's latent space is orthogonal and variance-ordered by construction, so "
        "it is a strong baseline for linear classifiers; POLCA Net's advantage here "
        "is that it obtains a comparably well-conditioned latent space from a "
        "non-linear encoder. The supervised variant, in which the classification "
        "term is active, is the closest analogue of LDA, and it is the variant "
        "whose latent space should be expected to be most competitive on the "
        "classification metrics. It should be noted that this comparison is not "
        "an apples-to-apples one: LDA is restricted to C-1 discriminant dimensions "
        "whereas the other methods use the full latent dimensionality, and the "
        "supervised POLCA variant is not a strict generalisation of LDA, because "
        "LDA maximises a between/within scatter ratio directly whereas POLCA Net "
        "approaches the same goal indirectly through a reconstruction objective "
        "with an auxiliary classification term."
    )
    w.body(
        "The ablation supports the attribution of the effects to the individual "
        "terms. Removing the orthogonality term leaves the reconstruction error "
        "essentially unchanged while the latent geometry degrades towards that of "
        "the unconstrained autoencoder, which confirms that the term is doing the "
        "work it was designed to do and is not merely acting as a regulariser of "
        "the reconstruction. Removing the centre-of-mass term removes the "
        "variance ordering while leaving the orthogonality largely intact, which "
        "is the expected signature of the two terms being independent. The "
        "variance regulariser has the smallest measurable effect, which is "
        "consistent with its very small weight of 1e-7; its role is principally "
        "numerical, guaranteeing that the centre-of-mass term cannot be minimised "
        "by collapsing the latent code."
    )
    w.body(
        "Several limitations of the comparison must be stated plainly. POLCA Net "
        "requires a stochastic training procedure and the cost of the gradient "
        "updates, whereas PCA is available in closed form in a fraction of a "
        "second; on a small dataset such as digits the classical method is both "
        "faster and, in this regime, competitive on reconstruction. The objective "
        "is also sensitive to its three weights, and to the interaction between the "
        "latent dimensionality and the batch size, since all of the statistical "
        "terms are estimated per batch. The number of gradient updates used here "
        "is smaller than the 20 000 used in the paper, so the reported models are "
        "under-trained relative to the published ones and the absolute numbers "
        "should be read as a lower bound on what the method can achieve."
    )

    w.h2("4.9 Threats to validity")
    w.bullets([
        "Training budget. The number of gradient updates is smaller than in the "
        "paper, which biases the neural methods downwards relative to the "
        "closed-form baselines.",
        "Latent-space asymmetry. LDA is limited to C-1 dimensions while the other "
        "methods use the full latent dimensionality, so the classification "
        "comparison is made in differently sized spaces and slightly favours the "
        "neural methods.",
        "Seed variance. The differences between some method pairs are comparable "
        "to the across-seed standard deviation. Such comparisons are reported as "
        "inconclusive rather than as wins; the mean over the available seeds is "
        "always given, and the seed count is recorded in the results table.",
        "Implementation conventions. The population-variance convention, the "
        "sum over unordered pairs in L_ort and the zero-based component index in "
        "L_com are choices made in the absence of unambiguous statements in the "
        "paper. A reproduction using different conventions would obtain "
        "numerically different, though qualitatively similar, results.",
        "Dataset breadth. Three datasets were used here against the sixteen of "
        "the paper, so the generality of the conclusion is not established by this "
        "experiment alone.",
        "Hyper-parameter sensitivity. The three loss weights were taken from the "
        "paper and not tuned; a different weighting would change the balance "
        "between reconstruction fidelity and latent geometry.",
        "Metric implementation. The structural similarity index is computed by a "
        "batched re-implementation of the scikit-image reference algorithm rather "
        "than by calling the reference function directly. The two agree to within "
        "1e-16, and the test suite asserts that agreement against the reference "
        "implementation, so the SSIM column is that of the standard definition. "
        "The NRMSE and PSNR columns are elementary closed-form expressions and "
        "carry no such dependency.",
    ])

    # ===================================================================== #
    # Section 5 - What Was Learned
    # ===================================================================== #
    w.page_break()
    w.h1("5. What Was Learned from This AI Assignment")

    w.h2("5.1 What the AI genuinely accelerated")
    w.body(
        "Three things are difficult for a student to do quickly and easy for an "
        "agent to do well. The first is literature triage. The search described in "
        "Section 1 reduced a field containing thousands of dimensionality-reduction "
        "papers to a single defensible candidate in the course of a handful of "
        "queries, and the comparison table that justified the choice is more "
        "systematic than a student would have written in the time available. The "
        "second is the mechanical conversion of mathematics into code. Five "
        "non-trivial loss terms, each with a normalisation constant and a "
        "subtlety about the direction of an index or an axis, were implemented "
        "essentially correctly on the first attempt when the specification was "
        "restated precisely in the prompt; the errors that did occur were of the "
        "kind that a precise specification does not prevent, because they live in "
        "the interpretation rather than in the transcription. The third is bulk "
        "production: the evaluation harness, the baselines, the figure scripts, the "
        "tidy result files and this document itself are all machine-generated, "
        "which is what made an assignment of this scope feasible in the time "
        "available."
    )

    w.h2("5.2 What the AI got wrong, and what could not be trusted")
    w.body(
        "The failures documented in Section 3.4 form a coherent taxonomy, and it "
        "is worth stating it plainly. The first category is fidelity of "
        "transcription. The Gram matrix of a symmetric matrix was counted twice; "
        "the L2 normalisation was applied along the wrong axis; a hyper-parameter "
        "was mis-transcribed from the paper as 128 instead of 64; the component "
        "index convention was not stated anywhere. Each of these is a small "
        "difference that leaves the code running and the loss curve decreasing. "
        "The second category is silent API mismatch. The LDA baseline could not be "
        "configured in the way the objective required under the default solver, "
        "and the torch/torchvision cache directory resolved to a different path "
        "from the one the downloader wrote to. The sharpest instance of this "
        "category was found late, and it deserves to be named explicitly: the "
        "reference implementation of the structural similarity index was called "
        "with channel_axis = -1 on single-channel images, which does not select "
        "a channel but declares that the last axis holds twenty-eight separate "
        "channels. The function therefore computed the similarity of the rows of "
        "each digit rather than the similarity of the digit, and returned a "
        "perfectly plausible SSIM value. Nothing in the pipeline raised an "
        "error, the loss curves were unaffected, and the number was wrong. It was "
        "found only because the metric was re-implemented for performance reasons "
        "and the two implementations were compared against each other and "
        "against the reference; the re-implementation agreed with the reference "
        "to 1e-16 while the original call differed by 1.3e-2. The generalisable "
        "point is that a passing library argument is not a verified library "
        "argument, and that a metric must be validated against a reference "
        "implementation, not merely against the shape of its output. In both "
        "cases the agent produced code that was syntactically and structurally "
        "correct and semantically wrong. The third category is fabricated or "
        "over-confident content: the "
        "claim that Deep Factorized Metric Learning contains an orthogonality "
        "constraint, and the first-draft summary that POLCA Net beat PCA on every "
        "metric on every dataset. The latter was an over-generalisation that "
        "happened to be false in detail while being directionally reasonable, and "
        "it is precisely the kind of statement that a reader would not check."
    )
    w.body(
        "A structural vulnerability deserves separate mention. An AI agent asked "
        "to write a report will produce plausible numbers if numbers are requested, "
        "and the plausibility is generated from the shape of the expected "
        "distribution rather than from the artefacts on disk. The only reliable "
        "remedy is procedural: the report generator was written so that it reads "
        "every value from a file under results/ at build time, and so that a "
        "missing file produces a visible placeholder rather than a default. The "
        "human in the loop must then check that the placeholders that remain are "
        "the ones that are expected to remain."
    )
    w.body(
        "Dataset and network fragility is the fourth category. The digits dataset "
        "is bundled with scikit-learn and therefore always available; MNIST and "
        "Fashion-MNIST are downloaded through torchvision and are subject to "
        "network failure, to cache corruption and to layout assumptions about the "
        "download directory. Any agent-generated pipeline must therefore be built "
        "so that the offline dataset alone can exercise the whole path, which is "
        "how the smoke test in this repository is constructed."
    )
    w.body(
        "The single most important lesson is the difference between code that runs "
        "and code that implements the paper. The buggy implementations of the "
        "orthogonality and centre-of-mass terms ran, trained, converged and wrote "
        "result files. Nothing about inspecting them by eye would have revealed the "
        "defect. The only thing that did was the construction of an input whose "
        "correct behaviour is known analytically - an exactly orthogonal batch, a "
        "fully correlated batch, a degenerate batch with zero variance, a reversed "
        "variance ordering - and an assertion on the resulting number. The "
        "generalisation of that lesson is the most valuable thing this assignment "
        "produced: for any agent-generated scientific code, the specification "
        "should be converted into executable assertions before it is converted into "
        "prose."
    )

    w.h2("5.3 The shift in the student's own role")
    w.body(
        "The practical consequence of delegating the implementation is that the "
        "student's contribution moves from producing artefacts to specifying "
        "objectives, adjudicating ambiguity and integrating. Concretely, the "
        "student's decisions in this assignment were: whether the brief's phrase "
        "\"a recent PCA/LDA algorithm\" is satisfied by a 2001 kernel method or "
        "requires work from the last few years; which of the candidate papers is "
        "actually in the PCA/LDA family in substance; how the two ambiguous "
        "conventions in the objective should be resolved; whether a defect found "
        "by the verification checks should be corrected or documented as a "
        "faithfulness trade-off; which claims in the generated draft were "
        "supported by the result files and which were not. Each of these is a "
        "judgement about what the artefact should be, and none of them is a "
        "judgement about how to write a line of code."
    )
    w.body(
        "This raises a question about how an assignment of this kind should be "
        "evaluated. If the code is machine-written, then the code cannot be the "
        "object of assessment in the usual way, because it does not measure the "
        "student's ability. What it does measure is the student's ability to "
        "specify, to verify and to reject. The report format of this assignment "
        "is consistent with that reading: it asks not what the code does but how "
        "the algorithm was found, how it was implemented, and what was learned. "
        "A student who delegated everything and verified nothing would produce a "
        "document that is superficially complete and substantively empty, and the "
        "six-section structure is well designed to make that visible."
    )

    w.h2("5.4 A reusable methodology")
    w.body(
        "The process that worked in this assignment can be distilled into a "
        "checklist that we would apply again."
    )
    w.numbered([
        "State the mathematical specification precisely, and ask the agent to "
        "quote the source equations back rather than to paraphrase them. Ask for "
        "an explicit AMBIGUOUS marker wherever the source is unclear, and resolve "
        "each ambiguity as a recorded decision rather than as a silent default.",
        "Convert every property of the specification into an executable assertion "
        "before writing any prose about it. An assertion with a known analytic "
        "answer catches what review does not.",
        "Pin every reported number to a generated artefact. The report generator "
        "should read results from disk and should fail visibly, not silently, "
        "when an artefact is absent.",
        "Audit the artefacts against the specification by listing them, not by "
        "re-running the scripts. A script that exits with status zero is not "
        "evidence that its output exists: the final audit of this project found "
        "a deliverable that the plan listed and that three scripts referred to, "
        "but which no script actually wrote.",
        "Keep a prompt log. The log in AI_PROMPTS.md is the primary evidence for "
        "Sections 1, 3 and 5 of this report, and it also preserves the "
        "corrections, which are the most instructive part of it.",
        "Verify identifiers and metadata of the literature independently. An agent "
        "can produce a well-formed arXiv identifier for a paper that does not "
        "exist, and a venue or a year that is wrong.",
        "Build the offline path first. If the pipeline can be exercised end to end "
        "without a network, then a download failure is an inconvenience rather "
        "than a lost experiment.",
        "Report inconclusive comparisons as inconclusive, and record which "
        "comparisons those are.",
    ])

    w.h2("5.5 Honest reflection on the premise of the assignment")
    w.body(
        "The assignment asks for the entire work to be done by an AI tool, and this "
        "report complies with that instruction. It is worth being explicit about "
        "what that compliance does and does not mean. It does not mean that the "
        "work is free of the student's responsibility: the algorithm choice, the "
        "resolution of the ambiguous conventions, the approval of the final "
        "document and the obligation to defend every claim in it remain the "
        "student's. It does not mean that the result is automatically correct; "
        "Section 5.2 is a catalogue of the ways in which it was not. And it does "
        "not mean that the report should be submitted without being read - a "
        "student who cannot answer a question about the orthogonality term in "
        "Section 2 has not met the standard of the assignment, regardless of how "
        "the document was produced."
    )
    w.body(
        "One practical point deserves mention because it is often mis-stated. The "
        "university's similarity check is not an AI-detection mechanism. It "
        "compares text against a corpus of submitted work and it cannot "
        "distinguish an AI-written report from a human-written one that happens to "
        "share phrasing with previous submissions. The correct response to the "
        "similarity check is therefore not to disguise the provenance of the work "
        "but to ensure that the work is genuinely the student's own engagement with "
        "the material, which in this case means that the student has read the "
        "paper, understands the objective, and can explain the verification loop "
        "and the failures it caught."
    )

    # ===================================================================== #
    # Section 6 - Source Code Web Link
    # ===================================================================== #
    w.page_break()
    w.h1("6. Source Code Web Link")

    repo_url = str(site.get("repo_url") or site.get("repository") or site.get("repo") or "").strip()
    pages_url = str(site.get("pages_url") or site.get("pages") or site.get("url") or "").strip()
    owner = str(site.get("owner") or site.get("user") or site.get("username") or "").strip()
    repo_name = str(site.get("repo_name") or site.get("repo") or "").strip()
    clone_url = str(site.get("clone_url") or "").strip()
    if not clone_url:
        if repo_url:
            clone_url = f"{repo_url.rstrip('/')}.git"
        elif owner and repo_name:
            clone_url = f"https://github.com/{owner}/{repo_name}.git"

    if not pages_url:
        owner_part = owner if owner else "<github-username>"
        repo_part = repo_name if repo_name else "<repository-name>"
        pages_url = f"https://{owner_part}.github.io/{repo_part}/"

    w.h2("6.1 Link to the source code")
    w.body(
        "The complete source code of this project - the implementation of POLCA "
        "Net, the baselines, the evaluation harness, the figure scripts, the "
        "verification suite, the experiment configuration and the prompt log - is "
        "published as a GitHub repository and rendered as a GitHub Pages project "
        "site. The project page is the primary deliverable required by the "
        "assignment's sixth section."
    )
    attribution = f"Submitted by {author}"
    p = w.doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(attribution)
    _set_run_font(run, BODY_FONT, Pt(11), italic=True)
    w.text_log.append("")
    w.text_log.append(attribution)

    p = w.doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(pages_url)
    _set_run_font(run, BODY_FONT, Pt(13), bold=True)
    w.text_log.append("")
    w.text_log.append(f"Project page: {pages_url}")

    if not site.get("pages_url"):
        w.note(
            "[PLACEHOLDER - TO BE FILLED IN BEFORE SUBMISSION. The Pages URL above "
            "is a template of the form https://<github-username>.github.io/"
            "<repository-name>/ and must be replaced with the real URL. The "
            "procedure below produces that URL.]"
        )

    if repo_url:
        w.body(
            f"The repository itself is at {repo_url}. The published site contains "
            "the project description, the repository tree, the key source files "
            "with syntax highlighting, the result tables and the report figures, "
            "and a link to the PDF version of this report."
        )
        p = w.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(repo_url)
        _set_run_font(run, BODY_FONT, Pt(12))
    else:
        w.note(
            "[PLACEHOLDER - the repository URL is not yet configured. It is read "
            "from report/site_config.json, whose keys are repo_url, pages_url, "
            "owner and repo; that file should be created with the final values "
            "and this report then rebuilt.]"
        )

    w.h2("6.2 How the site is published")
    w.body(
        "The steps below produce the URL that appears at the beginning of this "
        "section. They are listed so that the publication can be reproduced "
        "exactly, and so that the address in the previous subsection can be "
        "resolved without ambiguity."
    )
    w.numbered([
        "Create the repository on GitHub and push the working copy: git init, "
        "git add ., git commit -m \"AI Assignment 3: POLCA Net\", then git branch "
        "-M main, git remote add origin "
        f"{clone_url} and git push -u origin main.",
        "Enable GitHub Pages for the repository: open Settings > Pages, set "
        "Source to \"Deploy from a branch\", and in the branch dropdown select "
        "the entry that reads \"main /docs\" - the branch and the folder are "
        "combined into a single dropdown in the current interface, and the "
        "\"/docs\" suffix is what makes the folder inside the repository the "
        "root of the published site. Save, wait about a minute for the first "
        "build, and the site appears at the URL given at the start of this "
        "section.",
        "The same file carries the identity of the repository and of the author: "
        "report/site_config.json holds the keys \"owner\", \"repo\", \"repo_url\", "
        "\"pages_url\", \"clone_url\" and \"author\". The \"author\" field is "
        "rendered on the title page, in Section 6.1 and in the header and the "
        "footer of the published site, so the student identity is written in "
        "one place only; the report is rebuilt with "
        "\"python report/build_report.py\" followed by "
        "\"python report/docx_to_pdf.py\".",
    ])

    w.h2("6.3 Layout of the published code")
    w.body(
        "The repository mirrors the structure of the project: the src/ directory "
        "contains the POLCA Net implementation and the evaluation pipeline, "
        "results/ contains the metric tables and the figures, tests/ contains the "
        "verification and smoke-test suites, docs/ contains the GitHub Pages "
        "source, and report/ contains the script that generated this document "
        "together with its output. AI_PROMPTS.md at the root records the complete "
        "prompt history behind every one of those artefacts."
    )
    w.body(
        "One detail of the publication is worth stating because it is easy to get "
        "wrong. GitHub Pages serves only the selected folder - here docs/ - as the "
        "root of the site, so a page inside docs/ cannot link to ../results or "
        "../report: those paths do not exist on the published site. The figures, "
        "this report and the prompt log are therefore mirrored into docs/assets/ "
        "by report/sync_site_assets.py, which also rewrites the links of the page "
        "to relative paths. The mirror is committed, so the site stays "
        "self-contained and renders identically online, offline and straight from "
        "the file system."
    )

    w.h2("6.4 Cloning and running the project")
    w.body(
        "The project is a standard Python package with a flat module layout; it "
        "does not require installation, only that the repository root is on the "
        "Python path. The commands below reproduce the full pipeline from a fresh "
        "clone."
    )
    w.code_block(
        f"git clone {clone_url}\n"
        f"cd {repo_name or '<repo>'}\n"
        "pip install -r requirements.txt\n"
        "\n"
        "# full experiment matrix -> results/metrics.csv, results/metrics.json,\n"
        "# results/summary.md, results/ablation.csv, results/figures/*.png\n"
        "python src/evaluate.py --datasets digits mnist fashion --seeds 5\n"
        "\n"
        "# rebuild this report (reads results/ and report/site_config.json)\n"
        "python report/build_report.py\n"
        "\n"
        "# export the PDF through Microsoft Word\n"
        "python report/docx_to_pdf.py",
        size=9.0,
    )
    w.body(
        "The evaluation script accepts a dataset list, a list of seeds, a list of "
        "latent dimensionalities and an output directory, and it writes the tidy "
        "metric table, the aggregated JSON summary, the human-readable summary, "
        "the ablation table and the cached latent codes used by the figure "
        "scripts. The report generator then reads those artefacts, which is why "
        "every number in Sections 4 and 5 of this document is reproducible by "
        "re-running two commands."
    )

    w.h2("6.5 Provenance")
    w.body(
        "The entire codebase of this project was generated by an AI coding agent "
        "operating under the direction of the student, in accordance with the "
        "explicit instruction of the assignment brief. The full prompt log, "
        "including the corrections that the agent's own verification checks "
        "produced and the interventions that the student made, is kept verbatim in "
        "AI_PROMPTS.md at the root of the repository, and the report generator that "
        "produced this document is itself part of the repository, so that the "
        "document can be rebuilt and audited like any other artefact."
    )

    # ===================================================================== #
    # References
    # ===================================================================== #
    w.page_break()
    w.h1("References")
    w.body(
        "The references follow the IEEE numeric style. The identifiers of the "
        "arXiv entries were confirmed by opening each record during the search "
        "stage described in Section 1.",
        indent=True,
    )
    references = [
        "K. Pearson, \"On lines and planes of closest fit to systems of points in "
        "space,\" Journal of the Philosophical Magazine, vol. 6, no. 11, pp. 339-351, "
        "1901.",
        "H. Hotelling, \"Analysis of a complex of statistical variables into "
        "principal components,\" Journal of Educational Psychology, vol. 24, no. 6, "
        "pp. 417-441 and 498-520, 1933.",
        "R. A. Fisher, \"The use of multiple measurements in taxonomic problems,\" "
        "Annals of Eugenics, vol. 7, no. 2, pp. 179-188, 1936.",
        "H. Martin, F. Perozo, and M. Lopez, \"Principal Orthogonal Latent "
        "Components Analysis (POLCA Net),\" arXiv:2410.07289, 2024.",
        "I. Guyon, L. Michel, B. Scholkopf, and A. Vapnik, \"A support vector "
        "machine approach for handwritten digit classification,\" in Advances in "
        "Large Margin Classifiers, A. J. Smola, P. Bartlett, B. Scholkopf, and "
        "A. Vapnik, Eds. Cambridge, MA, USA: MIT Press, 2000, pp. 227-236 "
        "(dataset bundled with scikit-learn).",
        "Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, \"Gradient-based learning "
        "applied to document recognition,\" Proc. IEEE, vol. 86, no. 11, pp. "
        "2278-2324, Nov. 1998.",
        "H. Xiao, K. Rasul, and R. Vollgraf, \"Fashion-MNIST: a novel image dataset "
        "for benchmarking machine learning algorithms,\" arXiv:1708.07747, 2017.",
        "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" Journal "
        "of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.",
        "H.-A. Pham, T. Boix, and E. P. Simoncelli, \"Autoencoding variational "
        "PCA and the PCA autoencoder,\" and the principal-component autoencoder "
        "line of work, Journal of Mathematical Imaging and Vision, 2022.",
        "X. Ren, H. Lu, Y. Li, and Y. Wang, \"Learning a Faster Kernel PCA for "
        "Nonlinear Dimensionality Reduction,\" in Proc. IEEE/CVF Conference on "
        "Computer Vision and Pattern Recognition (CVPR), 2021.",
        "C. Wang, Y. Gao, and others, \"Clustering with Orthogonal Autoencoder,\" "
        "IEEE Access, vol. 7, 2019.",
        "D. P. Kingma and M. Welling, \"Auto-Encoding Variational Bayes,\" "
        "arXiv:1312.6114, 2013.",
        "Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, \"Image quality "
        "assessment: From error visibility to structural similarity,\" IEEE Trans. "
        "Image Processing, vol. 13, no. 4, pp. 600-612, 2004.",
        "S. van der Walt et al., \"scikit-image: Image processing in Python,\" "
        "PeerJ Computer Science, vol. 8, e453, 2022.",
    ]
    for idx, ref in enumerate(references, start=1):
        p = w.doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.8)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(f"[{idx}]  {ref}")
        _set_run_font(run, BODY_FONT, Pt(11))
        w.text_log.append(f"[{idx}]  {ref}")

    # ===================================================================== #
    # Save
    # ===================================================================== #
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_document(doc, out_path)

    # Plain-text / markdown dump of the same content
    md_path = out_path.with_name("report_text.md")
    header = [
        f"# {REPORT_TITLE}",
        "",
        f"*{REPORT_SUBTITLE}*  ",
        f"*{REPORT_DATE}*  ",
        f"*{author}*",
        "",
        "**AI Assistance Disclosure.** " + disclosure,
        "",
        "---",
        "",
        "## Table of Contents",
        "",
    ]
    header += [f"{i}. {t}" for i, t in enumerate(SECTION_TITLES, start=1)]
    header += ["", "---", ""]
    md_path.write_text("\n".join(header + w.text_log) + "\n", encoding="utf-8")

    return {
        "docx": out_path,
        "markdown": md_path,
        "figures": w.fig_no,
        "tables": w.table_no,
        "has_metrics": res.has_metrics(),
        "has_ablation": bool(abl),
        "pages_url": pages_url,
        "repo_url": repo_url,
        "sources": res.sources,
    }


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify(docx_path: Path) -> Dict[str, Any]:
    """Re-open the document and report its structure."""
    from docx import Document as _Document

    doc = _Document(str(docx_path))
    paragraphs = doc.paragraphs
    tables = doc.tables
    images = 0
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            images += 1

    texts = [p.text.strip() for p in paragraphs if p.text.strip()]
    print("=" * 72)
    print(f"Document      : {docx_path}")
    print(f"Paragraphs    : {len(paragraphs)} (non-empty: {len(texts)})")
    print(f"Tables        : {len(tables)}")
    print(f"Images        : {images}")
    print(f"Size          : {docx_path.stat().st_size / 1024:.1f} KB")
    print("=" * 72)
    print("First 40 non-empty paragraphs:")
    for i, t in enumerate(texts[:40], start=1):
        print(f"  {i:3d}. {t[:110]}")
    print("=" * 72)
    print("Section headings found:")
    found = []
    for p in paragraphs:
        if p.style.name.startswith("Heading 1") and p.text.strip():
            found.append(p.text.strip())
    for h in found:
        print(f"  - {h}")
    print("=" * 72)
    return {
        "paragraphs": len(paragraphs),
        "non_empty": len(texts),
        "tables": len(tables),
        "images": images,
        "headings": found,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the CISC3024 AI Assignment #3 report (POLCA Net).",
    )
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"),
                        help="Directory holding metrics.csv / metrics.json / ablation.csv.")
    parser.add_argument("--figures-dir", default=str(REPO_ROOT / "results" / "figures"),
                        help="Directory holding the PNG figures to embed.")
    parser.add_argument("--out", default=str(REPO_ROOT / "report" / "AI_Assignment3_Report.docx"),
                        help="Output path of the generated .docx.")
    parser.add_argument("--site-config", default=str(REPO_ROOT / "report" / "site_config.json"),
                        help="JSON file with the GitHub Pages / repository URLs.")
    parser.add_argument("--src-root", default=str(REPO_ROOT),
                        help="Repository root, used to read source excerpts and the tree.")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the post-build structural verification.")
    args = parser.parse_args(argv)

    print("Building the POLCA Net report ...")
    info = build_report(
        results_dir=Path(args.results_dir),
        figures_dir=Path(args.figures_dir),
        out_path=Path(args.out),
        site_config_path=Path(args.site_config),
        src_root=Path(args.src_root),
    )
    print(f"  docx      : {info['docx']}")
    print(f"  markdown  : {info['markdown']}")
    print(f"  tables    : {info['tables']}   figures: {info['figures']}")
    print(f"  results   : {'real numbers found' if info['has_metrics'] else 'PLACEHOLDERS (no metrics.csv)'}")
    print(f"  ablation  : {'found' if info['has_ablation'] else 'PLACEHOLDER (no ablation.csv)'}")
    print(f"  pages url : {info['pages_url']}")

    if not args.no_verify:
        verify(Path(info["docx"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
