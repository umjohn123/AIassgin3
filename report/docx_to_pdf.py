#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert the generated report from .docx to .pdf using Microsoft Word.

The machine used for this assignment has Microsoft Word installed but neither
LibreOffice nor pandoc, so the conversion is performed through the Word COM
automation interface (``win32com.client``).  The exact sequence is::

    import win32com.client as win32
    app  = win32.Dispatch('Word.Application')
    app.Visible = False
    doc  = app.Documents.Open(docx_path)
    doc.SaveAs(pdf_path, FileFormat=17)   # 17 = wdFormatPDF
    doc.Close(0)
    app.Quit()

If Word is not available (not installed, not activated, COM blocked by policy)
the script prints an explicit, actionable message and exits with a non-zero
status rather than failing silently.  The .docx is always left intact, so the
submission can still be delivered in Word format.

Usage
-----
    python report/docx_to_pdf.py
    python report/docx_to_pdf.py --docx report/AI_Assignment3_Report.docx \
        --pdf report/AI_Assignment3_Report.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCX = REPO_ROOT / "report" / "AI_Assignment3_Report.docx"
DEFAULT_PDF = REPO_ROOT / "report" / "AI_Assignment3_Report.pdf"

#: ``wdFormatPDF`` in the Word object model.
WD_FORMAT_PDF = 17


class WordUnavailable(RuntimeError):
    """Raised when Microsoft Word cannot be reached through COM automation."""


def convert(docx_path: Path, pdf_path: Path, verbose: bool = True) -> Path:
    """Convert ``docx_path`` to ``pdf_path`` through the Word COM interface."""
    docx_path = Path(docx_path).resolve()
    pdf_path = Path(pdf_path).resolve()

    if not docx_path.exists():
        raise FileNotFoundError(f"The source document does not exist: {docx_path}")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pdf_path.exists():
        try:
            pdf_path.unlink()
        except OSError as exc:  # pragma: no cover
            print(f"  ! could not remove the existing PDF: {exc}")

    try:
        import win32com.client as win32  # type: ignore
    except ImportError as exc:
        raise WordUnavailable(
            "pywin32 is not installed. Install it with:\n"
            "    pip install pywin32\n"
            "and then re-run:\n"
            "    python report/docx_to_pdf.py"
        ) from exc

    if verbose:
        print(f"  opening Word ...")

    word = None
    document = None
    try:
        try:
            word = win32.Dispatch("Word.Application")
        except Exception as exc:
            raise WordUnavailable(
                "Microsoft Word could not be started through COM automation "
                f"({exc}).\n"
                "Possible causes and remedies:\n"
                "  * Word is not installed on this machine.\n"
                "  * Word is not activated / is running in reduced-functionality mode.\n"
                "  * COM automation is blocked by a group policy.\n"
                "  * A first-run dialog is pending; open Word once manually and\n"
                "    accept the licence agreement, then re-run this script.\n"
                "The .docx report is intact and can be submitted as-is, or exported\n"
                "to PDF manually with File > Export > Create PDF/XPS."
            ) from exc

        word.Visible = False
        try:
            word.DisplayAlerts = 0
        except Exception:
            pass

        document = word.Documents.Open(str(docx_path), ReadOnly=False)
        document.SaveAs(str(pdf_path), FileFormat=WD_FORMAT_PDF)
        document.Close(0)
        document = None
        word.Quit()
        word = None
    finally:
        # Make sure no orphaned WINWORD.EXE is left behind.
        if document is not None:
            try:
                document.Close(0)
            except Exception:
                pass
        if word is not None:
            try:
                word.Quit()
            except Exception:
                pass

    if not pdf_path.exists():
        raise WordUnavailable(
            f"Word reported success but no PDF was written to {pdf_path}. "
            "Check the folder permissions and re-run."
        )
    return pdf_path


def check_pdf(path: Path) -> bool:
    """Cheap sanity check: the file exists, is large enough and starts with %PDF."""
    try:
        with open(path, "rb") as fh:
            header = fh.read(4)
    except OSError:
        return False
    size = path.stat().st_size
    return header == b"%PDF" and size > 100 * 1024


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert the report .docx to .pdf using Microsoft Word COM automation.",
    )
    parser.add_argument("--docx", default=str(DEFAULT_DOCX), help="Source .docx file.")
    parser.add_argument("--pdf", default=None,
                        help="Target .pdf file (default: the .docx name with a .pdf suffix).")
    parser.add_argument("--quiet", action="store_true", help="Reduce the console output.")
    args = parser.parse_args(argv)

    docx_path = Path(args.docx)
    pdf_path = Path(args.pdf) if args.pdf else docx_path.with_suffix(".pdf")

    print("Converting the report to PDF through Microsoft Word ...")
    try:
        out = convert(docx_path, pdf_path, verbose=not args.quiet)
    except (WordUnavailable, FileNotFoundError) as exc:
        print("")
        print("CONVERSION FAILED")
        print("-" * 72)
        print(str(exc))
        print("-" * 72)
        return 2

    size_kb = out.stat().st_size / 1024
    ok = check_pdf(out)
    print(f"  wrote       : {out}")
    print(f"  size        : {size_kb:.1f} KB")
    print(f"  header check: {'OK (starts with %PDF)' if ok else 'FAILED'}")
    if not ok:
        print("  ! the produced file does not look like a valid PDF")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
