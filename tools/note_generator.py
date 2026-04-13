import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone

from langchain_core.tools import tool

logger = logging.getLogger("agent_interview_prep.tools.note_generator")

_BASE_URL = (os.getenv("BACKEND_URL") or os.getenv("PUBLIC_URL") or "").rstrip("/")


_UNICODE_TO_ASCII = str.maketrans({
    # Greek letters (lowercase)
    "α": "alpha", "β": "beta",  "γ": "gamma", "δ": "delta",
    "ε": "epsilon","ζ": "zeta", "η": "eta",   "θ": "theta",
    "λ": "lambda", "μ": "mu",   "ξ": "xi",   "π": "pi",
    "σ": "sigma",  "τ": "tau",  "φ": "phi",  "χ": "chi",
    "ψ": "psi",    "ω": "omega",
    # Greek letters (uppercase)
    "Γ": "Gamma", "Δ": "Delta", "Θ": "Theta", "Λ": "Lambda",
    "Σ": "Sigma",  "Φ": "Phi",  "Ψ": "Psi",  "Ω": "Omega",
    # Math operators / symbols
    "∑": "sum",   "∏": "prod",  "∫": "integral",
    "∈": "in",    "∉": "not in","⊂": "subset","⊆": "subset=",
    "∪": "union", "∩": "intersect",
    "≤": "<=",    "≥": ">=",   "≠": "!=",
    "→": "->",    "←": "<-",   "↔": "<->",
    "∞": "inf",   "∂": "d",    "∇": "nabla",
    "⊤": "^T",    "⊥": "perp",
    "·": ".",
})


def _strip_latex_math(text: str) -> str:
    """Remove LaTeX $$...$$ and $...$ delimiters, keeping the inner expression."""
    text = re.sub(r'\$\$(.*?)\$\$', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', r'\1', text)
    return text


def _sanitize_for_pdf(text: str) -> str:
    """Transliterate Unicode math symbols to ASCII equivalents for Helvetica compatibility."""
    return text.translate(_UNICODE_TO_ASCII)


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:50]


def _generate_toc(content: str) -> str:
    """Generate a table of contents from markdown headings."""
    toc_lines = []
    for line in content.split("\n"):
        if line.startswith("## "):
            heading = line[3:].strip()
            anchor = _slugify(heading)
            toc_lines.append(f"- [{heading}](#{anchor})")
        elif line.startswith("### "):
            heading = line[4:].strip()
            anchor = _slugify(heading)
            toc_lines.append(f"  - [{heading}](#{anchor})")
    return "\n".join(toc_lines)


def _create_pdf_bytes(title: str, markdown_content: str) -> bytes:
    """Generate a PDF from markdown content using fpdf2. Returns bytes."""
    from fpdf import FPDF

    # Preprocess: strip LaTeX math delimiters and transliterate Unicode
    # symbols to ASCII so Helvetica can render them without errors.
    markdown_content = _sanitize_for_pdf(_strip_latex_math(markdown_content))
    title = _sanitize_for_pdf(title)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)

    # Content - simple line-by-line rendering
    pdf.set_font("Helvetica", size=11)
    for line in markdown_content.split("\n"):
        stripped = line.strip()

        if stripped.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            pdf.ln(4)
            pdf.cell(0, 10, stripped[2:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=11)
        elif stripped.startswith("## "):
            pdf.set_font("Helvetica", "B", 14)
            pdf.ln(3)
            pdf.cell(0, 9, stripped[3:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=11)
        elif stripped.startswith("### "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.ln(2)
            pdf.cell(0, 8, stripped[4:], new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=11)
        elif stripped.startswith("- ") or stripped.startswith("* "):
            bullet_text = stripped[2:]
            bullet_text = re.sub(r"\*\*(.*?)\*\*", r"\1", bullet_text)
            pdf.cell(8)
            pdf.multi_cell(0, 6, f"• {bullet_text}")
        elif stripped:
            plain = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped)
            plain = re.sub(r"\*(.*?)\*", r"\1", plain)
            pdf.multi_cell(0, 6, plain)
        else:
            pdf.ln(3)

    return pdf.output()


@tool
async def generate_study_notes(title: str, content: str, format: str = "markdown", source_file_id: str | None = None) -> str:
    """Generate downloadable study notes from the provided content.

    Args:
        title: The title of the study notes (e.g., "System Design Interview Notes").
        content: The full markdown content of the study notes. Can be empty if source_file_id is provided.
        format: Output format - "markdown" or "pdf". Defaults to "markdown".
        source_file_id: Optional ID of a previously generated markdown file to use as the source content.
    """
    from database.mongo import MongoDB

    file_id = uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    if source_file_id:
        result = await MongoDB.retrieve_file(source_file_id)
        if not result:
            return f"Error: source file {source_file_id} not found."
        data, meta = result
        try:
            retrieved_content = data.decode("utf-8")
        except Exception:
            return f"Error: source file is not valid text/markdown."
        title = title or "Study_Notes"
        slug = _slugify(title)
        full_content = retrieved_content
        pdf_content = retrieved_content
    else:
        slug = _slugify(title)
        # Add TOC to markdown content
        toc = _generate_toc(content)
        full_content = f"# {title}\n\n## Table of Contents\n{toc}\n\n---\n\n{content}"
        pdf_content = content

    # Determine filename
    if format == "pdf":
        filename = f"{timestamp}_{slug}.pdf"
    else:
        filename = f"{timestamp}_{slug}.md"

    try:
        if format == "pdf":
            file_bytes = _create_pdf_bytes(title, pdf_content)
        else:
            file_bytes = full_content.encode("utf-8")

        # Store in GridFS (persists across Railway deploys)
        await MongoDB.store_file(
            file_id=file_id,
            filename=filename,
            data=file_bytes,
            file_type="notes",
        )

        logger.info("Generated study notes: file_id='%s', format='%s', size=%d bytes",
                     file_id, format, len(file_bytes))

        return (
            f"Study notes generated successfully!\n\n"
            f"**Title:** {title}\n"
            f"**Format:** {format.upper()}\n"
            f"**Download:** [Download: {title}]({_BASE_URL}/download/{file_id})"
        )

    except Exception as e:
        logger.error("Failed to generate study notes: %s", e)
        return (
            f"Error generating study notes ({format}): {e}. "
            "Do NOT retry with the same format. "
            "If format was 'pdf', call this tool again with format='markdown' instead — "
            "markdown supports all Unicode and math notation without restrictions."
        )
