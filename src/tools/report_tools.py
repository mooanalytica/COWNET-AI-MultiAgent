from typing import Optional, Tuple, Dict, Any
import os
import re
from datetime import datetime

from langchain.tools import tool

# ReportLab for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def _sanitize_filename(name: str) -> str:
	"""Sanitize a filename to avoid invalid characters on Windows."""
	name = name.strip()
	name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
	return name or "report.pdf"


def _markdown_to_flowables(markdown: str) -> list:
	"""Convert a small subset of Markdown into ReportLab flowables.

	Supported:
	- # Heading 1, ## Heading 2, ### Heading 3
	- Bullets starting with "- " or "* "
	- Numbered lines like "1. Item"
	- Regular paragraphs
	"""
	styles = getSampleStyleSheet()

	heading1 = ParagraphStyle(
		"Heading1Custom",
		parent=styles["Heading1"],
		fontSize=18,
		spaceAfter=12,
	)
	heading2 = ParagraphStyle(
		"Heading2Custom",
		parent=styles["Heading2"],
		fontSize=16,
		spaceAfter=8,
	)
	heading3 = ParagraphStyle(
		"Heading3Custom",
		parent=styles["Heading3"],
		fontSize=14,
		spaceAfter=6,
	)

	bullet = styles["Bullet"]
	normal = styles["Normal"]

	flow = []
	for raw_line in markdown.splitlines():
		line = raw_line.strip()
		if not line:
			continue

		if line.startswith("# "):
			flow.append(Paragraph(line[2:], heading1))
		elif line.startswith("## "):
			flow.append(Paragraph(line[3:], heading2))
		elif line.startswith("### "):
			flow.append(Paragraph(line[4:], heading3))
		elif re.match(r"^[-*]\s+", line):
			flow.append(Paragraph(line[2:], bullet))
		elif re.match(r"^\d+\.\s+", line):
			# Simple numbered item rendered as normal paragraph
			flow.append(Paragraph(line, normal))
		else:
			flow.append(Paragraph(line, normal))

		flow.append(Spacer(1, 6))

	return flow


@tool("markdown_to_pdf", response_format="content_and_artifact")
def markdown_to_pdf(markdown: str, filename: Optional[str] = None, title: Optional[str] = "CowNet Report") -> Tuple[str, Dict[str, Any]]:
	"""
	Convert Markdown-like text to a PDF and save it to disk.

	Args:
		markdown: The Markdown content to render.
		filename: Optional filename (e.g., "cownet_report.pdf"). If not provided, a timestamped name is generated.
		title: Optional document title to place at the top.

	Returns:
		A tuple of (content, artifact) suitable for LangChain tools with response_format="content_and_artifact":
		- content: Short summary string of the operation
		- artifact: {"path": full_file_path, "filename": basename}

	Notes:
		- Creates a "reports" directory under the current working directory if it does not exist.
		- Uses ReportLab; ensure `reportlab` is installed.
	"""
	reports_dir = os.path.join(os.getcwd(), "reports")
	os.makedirs(reports_dir, exist_ok=True)

	if not filename:
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		filename = f"cownet_report_{timestamp}.pdf"

	filename = _sanitize_filename(filename)
	filepath = os.path.join(reports_dir, filename)

	# Build the PDF
	doc = SimpleDocTemplate(
		filepath,
		pagesize=letter,
		rightMargin=0.75 * inch,
		leftMargin=0.75 * inch,
		topMargin=0.75 * inch,
		bottomMargin=0.75 * inch,
	)

	styles = getSampleStyleSheet()
	title_style = ParagraphStyle(
		"DocTitle",
		parent=styles["Heading1"],
		fontSize=18,
		spaceAfter=12,
		alignment=1,
	)

	story = []
	if title:
		story.append(Paragraph(title, title_style))
		story.append(Paragraph(datetime.now().strftime("Generated %B %d, %Y %I:%M %p"), styles["Normal"]))
		story.append(Spacer(1, 18))

	story.extend(_markdown_to_flowables(markdown))

	try:
		doc.build(story)
		content = f"PDF generated: {os.path.basename(filepath)} saved to {filepath}"
		artifact = {"path": filepath, "filename": os.path.basename(filepath)}
		return content, artifact
	except Exception as e:
		return f"Failed to generate PDF: {e}", {}

