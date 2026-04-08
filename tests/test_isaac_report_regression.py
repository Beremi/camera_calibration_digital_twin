"""Compile regression for the Isaac paper template when TeX tools are present."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


LATEXMK = shutil.which("latexmk")
PDFLATEX = shutil.which("pdflatex")
BIBTEX = shutil.which("bibtex")


pytestmark = pytest.mark.skipif(
    LATEXMK is None and (PDFLATEX is None or BIBTEX is None),
    reason="LaTeX toolchain is not available in this environment.",
)


def test_publication_report_template_compiles_from_source(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "report_tex"
    build_summary_path = repo_root / "output" / "isaac_runs" / "latest_first_pass_suite" / "analysis" / "publication_build_summary.json"
    target_dir = tmp_path / "report_tex"
    shutil.copytree(source_dir, target_dir)
    tex_path = target_dir / "publication_report_template.tex"
    tex_source = tex_path.read_text(encoding="utf-8")

    assert r"\TBD" not in tex_source
    assert "latest_first_pass_suite" in tex_source
    assert "latest_complete" in tex_source
    assert "latest_any" in tex_source
    assert "IsaacFirstPassMatrixRows" in tex_source
    assert "ik_failure_timeline.png" in tex_source
    assert "system_architecture.png" in tex_source
    assert "current workspace does not include the Isaac Sim Python modules" not in tex_source
    if build_summary_path.exists():
        build_summary = json.loads(build_summary_path.read_text(encoding="utf-8"))
        assert build_summary["artifact_source"] == "latest_first_pass_suite"

    for generated_name in (
        "publication_report_template.aux",
        "publication_report_template.bbl",
        "publication_report_template.blg",
        "publication_report_template.fdb_latexmk",
        "publication_report_template.fls",
        "publication_report_template.log",
        "publication_report_template.out",
        "publication_report_template.pdf",
    ):
        generated_path = target_dir / generated_name
        if generated_path.exists():
            generated_path.unlink()

    if LATEXMK is not None:
        subprocess.run(
            [LATEXMK, "-g", "-pdf", tex_path.name],
            cwd=str(target_dir),
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        subprocess.run([PDFLATEX, tex_path.name], cwd=str(target_dir), check=True, capture_output=True, text=True)
        subprocess.run([BIBTEX, tex_path.stem], cwd=str(target_dir), check=True, capture_output=True, text=True)
        subprocess.run([PDFLATEX, tex_path.name], cwd=str(target_dir), check=True, capture_output=True, text=True)
        subprocess.run([PDFLATEX, tex_path.name], cwd=str(target_dir), check=True, capture_output=True, text=True)

    assert (target_dir / "publication_report_template.pdf").exists()
