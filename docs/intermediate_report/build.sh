#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v latexmk >/dev/null 2>&1; then
  exec latexmk -pdf -interaction=nonstopmode main.tex
fi

if command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode main.tex
  bibtex main
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
  exit 0
fi

echo "No TeX toolchain found. Install latexmk or pdflatex+bibtex to build docs/intermediate_report/main.tex." >&2
exit 1
