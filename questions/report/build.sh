#!/usr/bin/env bash
cd "$(dirname "$0")" && latexmk -pdf -interaction=nonstopmode report.tex
