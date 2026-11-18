# AI651 Assignment 3: Spatio-Temporal Traffic Forecasting (LargeST-SD)

## Setup
pip install -r requirements.txt

## Data
Download LargeST from Kaggle and place the SD files in data/raw/.
python src/data.py          # resample, split, normalise -> data/processed/

## Run experiments (seed 42)
python src/train.py --model temporal
python src/train.py --model fixed
python src/train.py --model adaptive
python src/train.py --model hybrid

## Figures and report
python src/plots.py         # writes PDFs to report/figures/
cd report && latexmk -pdf report.tex
