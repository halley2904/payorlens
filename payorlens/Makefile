# PayorLens Makefile
# Architecture v2.0

BENE_FILE   = DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv
CLAIMS_FILE = DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv
MODEL       = logistic

#  Setup
install:
	pip install -r requirements.txt

#  Data validation only 
validate:
	python cli.py validate \
		--bene-file $(BENE_FILE) \
		--claims-file $(CLAIMS_FILE)

#  Full evaluation pipeline 
evaluate:
	python cli.py evaluate \
		--bene-file $(BENE_FILE) \
		--claims-file $(CLAIMS_FILE) \
		--model $(MODEL)

#  Full evaluation with PDF
evaluate-pdf:
	python cli.py evaluate \
		--bene-file $(BENE_FILE) \
		--claims-file $(CLAIMS_FILE) \
		--model $(MODEL) \
		--pdf

#  Run on existing parquet (fast re-run, skip data processing) 
report:
	python cli.py evaluate \
		--parquet data/processed/claims_v1.parquet \
		--model $(MODEL) \
		--pdf

# Run GBM model 
gbm:
	python cli.py evaluate \
		--parquet data/processed/claims_v1.parquet \
		--model gbm \
		--pdf

#  Quick test on 5000 sample rows 
	python cli.py evaluate \
		--parquet data/processed/claims_v1.parquet \
		--model logistic \
		--samples 5000

.PHONY: install validate evaluate evaluate-pdf report gbm test