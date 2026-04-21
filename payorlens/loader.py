# payorlens/loader.py
"""
PayorLens Data Ingestion & Validation Module
"""

import os
import logging
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("PayorLens.Loader")

# ── CMS code mappings ────────────────────────────────────────────────────────
RACE_MAP = {1: "White", 2: "Black", 3: "Other", 5: "Hispanic"}
SEX_MAP  = {1: "Male",  2: "Female"}

STATE_MAP = {
    1:"AL",2:"AK",3:"AZ",4:"AR",5:"CA",6:"CO",7:"CT",8:"DE",9:"DC",10:"FL",
    11:"GA",12:"HI",13:"ID",14:"IL",15:"IN",16:"IA",17:"KS",18:"KY",19:"LA",20:"ME",
    21:"MD",22:"MA",23:"MI",24:"MN",25:"MS",26:"MO",27:"MT",28:"NE",29:"NV",30:"NH",
    31:"NJ",32:"NM",33:"NY",34:"NC",35:"ND",36:"OH",37:"OK",38:"OR",39:"PA",40:"RI",
    41:"SC",42:"SD",43:"TN",44:"TX",45:"UT",46:"VT",47:"VA",48:"WA",49:"WV",50:"WI",
    51:"WY",52:"PR"
}


# ── Pydantic schema ───────────────────────────────────────────────────────────
class ClaimRecord(BaseModel):
    """
    Validated data contract for one inpatient claim record.
    All raw CMS integer/float IDs are coerced to str BEFORE Pydantic sees them
    in the normalisation step, so no type errors occur.
    """
    # Identifiers
    beneficiary_id : str
    claim_id       : str

    # Demographics (decoded from CMS integer codes)
    gender         : str          # "Male" | "Female" | "Unknown"
    race           : str          # "White" | "Black" | "Other" | "Hispanic" | "Unknown"
    age_band       : str          # "18-34" | "35-49" | "50-64" | "65+"
    state_code     : str          # 2-letter abbreviation, e.g. "CA"
    age            : int

    # Clinical
    icd9_primary   : Optional[str] = None
    drg_code       : Optional[str] = None
    utilization_days: Optional[int] = None
    has_diabetes   : int           # 1 = yes, 0 = no
    has_chf        : int
    has_copd       : int
    has_cancer     : int
    chronic_count  : int           # total chronic conditions flagged

    # Financial
    claim_amount     : float
    deductible_amount: float
    claim_year       : int

    # Target
    denial_status : int            # 0 = approved, 1 = denied

    @field_validator("denial_status")
    @classmethod
    def binary_only(cls, v):
        if v not in (0, 1):
            raise ValueError(f"denial_status must be 0 or 1, got {v}")
        return v

    @field_validator("claim_amount")
    @classmethod
    def non_negative_amount(cls, v):
        if v < 0:
            raise ValueError(f"claim_amount must be >= 0, got {v}")
        return v

    @field_validator("age")
    @classmethod
    def plausible_age(cls, v):
        if not (0 <= v <= 115):
            raise ValueError(f"Age {v} is implausible")
        return v


# ── Normalisation helpers ─────────────────────────────────────────────────────
def _derive_age_band(age: int) -> str:
    if age < 35:   return "18-34"
    if age < 50:   return "35-49"
    if age < 65:   return "50-64"
    return "65+"


def _flag(val, true_val=1) -> int:
    """CMS chronic condition flags: 1=has condition, 2=no condition."""
    return 1 if val == true_val else 0


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw merged CMS DataFrame into the ClaimRecord-compatible schema.
    All coercions happen here so Pydantic only does type-validation, not parsing.
    """
    out = pd.DataFrame()

    # ── Identifiers (coerce ints → str) ──────────────────────────────────────
    out["beneficiary_id"] = df["DESYNPUF_ID"].astype(str)
    out["claim_id"]       = df["CLM_ID"].astype(str)

    # ── Demographics ──────────────────────────────────────────────────────────
    out["gender"]     = df["BENE_SEX_IDENT_CD"].map(SEX_MAP).fillna("Unknown")
    out["race"]       = df["BENE_RACE_CD"].map(RACE_MAP).fillna("Unknown")
    out["state_code"] = df["SP_STATE_CODE"].map(STATE_MAP).fillna("Unknown")

    # Age from birth date (format YYYYMMDD stored as int)
    birth_year = (pd.to_numeric(df["BENE_BIRTH_DT"], errors="coerce") // 10000).fillna(1940).astype(int)
    claim_year = (pd.to_numeric(df["CLM_FROM_DT"],   errors="coerce") // 10000).fillna(2008).astype(int)
    out["age"]       = (claim_year - birth_year).clip(lower=0, upper=115)
    out["age_band"]  = out["age"].apply(_derive_age_band)
    out["claim_year"] = claim_year

    # ── Clinical ──────────────────────────────────────────────────────────────
    out["icd9_primary"]    = df["ADMTNG_ICD9_DGNS_CD"].where(df["ADMTNG_ICD9_DGNS_CD"].notna(), None)
    out["drg_code"]        = df["CLM_DRG_CD"].astype(str).where(df["CLM_DRG_CD"].notna(), None)
    out["utilization_days"]= pd.to_numeric(df["CLM_UTLZTN_DAY_CNT"], errors="coerce").fillna(0).astype(int)

    # Chronic conditions (1 = has condition in CMS encoding)
    out["has_diabetes"] = df["SP_DIABETES"].apply(_flag)
    out["has_chf"]      = df["SP_CHF"].apply(_flag)
    out["has_copd"]     = df["SP_COPD"].apply(_flag)
    out["has_cancer"]   = df["SP_CNCR"].apply(_flag)

    chronic_cols = ["SP_ALZHDMTA","SP_CHF","SP_CHRNKIDN","SP_CNCR","SP_COPD",
                    "SP_DEPRESSN","SP_DIABETES","SP_ISCHMCHT","SP_OSTEOPRS","SP_RA_OA","SP_STRKETIA"]
    out["chronic_count"] = df[chronic_cols].apply(lambda row: sum(v == 1 for v in row), axis=1)

    # ── Financial ─────────────────────────────────────────────────────────────
    clm_pmt = pd.to_numeric(df["CLM_PMT_AMT"], errors="coerce").fillna(0.0)
    out["claim_amount"]      = clm_pmt
    out["deductible_amount"] = pd.to_numeric(df["NCH_BENE_IP_DDCTBL_AMT"], errors="coerce").fillna(0.0)

    # ── Target variable: ENGINEERED realistic prior-auth denial proxy ─────────
    # Intercept tuned to -2.2 → produces ~21% denial rate on CMS DE-SynPUF.
    # -1.8 (previous) produced 26% → after random.seed shift → 99.9% in some
    # environments. -2.2 verified empirically on this exact dataset.
    # Random Bernoulli draw used — NOT prob>0.5 threshold (threshold fails when
    # mean logit >> 0, which happens with mean age=73.7 and mean chronic=4.94).
    np.random.seed(42)
    _logit = (
        -2.2
        + 0.025 * out["utilization_days"].clip(0, 60)
        + 0.000008 * clm_pmt.clip(0, 200_000)
        - 0.012  * out["age"]
        + 0.35   * (out["race"] == "Black").astype(float)
        + 0.25   * (out["race"] == "Hispanic").astype(float)
        + 0.12   * (out["gender"] == "Female").astype(float)
        + 0.30   * (out["age"] < 50).astype(float)
        + 0.20   * out["chronic_count"].clip(0, 11)
        + np.random.normal(0, 1.2, len(out))
    )
    _prob = 1.0 / (1.0 + np.exp(-_logit))
    out["denial_status"] = (np.random.uniform(0, 1, len(out)) < _prob).astype(int)

    return out


# ── Main Loader ───────────────────────────────────────────────────────────────
class CMSLoader:
    """
    Joins CMS DE-SynPUF beneficiary + inpatient claims, normalises,
    validates via Pydantic, and saves clean parquet for downstream modules.
    """

    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir        = raw_dir
        self.processed_dir  = processed_dir
        self.validation_errors: list = []

    # ── public API ────────────────────────────────────────────────────────────
    def process(self, beneficiary_file: str, claims_file: str) -> pd.DataFrame:
        df_bene   = self._read(beneficiary_file)
        df_claims = self._read(claims_file)

        logger.info("Joining on DESYNPUF_ID …")
        df_merged = pd.merge(df_claims, df_bene, on="DESYNPUF_ID", how="inner")
        logger.info(f"Merged → {len(df_merged):,} rows")

        logger.info("Normalising …")
        df_norm = normalize(df_merged)

        logger.info("Validating with Pydantic …")
        valid_records = self._validate(df_norm)

        df_final = pd.DataFrame(valid_records)
        self._save(df_final)
        self._report(df_final)
        return df_final

    # ── internals ─────────────────────────────────────────────────────────────
    def _read(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.raw_dir, filename)
        logger.info(f"Reading {path}")
        return pd.read_csv(path, low_memory=False)

    def _validate(self, df: pd.DataFrame) -> list:
        records, errors = [], []
        for idx, row in enumerate(df.to_dict("records")):
            try:
                rec = ClaimRecord(**row)
                records.append(rec.model_dump())
            except ValidationError as e:
                errors.append({"row": idx, "error": str(e)})
        self.validation_errors = errors
        logger.info(f"Valid: {len(records):,}  |  Errors: {len(errors):,}")
        return records

    def _save(self, df: pd.DataFrame):
        os.makedirs(self.processed_dir, exist_ok=True)
        if df.empty:
            logger.warning("No valid records — parquet not saved.")
            return
        path = os.path.join(self.processed_dir, "claims_v1.parquet")
        df.to_parquet(path, index=False)
        logger.info(f"Saved → {path}")

    def _report(self, df: pd.DataFrame):
        if df.empty:
            return
        print("\n" + "="*60)
        print("  PayorLens  ·  Day 1 & 2 Data Foundation Report")
        print("="*60)
        print(f"  Total valid records   : {len(df):,}")
        print(f"  Validation errors     : {len(self.validation_errors):,}  "
              f"({len(self.validation_errors)/max(len(df)+len(self.validation_errors),1)*100:.1f}%)")
        print(f"  Denial rate (proxy)   : {df['denial_status'].mean()*100:.1f}%")
        print(f"  Age range             : {df['age'].min()}–{df['age'].max()}  "
              f"(mean {df['age'].mean():.1f})")
        print(f"  Gender split          : {df['gender'].value_counts().to_dict()}")
        print(f"  Race distribution     : {df['race'].value_counts().to_dict()}")
        print(f"  Age band distribution : {df['age_band'].value_counts().to_dict()}")
        print(f"  Avg claim amount      : ${df['claim_amount'].mean():,.0f}")
        print(f"  Chronic cond (mean)   : {df['chronic_count'].mean():.2f}")
        print("="*60 + "\n")


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RAW_DIR       = "D:/payorlens/data/raw/cms"
    PROCESSED_DIR = "D:/payorlens/data/processed"

    BENE_FILE   = "DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv"
    CLAIMS_FILE = "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv"

    loader = CMSLoader(RAW_DIR, PROCESSED_DIR)
    df = loader.process(BENE_FILE, CLAIMS_FILE)

    if loader.validation_errors:
        print(f"\nSample validation error:\n{loader.validation_errors[0]}")