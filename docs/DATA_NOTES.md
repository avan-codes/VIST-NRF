# Data Notes — VIST-NRF

## Primary dataset
PhysioNet/CinC Challenge 2019 — Early Prediction of Sepsis from Clinical Data.
License: Creative Commons Attribution 4.0 International (CC BY 4.0).
Official: https://physionet.org/content/challenge-2019/1.0.0/
Challenge docs: https://archive.physionet.org/challenge/2019/
AWS Open Data mirror: https://registry.opendata.aws/challenge-2019/

## Download
Status: NOT downloaded. Download requires a PhysioNet credential; do not commit credentials.
Command reference (specification section 4.1):
  python -m pip install awscli
  aws s3 ls --no-sign-request s3://physionet-open/challenge-2019/1.0.0/training/
  mkdir -p "$VIST_NRF_DATA/raw/physionet2019"
  aws s3 sync --no-sign-request s3://physionet-open/challenge-2019/1.0.0/training/ "$VIST_NRF_DATA/raw/physionet2019/"

## Acceptance checks (to record after download)
- Count .psv files recursively in A and B; verify vs 20,336 (A) and 20,000 (B).
- Read headers from actual files; verify columns and consistent schema.
- Malformed/empty files, row counts, duplicate IDs, missingness, measurement frequency, sepsis-positive per environment.
- Record disk size, file counts, columns, missingness, label prevalence, license, citation, source, access date.

## Label convention (specification section 5)
- SepsisLabel becomes 1 at t_sepsis - 6 hours and remains 1.
- Reconstructed operational onset: t_event = t_first_positive + 6 hours.
- Do not define target as "next positive SepsisLabel within H hours."

## Citation
Reyna et al., Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019: https://pmc.ncbi.nlm.nih.gov/articles/PMC6964870/

## Status
UNVERIFIED — no data downloaded or audited yet. All counts/schema above are from the specification and must be re-verified against actual files.
