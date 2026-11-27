"""Patient-level train/validation/test splits.

Specification section 6.1:
- Within each hospital system, split patients 70% train / 15% validation / 15% test,
  stratified by whether the patient ever has a sepsis label.
- Save the split once; never silently regenerate it.
- Cross-site: target-environment train/validation cannot be used for model selection,
  preprocessing, or threshold tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class PatientSplit:
    train: list[str]
    val: list[str]
    test: list[str]
    env: str
    stratify_ever_positive: list[int]
    seed: int

    def to_dict(self) -> dict:
        return {
            "env": self.env,
            "seed": self.seed,
            "train": sorted(self.train),
            "val": sorted(self.val),
            "test": sorted(self.test),
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
            "stratify_ever_positive": self.stratify_ever_positive,
        }

    def all_patients(self) -> list[str]:
        return self.train + self.val + self.test

    def is_disjoint(self) -> bool:
        sets = [set(self.train), set(self.val), set(self.test)]
        return (
            len(sets[0] & sets[1]) == 0
            and len(sets[0] & sets[2]) == 0
            and len(sets[1] & sets[2]) == 0
        )

    def covers_all(self, all_patients: list[str]) -> bool:
        return set(self.all_patients()) == set(all_patients)


def split_patients(
    patient_ids: list[str],
    ever_positive: dict[str, int],
    env: str,
    seed: int = 0,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> PatientSplit:
    """Stratified patient-level split. Raises if fractions don't sum to 1."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, "fractions must sum to 1"

    df = pd.DataFrame(
        {"pid": patient_ids, "ever_pos": [ever_positive.get(p, 0) for p in patient_ids]}
    )

    # First split: train vs temp (val+test), stratified by ever_positive when possible.
    stratify = df["ever_pos"] if df["ever_pos"].nunique() > 1 else None
    train_df, temp_df = train_test_split(
        df,
        train_size=train_frac,
        random_state=seed,
        stratify=stratify,
        shuffle=True,
    )

    # Split temp into val and test. Relative fraction of val within temp.
    val_within_temp = val_frac / (val_frac + test_frac)
    stratify_t = temp_df["ever_pos"] if temp_df["ever_pos"].nunique() > 1 else None
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_within_temp,
        random_state=seed,
        stratify=stratify_t,
        shuffle=True,
    )

    return PatientSplit(
        train=train_df["pid"].tolist(),
        val=val_df["pid"].tolist(),
        test=test_df["pid"].tolist(),
        env=env,
        stratify_ever_positive=sorted(df["ever_pos"].unique().tolist()),
        seed=seed,
    )


def split_by_environment(
    patients_by_env: dict[str, list[str]],
    ever_positive_by_env: dict[str, dict[str, int]],
    seed: int = 0,
) -> dict[str, PatientSplit]:
    """Build a PatientSplit per environment."""
    return {
        env: split_patients(
            patients_by_env[env],
            ever_positive_by_env[env],
            env=env,
            seed=seed,
        )
        for env in patients_by_env
    }


def assert_splits_valid(splits: dict[str, PatientSplit]) -> None:
    """Automated leakage checks: disjointness and full coverage per env."""
    for env, sp in splits.items():
        if not sp.is_disjoint():
            raise AssertionError(f"patient overlap across splits in env {env}")
        # coverage checked by caller against the full patient list
    # cross-env disjointness too (a patient should belong to one env's split only)
    seen: set[str] = set()
    for env, sp in splits.items():
        for p in sp.all_patients():
            if p in seen:
                raise AssertionError(f"patient {p} appears in multiple environments")
            seen.add(p)
