# Dataset preparation utilities for patient-level Synthea records.

from __future__ import annotations

import re

import pandas as pd
from src.config import SEED


def normalize_diagnosis_surface(label: object) -> str:
    # Normalize diagnosis labels for stable matching and frequency counts.
    label = str(label).strip().lower()
    label = re.sub(r"\s+", " ", label)
    label = re.sub(r"\s*\((finding|disorder)\)\s*$", "", label)
    return label


def symptom_mentions_label(symptom: object, label_space: list[str]) -> bool:
    # Return True when a symptom item directly mentions a diagnosis label.
    symptom_norm = normalize_diagnosis_surface(symptom)

    for label in label_space:
        label_norm = normalize_diagnosis_surface(label)
        if label_norm and label_norm in symptom_norm:
            return True

    return False


def strip_diagnosis_mentions_from_symptoms(
    symptoms: object,
    label_space: list[str],
    fallback_text: str = "no non-diagnostic findings listed",
) -> str:
    # Remove symptom entries that explicitly name one of the diagnosis labels.
    parts = [part.strip() for part in str(symptoms).split(",") if part.strip()]
    cleaned_parts = [
        part for part in parts if not symptom_mentions_label(part, label_space)
    ]

    if not cleaned_parts:
        return fallback_text

    return ", ".join(cleaned_parts)


def remove_leaky_symptoms(df: pd.DataFrame, label_space: list[str]) -> pd.DataFrame:
    # Remove diagnosis-like symptom mentions after the label space is known.
    out = df.copy()
    out["symptoms"] = out["symptoms"].apply(
        lambda symptoms: strip_diagnosis_mentions_from_symptoms(symptoms, label_space)
    )
    return out


def rebalance_top_labels(
    df: pd.DataFrame,
    random_state: int = SEED,
) -> pd.DataFrame:
    # Downsample the runaway majority class to the second-largest class size.
    counts = df["diagnosis"].value_counts()
    if len(counts) < 2:
        return df.reset_index(drop=True)

    target_max = int(counts.iloc[1])
    balanced_parts: list[pd.DataFrame] = []

    for diagnosis, group in df.groupby("diagnosis", sort=False):
        if len(group) > target_max:
            balanced_parts.append(group.sample(n=target_max, random_state=random_state))
        else:
            balanced_parts.append(group)

    out = pd.concat(balanced_parts, ignore_index=True)
    return out.sample(frac=1, random_state=random_state).reset_index(drop=True)


def build_patient_dataset(
    patients_df: pd.DataFrame,
    conditions_df: pd.DataFrame,
) -> pd.DataFrame:
    # Build one patient-level row with demographics, symptoms, and diagnosis.
    patients_df = patients_df.copy()
    conditions_df = conditions_df.copy()

    patients_df["BIRTHDATE"] = pd.to_datetime(
        patients_df["BIRTHDATE"], errors="coerce"
    )
    patients_df["age"] = 2020 - patients_df["BIRTHDATE"].dt.year

    patient_base = patients_df[["Id", "age", "GENDER"]].copy()
    patient_base.columns = ["PATIENT", "age", "sex"]

    conditions_df["DESCRIPTION"] = (
        conditions_df["DESCRIPTION"].fillna("").astype(str).str.strip()
    )
    conditions_df = conditions_df[conditions_df["DESCRIPTION"] != ""].copy()
    conditions_df["DESCRIPTION_NORM"] = conditions_df["DESCRIPTION"].apply(
        normalize_diagnosis_surface
    )

    finding_like_norm_labels = set(
        conditions_df.loc[
            conditions_df["DESCRIPTION"].str.contains(
                r"\(finding\)\s*$", case=False, na=False
            ),
            "DESCRIPTION_NORM",
        ].unique()
    )

    diagnosis_df = (
        conditions_df.groupby(["PATIENT", "DESCRIPTION_NORM"])
        .size()
        .reset_index(name="count")
        .sort_values(
            ["PATIENT", "count", "DESCRIPTION_NORM"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["PATIENT"])
        [["PATIENT", "DESCRIPTION_NORM"]]
        .rename(columns={"DESCRIPTION_NORM": "diagnosis"})
    )

    conditions_with_target = conditions_df.merge(diagnosis_df, on="PATIENT", how="left")
    symptoms_only_df = conditions_with_target[
        conditions_with_target["DESCRIPTION_NORM"] != conditions_with_target["diagnosis"]
    ].copy()

    symptoms_df = (
        symptoms_only_df.groupby("PATIENT")["DESCRIPTION"]
        .apply(
            lambda values: ", ".join(sorted(set(values)))
            if len(values) > 0
            else "no additional findings listed"
        )
        .reset_index()
    )
    symptoms_df.columns = ["PATIENT", "symptoms"]

    df = patient_base.merge(symptoms_df, on="PATIENT", how="inner")
    df = df.merge(diagnosis_df, on="PATIENT", how="inner")

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["sex"] = df["sex"].fillna("unknown").astype(str).str.strip()
    df["symptoms"] = df["symptoms"].fillna("").astype(str).str.strip()
    df["diagnosis"] = df["diagnosis"].fillna("").astype(str).str.strip()

    df = df.dropna(subset=["age"])
    df = df[df["symptoms"] != ""]
    df = df[df["diagnosis"] != ""]
    df = df[~df["diagnosis"].isin(finding_like_norm_labels)]
    df = df.reset_index(drop=True)

    return df


def filter_top_labels(
    df: pd.DataFrame,
    top_n: int = 8,
    rebalance: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    # Keep the most frequent labels, remove leakage, and optionally rebalance.
    top_labels = df["diagnosis"].value_counts().nlargest(top_n).index.tolist()
    out = df[df["diagnosis"].isin(top_labels)].reset_index(drop=True)
    out = remove_leaky_symptoms(out, top_labels)

    if rebalance:
        out = rebalance_top_labels(out, random_state=SEED)

    label_space = sorted(out["diagnosis"].unique().tolist())
    return out, label_space
