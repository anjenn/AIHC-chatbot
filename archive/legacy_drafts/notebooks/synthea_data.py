!pip install -q openai pandas numpy scikit-learn tqdm

!pip install -q -U pip setuptools wheel
!pip install -q spacy==3.7.5 scispacy==0.5.4
!pip install -q https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz


patients_df = pd.read_csv(os.path.join(SYNTHEA_DIR, "patients.csv"))
conditions_df = pd.read_csv(os.path.join(SYNTHEA_DIR, "conditions.csv"))

print("patients:", patients_df.shape)
print("conditions:", conditions_df.shape)

display(patients_df.head(3))
display(conditions_df.head(3))

def normalize_diagnosis_surface(label):
    label = str(label).strip().lower()
    label = re.sub(r"\s+", " ", label)
    label = re.sub(r"\s*\((finding|disorder)\)\s*$", "", label)
    return label

# age
patients_df["BIRTHDATE"] = pd.to_datetime(patients_df["BIRTHDATE"], errors="coerce")
patients_df["age"] = 2020 - patients_df["BIRTHDATE"].dt.year

patient_base = patients_df[["Id", "age", "GENDER"]].copy()
patient_base.columns = ["PATIENT", "age", "sex"]

# clean conditions
conditions_df["DESCRIPTION"] = conditions_df["DESCRIPTION"].fillna("").astype(str).str.strip()
conditions_df = conditions_df[conditions_df["DESCRIPTION"] != ""].copy()
conditions_df["DESCRIPTION_NORM"] = conditions_df["DESCRIPTION"].apply(normalize_diagnosis_surface)

# labels that originated as "(finding)"
finding_like_norm_labels = set(
    conditions_df.loc[
        conditions_df["DESCRIPTION"].str.contains(r"\(finding\)\s*$", case=False, na=False),
        "DESCRIPTION_NORM"
    ].unique()
)

# most frequent normalized label per patient = target diagnosis
diagnosis_df = (
    conditions_df.groupby(["PATIENT", "DESCRIPTION_NORM"])
    .size()
    .reset_index(name="count")
    .sort_values(["PATIENT", "count", "DESCRIPTION_NORM"], ascending=[True, False, True])
    .drop_duplicates(subset=["PATIENT"])
    [["PATIENT", "DESCRIPTION_NORM"]]
    .rename(columns={"DESCRIPTION_NORM": "diagnosis"})
)

# remove target diagnosis from symptom text
conditions_with_target = conditions_df.merge(diagnosis_df, on="PATIENT", how="left")
symptoms_only_df = conditions_with_target[
    conditions_with_target["DESCRIPTION_NORM"] != conditions_with_target["diagnosis"]
].copy()

symptoms_df = (
    symptoms_only_df.groupby("PATIENT")["DESCRIPTION"]
    .apply(lambda x: ", ".join(sorted(set(x))) if len(x) > 0 else "no additional findings listed")
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

print(df.shape)
display(df.head())

TOP_N_LABELS = 4

top_labels = df["diagnosis"].value_counts().nlargest(TOP_N_LABELS).index.tolist()
df = df[df["diagnosis"].isin(top_labels)].reset_index(drop=True)

LABEL_SPACE = sorted(df["diagnosis"].unique().tolist())

print("Label distribution:")
print(df["diagnosis"].value_counts())
print("LABEL_SPACE:", LABEL_SPACE)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df["diagnosis"]
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Train:", train_df.shape)
print("Test:", test_df.shape)