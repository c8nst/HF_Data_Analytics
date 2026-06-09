
from pathlib import Path

import pandas as pd
import re

_DATA_DIR = Path(__file__).resolve().parent

data = pd.read_excel(_DATA_DIR / "working.xlsx", sheet_name=None)

print(data.keys())

df_main = data["MAIN"]
df_labs = data["LABS"]
df_meds = data["MEDS"]
df_echo = data["ECHO"]

df_main["REHOSPITAL"] = df_main["REHOSPITAL"].notna()

def parse_clinical_text(text):
    if pd.isna(text):
        return []

    code_pattern = r'\(([A-Z][0-9]{1,2}(?:\.[0-9])?)\)'

    codes = re.findall(code_pattern, str(text))

    descriptions = re.split(code_pattern, str(text)) # Split text by the code pattern to get descriptions

    pairs = [] #we create code-desc pairs

    for i in range(1, len(descriptions), 2):
        code = descriptions[i].strip()
        desc = descriptions[i + 1].strip(" |").strip()
        pairs.append((code, desc))
    return pairs

def gender_normalisation(text):
    if text == "კაცი":
        return "male"
    return "female"

df_main["SEX"] = df_main["SEX"].apply(gender_normalisation)

cols = ["MAIN", "COMPLICATION", "FOLLOWING"]
for col in cols:
    df_main[f"{col}_PAIRS"] = df_main[col].apply(parse_clinical_text)


# 1. Create Lookup Table (Code -> Description)
all_pairs = [p for col in cols for row in df_main[f"{col}_PAIRS"] for p in row]
diag_lookup = pd.DataFrame(all_pairs, columns=["Code", "Description"]).drop_duplicates("Code")
# Sort the lookup table alphabetically/numerically by Code
diag_lookup = diag_lookup.sort_values(by="Code").reset_index(drop=True)
# 2. Simplify Main Table (Keep only codes for ML)
for col in cols:
    df_main[col] = df_main[f"{col}_PAIRS"].apply(lambda x: [p[0] for p in x])

df_main = df_main.drop(columns=[f"{col}_PAIRS" for col in cols])

with pd.ExcelWriter(_DATA_DIR / "Processed_HF_Project.xlsx") as writer:
    df_main.to_excel(writer, sheet_name="Main_Data", index=False)
    df_main[df_main["REHOSPITAL"] == True].to_excel(writer, sheet_name="Rehospital_True", index=False)
    df_main[df_main["REHOSPITAL"] == False].to_excel(writer, sheet_name="Rehospital_False", index=False)
    diag_lookup.to_excel(writer, sheet_name="Diagnosis_Lookup", index=False)

    # Add the remaining key data tables into the consolidated workbook.
    df_labs.to_excel(writer, sheet_name="Labs", index=False)
    df_meds.to_excel(writer, sheet_name="Meds", index=False)
    df_echo.to_excel(writer, sheet_name="Echo", index=False)
