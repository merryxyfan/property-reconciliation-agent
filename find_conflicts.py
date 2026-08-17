import re
import pandas as pd

# Load linked properties after data_preprocess.ipynb
linked = pd.read_csv("data/linked_properties.csv")

#- built_form   = whether/how the whole building shares walls with neighbors
#                 (Detached / Semi-Detached / Mid-Terrace / End-Terrace)
#- property_type = whether this transaction unit is a whole dwelling or a subdivided unit within that building 
#                 (House/Bungalow vs Flat/Maisonette)

# mapping from HMLR to EPC built_form
HMLR_TO_EPC_built_form = {
    "D": ["Detached"],
    "S": ["Semi-Detached"],
    "T": ["Mid-Terrace", "End-Terrace", "Enclosed Mid-Terrace", "Enclosed End-Terrace"],
}
# mapping from HMLR to EPC property_type
HMLR_TO_EPC_property_type = {
    "D": ["House", "Bungalow"],
    "S": ["House", "Bungalow"],
    "T": ["House", "Bungalow"],
    "F": ["Flat", "Maisonette"]
}

# Helper Function in identifying neighboring building in full EPC dataset - Conlict 1
def extract_house_number(address):
    if pd.isna(address):
        return None
    match = re.match(r"^\s*(\d+)", str(address))
    return int(match.group(1)) if match else None

def extract_street_name(address):
    if pd.isna(address):
        return None
    return re.sub(r"^\s*\d+[A-Za-z]?\s*", "", str(address)).strip().upper()

def get_neighbors_from_epc_full(address, epc_full_df, postcode, ranges=(2, 4, 6)):
    target_number = extract_house_number(address)
    if target_number is None:
        return epc_full_df.iloc[0:0] 
    target_street = extract_street_name(address)

    candidates = epc_full_df[
        (epc_full_df["postcode_clean"] == postcode) &
        (epc_full_df["street_name"] == target_street)
    ].copy()

    neighbor_numbers = {target_number - r for r in ranges} | {target_number + r for r in ranges}
    return candidates[candidates["house_number"].isin(neighbor_numbers)]

# Helper Function in Quantile-based scoring for property type - Conflict 1
def property_size_evidence(area, rooms):

    score_house = 0
    score_flat = 0

    # Floor area
    if area <= 40:
        score_flat += 2
    elif area <= 59:
        score_flat += 1
    elif area >= 86:
        score_house += 2
    elif area >= 71:
        score_house += 1

    # Habitable rooms
    if rooms <= 2:
        score_flat += 2
    elif rooms == 3:
        score_flat += 1
    elif rooms >= 5:
        score_house += 2
    elif rooms >= 4:
        score_house += 1

    if score_house > score_flat:
        return "House-like"
    elif score_flat > score_house:
        return "Flat-like"
    else:
        return "Ambiguous"

# Conflict 1: HMLR "Property Type" VS EPC "Property Type"
def check_property_type(hmlr_type, epc_property_type, address, epc_floor_level, area, rooms):

    if pd.isna(hmlr_type) or pd.isna(epc_property_type):
        return {
            "conflict": False,
            "winner": None,
            "reason": "Missing data"
        }

    if hmlr_type == "O":
        return {
            "conflict": False,
            "winner": None,
            "reason": "HMLR property type 'Other' cannot be mapped reliably"
        }

    expected_types = HMLR_TO_EPC_property_type.get(hmlr_type, [])

    # Consistent
    if epc_property_type in expected_types:
        return {
            "conflict": False,
            "winner": "consistent",
            "reason": (
                "HMLR property type is consistent with the EPC property type."
            )
        }

    # Rule 1:
    # If "FLAT" or letter suffix in the address or floor number is provided in EPC, the winner should be the one with "F" or "Flat/Manisonette"
    if hmlr_type in ("F") or epc_property_type in ("Flat", "Maisonette"):
        has_flat_keyword = bool(re.search(r"\bFLAT\b", address, re.IGNORECASE))
        letter_suffix_match = re.match(r"^\s*(\d+[A-Za-z])\b", address.strip()) if pd.notna(address) else None
        has_letter_suffix = bool(letter_suffix_match)
        has_floor_level = pd.notna(epc_floor_level)

        if has_flat_keyword or has_letter_suffix or has_floor_level:
            triggered = []
            if has_flat_keyword:
                triggered.append("Address text contains 'FLAT'")
            if has_letter_suffix:
                triggered.append(f"Address has a letter-suffixed house number (i.e., {letter_suffix_match.group(1)}), consistent with a subdivided property (maisonette-style)")
            if has_floor_level:
                triggered.append(f"EPC FLOOR_LEVEL is populated ('{epc_floor_level}'), which under RdSAP methodology only happens for Flat/Maisonette units")

            if hmlr_type in ("F"):
                return {
                    "conflict": True,
                    "winner": "HMLR",
                    "reason": (
                        "{evidence}. Therefore, this conflict is resolved by following HMLR for unit-level property type."
                    ).format(evidence="; ".join(triggered))
                }
            else:
                return {
                    "conflict": True,
                    "winner": "EPC",
                    "reason": (
                        "{evidence}. Therefore, this conflict is resolved by following EPC for unit-level property type."
                    ).format(evidence="; ".join(triggered))
                }
    # Rule 2:
    # Use quantile distributions of EPC total floor area and habitable room count as supporting evidence for broad property type (analysis shown in data_preprocess.ipynb).
    # Comparatively low values that fall within the typical lower range of flats/maisonettes and are uncommon for houses/bungalows provide evidence towards Flat/Maisonette;
    # Conversely, comparatively high values provide evidence towards House/Bungalow.
    if hmlr_type in ("F",) and epc_property_type in ("House", "Maisonette"):
        scored_result = property_size_evidence(area, rooms)
        if scored_result == "House-like":
            return {
                "conflict": True,
                "winner": "EPC",
                "reason": (
                    "After applying the quantile-based scoring rules based on floor area and number of habitable rooms, the result is {evidence}. Therefore, this conflict is resolved by following the EPC classification for the unit-level property type."
                ).format(evidence=scored_result)
            }
        elif scored_result == "Flat-like":
            return {
                "conflict": True,
                "winner": "HMLR",
                "reason": (
                    "After applying the quantile-based scoring rules based on floor area and number of habitable rooms, the result is {evidence}. Therefore, this conflict is resolved by following the HMLR classification for the unit-level property type."
                ).format(evidence=scored_result)
            }
    if hmlr_type in ("D", "S", "T") and epc_property_type in ("Flat", "Maisonette"):
        scored_result = property_size_evidence(area, rooms)
        if scored_result == "Flat-like":
            return {
                "conflict": True,
                "winner": "EPC",
                "reason": (
                    "After applying the quantile-based scoring rules based on floor area and number of habitable rooms, the result is {evidence}. Therefore, this conflict is resolved by following the EPC classification for the unit-level property type."
                ).format(evidence=scored_result)
            }
        elif scored_result == "House-like":
            return {
                "conflict": True,
                "winner": "HMLR",
                "reason": (
                    "After applying the quantile-based scoring rules based on floor area and number of habitable rooms, the result is {evidence}. Therefore, this conflict is resolved by following the HMLR classification for the unit-level property type."
                ).format(evidence=scored_result)
            }

    # All other cases with ambiguous score
    return {
        "conflict": True,
        "winner": "HMLR",
        "reason": (
            "HMLR is authoritative for the property type recorded, while EPC property_type provides a secondary classification of the dwelling type. The available independent evidence from the address, EPC floor level, floor area, and habitable room count is insufficient or ambiguous to overturn the HMLR property type. "
            "Therefore, this conflict is resolved by following HMLR for unit-level property type."
        )
    }

# Conclict 2: HMLR "Property Type" VS EPC "Built Form"
def check_built_form(hmlr_type, epc_built_form, address, postcode, epc_full_df):

    if pd.isna(hmlr_type) or pd.isna(epc_built_form):
        return {
            "conflict": False,
            "winner": None,
            "reason": "Missing data"
        }

    # HMLR F = Flat/Maisonette
    # A building's attachment status does not change just because it has been subdivided into flats.
    # So when HMLR records "F", comparing it against EPC built_form would conflate two dimensions of description and produce false conflicts.
    # This combination is therefore handled separately in Conflict 1 (property_type vs property_type), where the unit-level question actually belongs.
    if hmlr_type == "F":
        return {
            "conflict": False,
            "winner": None,
            "reason": "Flat/Maisonette cannot be evaluated using EPC built_form alone, handled by a separate rule"
        }

    if hmlr_type == "O":
        return {
            "conflict": False,
            "winner": None,
            "reason": "HMLR property type 'Other' cannot be mapped reliably"
        }
    if epc_built_form == "Not Recorded":
        return {
            "conflict": False,
            "winner": None,
            "reason": "EPC built form 'Not Recorded' cannot be mapped reliably"
        }

    expected_forms = HMLR_TO_EPC_built_form.get(hmlr_type, [])

    # Consistent
    if epc_built_form in expected_forms:
        return {
            "conflict": False,
            "winner": "consistent",
            "reason": (
                "HMLR property type is consistent with the EPC built form."
            )
        }

    # Rule 1:
    # Use the majority of neighboring built form as strong evidence for built form conflict
    neighbors = get_neighbors_from_epc_full(address, epc_full_df, postcode)

    if not neighbors.empty:
        neighbor_forms = neighbors["built_form"].value_counts()
        majority_form = neighbor_forms.idxmax()
        majority_count = neighbor_forms.max()
        total_neighbors = len(neighbors)

        if majority_form == epc_built_form and majority_count / total_neighbors >= 0.5:
            return {
                "conflict": True,
                "winner": "EPC",
                "reason": (
                    f"HMLR records this property as '{hmlr_type}', but EPC records built_form as '{epc_built_form}'. This is independently corroborated by {majority_count} out of {total_neighbors} nearby properties (within 6 house numbers on the same street) also recorded as '{majority_form}' in the EPC register. "
                    "Physical building form (attached/detached status) is directly observable and unlikely to change over time, so this neighbor-consistency evidence is treated as strong support for EPC."
                )
            }
        elif majority_form in HMLR_TO_EPC_built_form.get(hmlr_type, []) and majority_count / total_neighbors >= 0.5:
            return {
                "conflict": True,
                "winner": "HMLR",
                "reason": (
                    f"EPC records built_form as '{epc_built_form}', inconsistent with HMLR's '{hmlr_type}'. This is independently corroborated by {majority_count} out of {total_neighbors} nearby properties (within 6 house numbers on the same street) also recorded as '{majority_form}' in the HMLR register. "
                    "Physical building form (attached/detached status) is directly observable and unlikely to change over time, so this neighbor-consistency evidence is treated as strong support for HMLR."
                )
            }

    # All other cases: no neighbor data available
    return {
        "conflict": True,
        "winner": "HMLR",
        "reason": (
            "No sufficiently strong neighbor-consistency evidence was available to independently adjudicate the conflict between the two classifications. "
            "As the available data cannot reliably determine which physical-form classification is correct, HMLR is retained as the fallback source for the reconciled transaction-level property classification"
        )
    }

# Conflict 3: HMLR "Old/New" VS EPC "Construction Age Band"
def check_construction_age(hmlr_old_new, epc_construction_age_band, epc_transaction_type, has_prior_transaction):

    if pd.isna(hmlr_old_new) or pd.isna(epc_construction_age_band) or pd.isna(epc_transaction_type):
        return {
            "conflict": False,
            "winner": None,
            "reason": "Missing data"
        }

    # HMLR Y - newly built property when transaction.
    # HMLR N - established residential building when transaction.
    # EPC construction_age_band describes the approximate construction period of the property, which does not directly determine HMLR's Old/New classification.

    # However, EPC transaction_type = "New dwelling" provides additional evidence that the property was newly constructed when EPC was lodged.
    # Since the EPC data used in this project covers records available in 2026, this evidence can only support the property's new-build status around the 2026 EPC record; it cannot establish the property's status in earlier years.
    # Then no matter the HMLR transaction date is before or after EPC inspectation date, this is a strong evidence that the property should be marked as "Y" in HMLR. 
    # This is flagged under the assumption that HMLR PPD fully captures releveant prior transactions, the absence of earlier transactions further strengthens the evidence.

    # Rule 1: Strong evidence as described above
    if (hmlr_old_new == "N" and epc_transaction_type == "New dwelling" and not has_prior_transaction and epc_construction_age_band in ["2026"]):
        return {
            "conflict": True,
            "winner": "EPC",
            "reason": ( 
                "EPC identifies the property as a 'New dwelling' and no earlier HMLR transaction was found for this record. Under the assumption that the available HMLR PPD captures all relevant prior transactions, there is strong evidence that the property would be expected be classified as 'Y' (Yes - newly built) in HMLR. However, HMLR remains authoritative for the recorded Old/New field."
            )
        }
    # Rule 2: Case with earlier transaction found
    if (hmlr_old_new == "N" and epc_transaction_type == "New dwelling" and has_prior_transaction and epc_construction_age_band in ["2026"]):
        return {
            "conflict": True,
            "winner": "HMLR",
            "reason": (
                "EPC identifies the property as a 'New dwelling', while HMLR records the transaction as an established property (N - No). However, an earlier record for the same property was found, so this indicates a conflict in EPC transaction type, which would be expected to be 'Marketed sale'."
            )
        }
    # Rule 3: Weak evidence as described above
    if (hmlr_old_new == "N" and epc_construction_age_band in ["2026"]):
        return {
            "conflict": "Warning",
            "winner": None,
            "reason": (
                f"EPC indicates a recent construction age ({epc_construction_age_band}), while HMLR records the transaction as an established property (N - No). "
                f"However, EPC construction age alone does not determine HMLR's new-build transaction classification, so this is flagged as a weak conflict warning."
            )
        }

    return {
        "conflict": False,
        "winner": "consistent",
        "reason": "HMLR Old/New status is consistent with the EPC construction age."
    }

def run_all():
    # Run over ALL linked properties
    conflicts1 = []
    conflicts2 = []
    conflicts3 = []

    # Load and preprocess EPC for Rule of Conflict 1
    epc = pd.read_csv("data/epc-2026.csv")
    extracted = epc["address1"].str.extract(r"^\s*(\d+)[A-Za-z]?\s*(.*)$")
    epc["house_number"] = pd.to_numeric(extracted[0], errors="coerce")
    epc["street_name"] = extracted[1].str.strip().str.upper()
    epc["postcode_clean"] = epc["postcode"].str.replace(" ", "", regex=False)

    for _, row in linked.iterrows():

        property_type_result = check_property_type(
            row["HMLR_property_type"],
            row["EPC_property_type"],
            row["address"],
            row["EPC_floor_level"],
            row["EPC_total_floor_area"],
            row["EPC_number_habitable_rooms"],
        )

        built_form_result = check_built_form(
            row["HMLR_property_type"],
            row["EPC_built_form"],
            row["address"],
            row["postcode"],
            epc
        )

        prior_transactions = linked[
            (linked["postcode"] == row["postcode"]) &
            (linked["address"] == row["address"]) &
            (linked["HMLR_transaction_date"] < row["HMLR_transaction_date"])
        ]
        has_prior_transaction = len(prior_transactions) > 0
        construction_age_result = check_construction_age(
            row["HMLR_old/new"],
            row["EPC_construction_age_band"],
            row["EPC_transaction_type"],
            has_prior_transaction
        )

        if property_type_result["conflict"]:

            conflicts1.append({
                "transaction_id": row["transaction_id"],
                "postcode": row["postcode"],
                "address": row["address"],

                "conflict_type": "Property_Type",

                "HMLR_property_type":
                    row["HMLR_property_type"],

                "EPC_property_type":
                    row["EPC_property_type"],

                "conflict": True,
                "winner":
                    property_type_result["winner"],
                "reason":
                    property_type_result["reason"]
            })

        if built_form_result["conflict"]:

            conflicts2.append({
                "transaction_id": row["transaction_id"],
                "postcode": row["postcode"],
                "address": row["address"],

                "conflict_type": "Built_Form",

                "HMLR_property_type":
                    row["HMLR_property_type"],

                "EPC_built_form":
                    row["EPC_built_form"],

                "conflict": True,
                "winner":
                    built_form_result["winner"],
                "reason":
                    built_form_result["reason"]
            })

        if construction_age_result["conflict"]:

            conflicts3.append({
                "transaction_id": row["transaction_id"],
                "postcode": row["postcode"],
                "address": row["address"],

                "conflict_type": "Construction_Age_Warning",

                "HMLR_old/new":
                    row["HMLR_old/new"],

                "EPC_construction_age_band":
                    row["EPC_construction_age_band"],

                "HMLR_transaction_date":
                    row["HMLR_transaction_date"],

                "EPC_inspection_date":
                    row["EPC_inspection_date"],

                "EPC_transaction_type":
                    row["EPC_transaction_type"],

                "conflict": 
                    construction_age_result["conflict"],
                "winner":
                    construction_age_result["winner"],
                "reason":
                    construction_age_result["reason"]
            })

    # Create conflict report
    conflict_report1 = pd.DataFrame(conflicts1)
    conflict_report2 = pd.DataFrame(conflicts2)
    conflict_report3 = pd.DataFrame(conflicts3)

    print(f"Linked properties: {len(linked)}")
    print(f"Total conflicts:   {len(conflict_report1)+len(conflict_report2)+len(conflict_report3)}")

    print("\n")
    print(
        f"Property_Type conflicts: "
        f"{len(conflict_report1)}"
    )

    print(
        f"Built_Form conflicts: "
        f"{len(conflict_report2)}"
    )

    print(
        f"Construction_Age conflicts: "
        f"{len(conflict_report3)}"
    )

    if not conflict_report1.empty:
        conflict_report1.to_csv(
            "hmlr_property_type_vs_epc_property_type_conflicts.csv",
            index=False
        )
    if not conflict_report2.empty:
        conflict_report2.to_csv(
            "hmlr_property_type_vs_epc_built_form_conflicts.csv",
            index=False
        )
    if not conflict_report3.empty:
        conflict_report3.to_csv(
            "hmlr_old_new_vs_epc_construsction_age_conflicts.csv",
            index=False
        )
    else:
        print("No property type conflicts found.")

if __name__ == "__main__":
    run_all()