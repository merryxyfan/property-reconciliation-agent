import re
import pandas as pd
from find_conflicts import check_built_form, check_property_type, check_construction_age

DATA_LINKED = "data/linked_properties.csv"
DATA_EPC = "data/epc-2026.csv"

def normalize_postcode(pc: str) -> str:
    return re.sub(r"\s+", "", str(pc)).upper()

def load_data():
    print("Loading linked_properties.csv...", flush=True)
    linked = pd.read_csv(DATA_LINKED)
    print("Loading epc-2026.csv...", flush=True)
    epc = pd.read_csv(DATA_EPC)

    # pre-process EPC for conflict 2
    print("Data preprocessing...", flush=True)
    extracted = epc["address1"].str.extract(r"^\s*(\d+)[A-Za-z]?\s*(.*)$")
    epc["house_number"] = pd.to_numeric(extracted[0], errors="coerce")
    epc["street_name"] = extracted[1].str.strip().str.upper()
    epc["postcode_clean"] = epc["postcode"].str.replace(" ", "", regex=False).str.upper()
    return linked, epc

def find_addresses_by_postcode(postcode_input: str, linked: pd.DataFrame) -> pd.DataFrame:
    target = normalize_postcode(postcode_input)
    matches = linked[linked["postcode"] == target]
    return matches

# List unique addresses for the user to choose from the input postcode.
# If the chosen address has multiple transactions, promt for another selection.
def choose_row(matches: pd.DataFrame) -> pd.Series:
    unique_addresses = matches["address"].drop_duplicates().reset_index(drop=True)

    if len(unique_addresses) == 0:
        raise ValueError("No address found for this postcode.")

    if len(unique_addresses) == 1:
        chosen_address = unique_addresses.iloc[0]
    else:
        print("\nFound multiple addresses under this postcode:")
        for i, addr in enumerate(unique_addresses, start=1):
            print(f"{i}. {addr}")
        idx = int(input("Select an address (enter number): ").strip())
        chosen_address = unique_addresses.iloc[idx - 1]

    rows = matches[matches["address"] == chosen_address]

    if len(rows) == 1:
        return rows.iloc[0]

    # In case one address has multiple transaction records by HMLR or inspection records by EPC 
    rows_sorted = rows.sort_values("HMLR_transaction_date", ascending=False).reset_index(drop=True)
    print(f"\nFound {len(rows_sorted)} transactions for this address:")
    for i, r in rows_sorted.iterrows():
        print(f"{i+1}. transaction_id={r['transaction_id']}  HMLR transaction date={r['HMLR_transaction_date']} EPC transaction date={r['EPC_inspection_date']}")
    idx = int(input("Select a transaction (enter number): ").strip())
    return rows_sorted.iloc[idx - 1]

def reconcile_row(row: pd.Series, linked: pd.DataFrame, epc: pd.DataFrame) -> dict:
    prior_transactions = linked[
        (linked["postcode"] == row["postcode"]) &
        (linked["address"] == row["address"]) &
        (linked["HMLR_transaction_date"] < row["HMLR_transaction_date"])
    ]
    has_prior_transaction = len(prior_transactions) > 0

    built_form_result = check_built_form(
        row["HMLR_property_type"], row["EPC_built_form"],
        row["address"], row["postcode"], epc
    )
    property_type_result = check_property_type(
        row["HMLR_property_type"], row["EPC_property_type"],
        row["address"], row["EPC_floor_level"],
        row["EPC_total_floor_area"], row["EPC_number_habitable_rooms"]
    )
    construction_age_result = check_construction_age(
        row["HMLR_old/new"], row["EPC_construction_age_band"],
        row["EPC_transaction_type"], has_prior_transaction
    )

    reconciled = {
        "transaction_id": row["transaction_id"],
        "address": row["address"],
        "postcode": row["postcode"],

        "property_type": {
            "conflict_type": "Property_Type",
            "HMLR_property_type": row["HMLR_property_type"],
            "EPC_property_type": row["EPC_property_type"],
            "conflict": property_type_result["conflict"],
            "winner": property_type_result["winner"],
            "reason": property_type_result["reason"],
        },

        "built_form": {
            "conflict_type": "Built_Form",
            "HMLR_property_type": row["HMLR_property_type"],
            "EPC_built_form": row["EPC_built_form"],
            "conflict": built_form_result["conflict"],
            "winner": built_form_result["winner"],
            "reason": built_form_result["reason"],
        },

        "construction_age": {
            "conflict_type": "Construction_Age_Warning",
            "HMLR_old/new": row["HMLR_old/new"],
            "EPC_construction_age_band": row["EPC_construction_age_band"],
            "EPC_transaction_type": row["EPC_transaction_type"],
            "conflict": construction_age_result["conflict"],
            "winner": construction_age_result["winner"],
            "reason": construction_age_result["reason"],
        },
    }

    return reconciled

def print_result(result: dict):
    print(f"\n=== Reconciled Record: {result['address']} ({result['postcode']}) ===")
    print(f"transaction_id: {result['transaction_id']}")

    pt = result["property_type"]
    print(f"\n[Property_Type]")
    print(f"  HMLR_property_type: {pt['HMLR_property_type']}   EPC_property_type: {pt['EPC_property_type']}")
    print(f"  conflict: {pt['conflict']}   winner: {pt['winner']}")
    print(f"  reason:   {pt['reason']}")

    bf = result["built_form"]
    print(f"\n[Built_Form]")
    print(f"  HMLR_property_type: {bf['HMLR_property_type']}   EPC_built_form: {bf['EPC_built_form']}")
    print(f"  conflict: {bf['conflict']}   winner: {bf['winner']}")
    print(f"  reason:   {bf['reason']}")

    ca = result["construction_age"]
    print(f"\n[Construction_Age]")
    print(f"  HMLR_old/new: {ca['HMLR_old/new']}   EPC_construction_age_band: {ca['EPC_construction_age_band']}")
    print(f"  conflict: {ca['conflict']}   winner: {ca['winner']}")
    print(f"  reason:   {ca['reason']}")

def main():
    print("Hi, I'm your Property Data Reconciliation Agent🏡")
    linked, epc = load_data()

    while True:
        postcode_input = input("Enter postcode (e.g. CM7 1XF): ").strip()
        matches = find_addresses_by_postcode(postcode_input, linked)

        if matches.empty:
            print("No records found for this postcode.")
            return
        else:
            row = choose_row(matches)
            result = reconcile_row(row, linked, epc)
            print_result(result)

        while True:
            continue_query = input("\nWould you like to search for another property? (Y/N): ").strip().upper()

            if continue_query == "Y":
                break
            elif continue_query == "N":
                print("\nThank you for using the Property Data Reconciliation Agent🏡")
                return
            else:
                print("Invalid input. Please enter Y or N.")

if __name__ == "__main__":
    main()