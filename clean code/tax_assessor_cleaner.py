import pandas as pd
import sys
import os

def clean_txt(file_path):
    columns_to_remove = [
        "SitusStateCode", "SitusCounty", "PropertyJurisdictionName", 
        "CombinedStatisticalArea", "CBSAName", "MSAName", "MSACode",
        "MetropolitanDivision", "MinorCivilDivisionName", "CensusBlock",
        "ParcelNumberRaw", "ParcelNumberFormatted", "ParcelNumberYearAdded",
        "ParcelNumberAlternate", "ParcelMapBook", "ParcelMapPage",
        "ParcelNumberYearChange", "ParcelNumberPrevious", "ParcelAccountNumber",
        "PropertyAddressHouseNumber", "PropertyAddressStreetSuffix",
        "PropertyAddressUnitPrefix", "PropertyAddressUnitValue",
        "PropertyAddressState", "PropertyAddressZIP4", "PropertyAddressCRRT",
        "PropertyAddressInfoPrivacy","LegalDescription", "LegalRange", "LegalTownship",
        "LegalSection", "LegalQuarter", "LegalQuarterQuarter",
        "LegalPhase", "LegalTractNumber", "LegalBlock1", "LegalBlock2",
        "LegalLotNumber1", "LegalLotNumber2", "LegalLotNumber3", "LegalUnit",
        "PartyOwner1NameFull", "PartyOwner1NameFirst", "PartyOwner1NameMiddle",
        "PartyOwner1NameLast", "PartyOwner1NameSuffix","TrustDescription", "CompanyFlag",
        "PartyOwner2NameFull", "PartyOwner2NameFirst", "PartyOwner2NameMiddle",
        "PartyOwner2NameLast", "PartyOwner2NameSuffix", "OwnerTypeDescription1",
        "OwnershipVestingRelationCode", "PartyOwner3NameFull",
        "PartyOwner3NameFirst", "PartyOwner3NameMiddle", "PartyOwner3NameLast",
        "PartyOwner3NameSuffix", "PartyOwner4NameFull", "PartyOwner4NameFirst",
        "PartyOwner4NameMiddle", "PartyOwner4NameLast", "PartyOwner4NameSuffix",
        "OwnerTypeDescription2", "ContactOwnerMailingCounty",
        "ContactOwnerMailingFIPS", "ContactOwnerMailAddressFull",
        "ContactOwnerMailAddressHouseNumber", "ContactOwnerMailAddressStreetDirection",
        "ContactOwnerMailAddressStreetName", "ContactOwnerMailAddressStreetSuffix",
        "ContactOwnerMailAddressStreetPostDirection", "ContactOwnerMailAddressUnitPrefix",
        "ContactOwnerMailAddressUnit", "ContactOwnerMailAddressCity",
        "ContactOwnerMailAddressState", "ContactOwnerMailAddressZIP",
        "ContactOwnerMailAddressZIP4", "ContactOwnerMailAddressCRRT",
        "ContactOwnerMailAddressInfoFormat", "ContactOwnerMailInfoPrivacy",
        "StatusOwnerOccupiedFlag", "DeedOwner1NameFull", "DeedOwner1NameFirst",
        "DeedOwner1NameMiddle", "DeedOwner1NameLast", "DeedOwner1NameSuffix",
        "DeedOwner2NameFull", "DeedOwner2NameFirst", "DeedOwner2NameMiddle",
        "DeedOwner2NameLast", "DeedOwner2NameSuffix", "DeedOwner3NameFull",
        "DeedOwner3NameFirst", "DeedOwner3NameMiddle", "DeedOwner3NameLast",
        "DeedOwner3NameSuffix", "DeedOwner4NameFull", "DeedOwner4NameFirst",
        "DeedOwner4NameMiddle", "DeedOwner4NameLast", "DeedOwner4NameSuffix",
        "TaxFiscalYear", "TaxBilledAmount", "TaxDelinquentYear",
        "LastAssessorTaxRollUpdate", "AssrLastUpdated",
        "PropertyUseMuni", "LastOwnershipTransferDate", "LastOwnershipTransferDocumentNumber",
        "LastOwnershipTransferTransactionID", "DeedLastSaleDocumentBook",
        "DeedLastSaleDocumentPage", "DeedLastDocumentNumber", "DeedLastSaleTransactionID"
    ]
    
    
    # Load the TXT file assuming its tab-separated. chang del if needed
    df = pd.read_csv(file_path, delimiter="\t", low_memory=False)
    print(f"Original shape: {df.shape}")





    # --- Normalize column names ---
    df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace
    df_lower_cols = {col.strip().lower(): col for col in df.columns}  # Normalized actual column names
    normalized_remove = [col.strip().lower() for col in columns_to_remove]  # Normalized targets

    # Find actual columns to drop
    actual_cols_to_remove = [df_lower_cols[col] for col in normalized_remove if col in df_lower_cols]

    # Optional: Check for columns that *should* have matched but didnt
    unmatched_cols = [col for col in normalized_remove if col not in df_lower_cols]
    if unmatched_cols:
        print("Warning: These columns from 'columns_to_remove' were not found in the dataset:")
        for col in unmatched_cols:
            print(f" - {col}")

    # Drop matched columns
    df_cleaned = df.drop(columns=actual_cols_to_remove, errors="ignore")
    print(f"After column removal: {df_cleaned.shape}")

    
    
    
    
    
    
    
    
    
    # --- Filter to residential only ---
    if "PropertyUseGroup" in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned["PropertyUseGroup"].str.strip().str.upper() == "RESIDENTIAL"]
        print(f"After filtering for Residential: {df_cleaned.shape}")


    # --- Format ZIP codes as 5-digit strings ---
    if "PropertyAddressZIP" in df_cleaned.columns:
        df_cleaned["PropertyAddressZIP"] = df_cleaned["PropertyAddressZIP"].astype(str).str.extract(r'(\d{5})')
        print("ZIP codes cleaned and formatted.")
    
    # --- Filter out rows with missing or 0 values in must-have columns ---
    must_have_cols = [
        "PropertyLatitude",
        "PropertyLongitude",
        "TaxMarketValueTotal"
    ]
    must_have_cols = [col for col in must_have_cols if col in df_cleaned.columns]

    print(f"Rows before must-have filter: {df_cleaned.shape[0]}")
    df_cleaned = df_cleaned[df_cleaned[must_have_cols].notna().all(axis=1)]
    df_cleaned = df_cleaned[(df_cleaned[must_have_cols] != 0).all(axis=1)]
    print(f"Rows after must-have filter: {df_cleaned.shape[0]}")
    
    


    # Remove top and bottom 1% outliers for price/valuation columns
    outlier_columns = [
        "AssessorLastSaleAmount",
        "AssessorPriorSaleAmount",
        "DeedLastSalePrice",
        "TaxAssessedValueTotal",
        "TaxAssessedValueLand",
        "TaxMarketValueTotal",
        "TaxMarketValueLand",
        "PreviousAssessedValue"
    ]

    for col in outlier_columns:
        if col in df_cleaned.columns:
            lower = df_cleaned[col].quantile(0.01)
            upper = df_cleaned[col].quantile(0.99)
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower) & (df_cleaned[col] <= upper)]
    print(f"After outlier removal: {df_cleaned.shape}")
    
    # Now fill any remaining NaNs (non-critical columns only)
    df_cleaned.fillna(0, inplace=True)

    
    
    
    
    # --- Save cleaned file as a comma-separated CSV ---

    # Extract the county name from the file name
   # Extract county name assuming file format like "X.txt"
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    if "JOINED_" in base_name:
        county = base_name.split("JOINED_")[-1]
    else:
        county = "unknown"

    # Build output filename in the format: "<County>_tax_assessor_cleaned.csv"
    output_file = f"{county}_tax_assessor_cleaned.csv"

    # Save DataFrame as CSV
    df_cleaned.to_csv(output_file, index=False)
    print(f"Cleaned CSV saved as: {output_file}")
    print(f"Final shape: {df_cleaned.shape}")

    
 # CLI entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_txt.py <filename>")
    else:
        clean_txt(sys.argv[1])