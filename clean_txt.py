import pandas as pd
import sys

def clean_txt(file_path):
    # List of columns to remove
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
        "PropertyAddressInfoPrivacy", "LegalRange", "LegalTownship",
        "LegalSection", "LegalQuarter", "LegalQuarterQuarter",
        "LegalPhase", "LegalTractNumber", "LegalBlock1", "LegalBlock2",
        "LegalLotNumber1", "LegalLotNumber2", "LegalLotNumber3", "LegalUnit",
        "PartyOwner1NameFull", "PartyOwner1NameFirst", "PartyOwner1NameMiddle",
        "PartyOwner1NameLast", "PartyOwner1NameSuffix", "CompanyFlag",
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
        "LastAssessorTaxRollUpdate", "AssrLastUpdated", "ZonedCodeLocal",
        "PropertyUseMuni", "AssessorLastSaleDate", "AssessorLastSaleAmount",
        "AssessorPriorSaleDate", "AssessorPriorSaleAmount",
        "LastOwnershipTransferDate", "LastOwnershipTransferDocumentNumber",
        "LastOwnershipTransferTransactionID", "DeedLastSaleDocumentBook",
        "DeedLastSaleDocumentPage", "DeedLastDocumentNumber", "DeedLastSaleDate",
        "DeedLastSalePrice", "DeedLastSaleTransactionID"
    ]

    # Load the TXT file assuming it's tab-separated. Change delimiter if needed.
    df = pd.read_csv(file_path, delimiter="\t", low_memory=False)

    # Remove specified columns
    df_cleaned = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors="ignore")

    # Remove rows where 'TrustDescription' contains "Name is a Trust"
    if "TrustDescription" in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned["TrustDescription"] != "Name is a Trust"]

    # Fill empty cells (NaN values) with 0
    df_cleaned.fillna(0, inplace=True)

    # Critical columns for row deletion if missing
    critical_columns = [
        "PropertyLatitude",
        "PropertyLongitude",
        "TaxAssessedValueTotal",
        "TaxAssessedValueLand",
        "TaxMarketValueTotal",
        "TaxMarketValueLand",
        "PreviousAssessedValue"
    ]

    # Remove rows where any critical columns are missing (NaN or 0)
    df_cleaned = df_cleaned[
        df_cleaned[critical_columns].notna().all(axis=1) & (df_cleaned[critical_columns] != 0).all(axis=1)
    ]

    # Outlier detection columns
    outlier_columns = [
        "TaxAssessedValueTotal",
        "TaxAssessedValueLand",
        "TaxMarketValueTotal",
        "TaxMarketValueLand",
        "PreviousAssessedValue"
    ]

    # Remove outliers using IQR method
    for col in outlier_columns:
        if col in df_cleaned.columns:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]

    # Save the cleaned file as a .csv file
    output_file = "cleaned_" + file_path.replace(".txt", ".csv")
    df_cleaned.to_csv(output_file, index=False)

    print(f"Cleaned file saved as: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_txt.py <filename>")
    else:
        clean_txt(sys.argv[1])
