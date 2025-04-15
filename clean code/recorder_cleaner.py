import pandas as pd
import re
import os

def remove_columns_and_invalid_rows(input_file, columns_to_remove, critical_columns):
    # Extract county name from filename
    match = re.search(r"Recorder_(.*)\.csv", os.path.basename(input_file), re.IGNORECASE)
    if not match:
        raise ValueError("Input file name must follow the format 'Recorder_<County>.csv'")
    
    county = match.group(1)
    output_file = f"{county}_recorder_cleaned.csv"

    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Drop the specified columns
    df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)
    
    # Remove rows where any critical column has 'NA'
    df = df.dropna(subset=critical_columns)

    # Convert ArmsLengthFlag to numeric (handle cases where it's not already)
    df['ArmsLengthFlag'] = pd.to_numeric(df['ArmsLengthFlag'], errors='coerce')

    # Keep rows with ArmsLengthFlag == 1
    arm_length_1 = df[df['ArmsLengthFlag'] == 1]

    # Process ArmsLengthFlag == 0
    arm_length_0 = df[df['ArmsLengthFlag'] == 0]
    if not arm_length_0.empty:
        if 'Grantee1NameLast' in arm_length_0.columns and 'Grantor1NameLast' in arm_length_0.columns:
            arm_length_0['Grantee1NameLast'] = arm_length_0['Grantee1NameLast'].fillna('').str.lower()
            arm_length_0['Grantor1NameLast'] = arm_length_0['Grantor1NameLast'].fillna('').str.lower()
            # Filter out rows where last names match
            arm_length_0 = arm_length_0[arm_length_0['Grantee1NameLast'] != arm_length_0['Grantor1NameLast']]
        else:
            print("One or both of the columns 'Grantee1NameLast' and 'Grantor1NameLast' are missing from the dataset.")

    # Combine valid rows from ArmsLengthFlag 1 and filtered 0
    df = pd.concat([arm_length_1, arm_length_0], ignore_index=True)

    # Remove rows where ArmsLengthFlag is > 1 or NaN
    df = df[df['ArmsLengthFlag'].isin([0, 1])]

    # Filter out top 1% and bottom 1% TransferAmount
    if 'TransferAmount' in df.columns:
        lower_bound = df['TransferAmount'].quantile(0.01)
        upper_bound = df['TransferAmount'].quantile(0.99)
        df = df[(df['TransferAmount'] >= lower_bound) & (df['TransferAmount'] <= upper_bound)]

    
    # Save the modified DataFrame back to CSV
    df.to_csv(output_file, index=False)
    print(f"Updated CSV saved as {output_file}")

        # Save the modified DataFrame back to CSV
    df.to_csv(output_file, index=False)
    print(f"Updated CSV saved as {output_file}")

# List of columns to remove
columns_to_remove = [
    "Grantor1NameFirst",
    "Grantor1NameMiddle", "Grantor1NameLast", "Grantor1NameSuffix", "Grantor2NameFirst",
    "Grantor2NameMiddle", "Grantor2NameLast", "Grantor2NameSuffix", "Grantee1NameFirst",
    "Grantee1NameMiddle", "Grantee1NameLast", "Grantee1NameSuffix", "Grantee2NameFirst",
    "Grantee2NameMiddle", "Grantee2NameLast", "Grantee2NameSuffix", "Mortgage1LenderNameFirst",
    "Mortgage1LenderNameLast", "Mortgage2LenderNameFirst", "Mortgage2LenderNameLast",
    "Grantor1NameFull", "Grantor1InfoEntityClassification", "Grantor1InfoOwnerType",
    "Grantor2NameFull", "Grantor2InfoEntityClassification", "Grantor2InfoOwnerType",
    "Grantee1NameFull", "Grantee1InfoEntityClassification", "Grantee1InfoOwnerType",
    "Grantee2NameFull", "Grantee2InfoEntityClassification", "GranteeInfoVesting1",
    "Mortgage1LenderNameFullStandardized", "Mortgage1LenderAddress",
    "Mortgage2LenderNameFullStandardized", "Mortgage1DocumentNumberFormatted",
    "Mortgage1DocumentNumberLegacy","Mortgage1InstrumentNumber","Mortgage1RecordingDate",
    "Mortgage1Type","Mortgage1Amount","Mortgage1LenderCode","Mortgage1LenderNameFullStandardized",
    "Mortgage1LenderNameFirst","Mortgage1LenderNameLast","Mortgage1LenderAddress",
    "Mortgage1LenderInfoEntityClassification","Mortgage1LenderInfoSellerCarryBackFlag","Mortgage1Term",
    "Mortgage1TermType","Mortgage1TermDate","Mortgage1InfoPrepaymentPenaltyFlag","Mortgage1InfoPrepaymentTerm",
    "Mortgage1InterestRateType","Mortgage2DocumentNumberFormatted","Mortgage2DocumentNumberLegacy",
    "Mortgage2InstrumentNumber","Mortgage2RecordingDate","Mortgage2Type","Mortgage2Amount","Mortgage2LenderCode",
    "Mortgage2LenderNameFullStandardized","Mortgage2LenderNameFirst","Mortgage2LenderNameLast",
    "Mortgage2LenderAddress","Mortgage2LenderInfoEntityClassification","Mortgage2LenderInfoSellerCarryBackFlag",
    "Mortgage2Term","Mortgage2TermType","Mortgage2TermDate","Mortgage2InfoPrepaymentPenaltyFlag",
    "Mortgage2InfoPrepaymentTerm","Mortgage2InterestRateType"
]

# Critical columns required for price prediction
critical_columns = [
    "TransactionID", "AttomID", "PropertyAddressFull", "DocumentRecordingStateCode", "DocumentRecordingCountyName",
    "TransferAmount"
]
#ArmsLengthFlag
#1 - keep
#0 - unknown, get grantee and grantors names and check if theyre similar, kick out similarities
#>1 - remove

#instrument, recording, mortgage dates, actual transaction date is unknown

#prices need to be adjusted for inflation

#kick out top 1% and bottom 1% based on property value

# Example usage
input_csv = "Recorder_Miami-Dade.csv"  # Use your actual file here
remove_columns_and_invalid_rows(input_csv, columns_to_remove, critical_columns)

#transferinfopurchasetypecode, 