import pandas as pd
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Data
def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logging.info("CSV file loaded successfully.")
        return df
    except FileNotFoundError:
        logging.error(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

# Convert Columns to DateTime
def convert_to_datetime(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Convert specified columns to datetime format."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            logging.info(f"Converted column '{col}' to datetime.")
        else:
            logging.warning(f"Column '{col}' not found in the DataFrame.")
    return df

# Calculate Age
def calculate_age(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate patient age based on STARTDATE and DOB."""
    if 'STARTDATE' in df.columns and 'DOB' in df.columns:
        df = df[df['DOB'] >= '1901-01-01']  # Filter unrealistic birthdates
        df['AGE'] = (df['STARTDATE'] - df['DOB']).dt.days / 365.25
        logging.info("Age calculated successfully.")
    else:
        logging.warning("STARTDATE or DOB column missing.")
    return df

# Process Death Column
def process_death_column(df: pd.DataFrame) -> pd.DataFrame:
    """Create DEATH column based on DEATHTIME."""
    if 'DEATHTIME' in df.columns:
        df['DEATH'] = df['DEATHTIME'].notna().astype(int)
        logging.info("DEATH column processed.")
    else:
        logging.warning("DEATHTIME column not found.")
    return df

# Encode Categorical Variables
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables like GENDER and apply one-hot encoding."""
    if 'GENDER' in df.columns:
        df['GENDER'] = df['GENDER'].map({'M': 0, 'F': 1})
        logging.info("GENDER column encoded.")
    else:
        logging.warning("GENDER column not found.")
    
    categorical_columns = ['ROUTE', 'DRUG']
    for col in categorical_columns:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
            logging.info(f"One-hot encoding applied to '{col}'.")
        else:
            logging.warning(f"Column '{col}' not found for one-hot encoding.")
    
    return df

# Convert Numeric Columns
def convert_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Convert specified columns to numeric format."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            logging.info(f"Converted column '{col}' to numeric.")
        else:
            logging.warning(f"Column '{col}' not found for numeric conversion.")
    return df

# Process INTIME Column
def process_intime(df: pd.DataFrame) -> pd.DataFrame:
    """Process INTIME column and extract ICU_DATE and ICU_TIME."""
    if 'INTIME' in df.columns:
        df['INTIME'] = pd.to_datetime(df['INTIME'], errors='coerce')
        df['ICU_DATE'] = df['INTIME'].dt.date
        df['ICU_TIME'] = df['INTIME'].dt.time
        logging.info("Processed INTIME column.")
    else:
        logging.warning("INTIME column not found.")
    return df

# Main Processing Function
def main():
    file_path = os.path.join("data", "raw", "ehr_data_raw.csv")
    df = load_data(file_path)
    
    df = convert_to_datetime(df, ['STARTDATE', 'DOB', 'DEATHTIME', 'INTIME'])
    df = calculate_age(df)
    df = process_death_column(df)
    df = encode_categorical(df)
    df = convert_numeric(df, ['DOSE_VAL_RX'])
    df = process_intime(df)
    
    # Save processed data
    output_path = 'data/processed/ehr_data_processed.csv'
    df.to_csv(output_path, index=False)
    logging.info(f"Processed data saved to '{output_path}'.")

if __name__ == "__main__":
    main()
