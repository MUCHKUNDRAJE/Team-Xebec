import pandas as pd
import numpy as np

def check_and_clean_data(input_file, output_file=None):
    
    print("=" * 70)
    print("CME DATA QUALITY CHECKER")
    print("=" * 70)
    
    # Load data
    print(f"\n[1/5] Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"   Initial rows: {len(df)}")
    
    # Check date issues
    print("\n[2/5] Checking date quality...")
    df['time21_5'] = pd.to_datetime(df['time21_5'].str.replace('Z', ''), errors='coerce')
    
    invalid_dates = df['time21_5'].isna().sum()
    if invalid_dates > 0:
        print(f"   ⚠ Found {invalid_dates} invalid dates")
        print(f"   Sample invalid dates:")
        invalid_rows = df[df['time21_5'].isna()].head(3)
        for idx, row in invalid_rows.iterrows():
            orig_date = pd.read_csv(input_file, nrows=idx+1).iloc[-1]['time21_5'] # type: ignore
            print(f"      Row {idx}: {orig_date}")
    
    # Remove invalid dates
    df = df.dropna(subset=['time21_5'])
    
    # Check date range
    min_year = df['time21_5'].dt.year.min()
    max_year = df['time21_5'].dt.year.max()
    print(f"   Date range: {min_year} to {max_year}")
    
    # Filter to reasonable range (2000-2026)
    before_filter = len(df)
    df = df[(df['time21_5'].dt.year >= 2000) & (df['time21_5'].dt.year <= 2026)]
    filtered_count = before_filter - len(df)
    if filtered_count > 0:
        print(f"   ⚠ Filtered {filtered_count} events outside 2000-2026")
    
    print(f"   ✓ Valid dates: {len(df)}")
    
    # Check for duplicates
    print("\n[3/5] Checking for duplicates...")
    duplicates = df.duplicated(subset=['time21_5'], keep='first').sum()
    if duplicates > 0:
        print(f"   ⚠ Found {duplicates} duplicate timestamps")
        df = df.drop_duplicates(subset=['time21_5'], keep='first')
        print(f"   ✓ Removed duplicates")
    else:
        print(f"   ✓ No duplicates found")
    
    # Check missing values
    print("\n[4/5] Checking missing values...")
    critical_fields = ['latitude', 'longitude', 'halfAngle', 'speed']
    
    for field in critical_fields:
        if field in df.columns:
            missing = df[field].isna().sum()
            missing_pct = (missing / len(df)) * 100
            if missing > 0:
                print(f"   {field}: {missing} missing ({missing_pct:.1f}%)")
        else:
            print(f"   ⚠ Missing column: {field}")
    
    # Check for outliers
    print("\n[5/5] Checking for outliers...")
    
    # Speed outliers (> 3000 km/s is extremely rare)
    if 'speed' in df.columns:
        high_speed = (df['speed'] > 3000).sum()
        if high_speed > 0:
            print(f"   ⚠ Found {high_speed} events with speed > 3000 km/s")
            print(f"      Max speed: {df['speed'].max():.1f} km/s")
    
    # Latitude outliers (should be -90 to 90)
    if 'latitude' in df.columns:
        invalid_lat = ((df['latitude'] < -90) | (df['latitude'] > 90)).sum()
        if invalid_lat > 0:
            print(f"   ⚠ Found {invalid_lat} events with invalid latitude")
            df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
            print(f"   ✓ Filtered invalid latitudes")
    
    # Longitude outliers (should be -180 to 180)
    if 'longitude' in df.columns:
        invalid_lon = ((df['longitude'] < -180) | (df['longitude'] > 180)).sum()
        if invalid_lon > 0:
            print(f"   ⚠ Found {invalid_lon} events with invalid longitude")
            df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]
            print(f"   ✓ Filtered invalid longitudes")
    
    # Sort by time
    df = df.sort_values('time21_5').reset_index(drop=True)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Final dataset: {len(df)} events")
    print(f"Date range: {df['time21_5'].min()} to {df['time21_5'].max()}")
    print(f"Years covered: {df['time21_5'].dt.year.max() - df['time21_5'].dt.year.min() + 1}")
    
    # Statistics
    if 'speed' in df.columns:
        print(f"\nSpeed statistics:")
        print(f"  Mean: {df['speed'].mean():.1f} km/s")
        print(f"  Median: {df['speed'].median():.1f} km/s")
        print(f"  Min: {df['speed'].min():.1f} km/s")
        print(f"  Max: {df['speed'].max():.1f} km/s")
    
    # Save cleaned data
    if output_file:
        # Convert time back to string format
        df_save = df.copy()
        df_save['time21_5'] = df_save['time21_5'].dt.strftime('%Y-%m-%dT%H:%MZ')
        df_save.to_csv(output_file, index=False)
        print(f"\n✓ Saved cleaned data to: {output_file}")
    
    print("=" * 70)
    
    return df

class DataQualityChecker:
    @staticmethod
    def check_and_clean(input_file, output_file=None):
        return check_and_clean_data(input_file, output_file)

if __name__ == "__main__":
    # Check and clean your data
    cleaned_df = DataQualityChecker.check_and_clean(
        input_file="donki_data.csv",
        output_file="donki_data_cleaned.csv"
    )
    
    print("\nNow you can train with the cleaned data:")
    print("  python train_cme_model.py")
    print("\nMake sure to update train_cme_model.py to use 'donki_data_cleaned.csv'")