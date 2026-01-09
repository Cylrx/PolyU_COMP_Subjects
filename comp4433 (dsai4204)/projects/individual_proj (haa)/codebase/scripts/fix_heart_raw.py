import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

# Ensure src module is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from src.config import cfg

def validate_subset(series, allowed_set, col_name):
    """Fail-fast if series contains values not in allowed_set."""
    unique_vals = set(series.unique())
    # Handle NaN in unique_vals if necessary, though raw check usually assumes no NaN yet for these cats
    # For safe checking, we filter out NaNs from unique_vals if allowed_set doesn't explicitly contain them
    # But here we expect raw data to be integers mostly.
    
    # If allowed_set implies numeric, ignore numpy nan types for subset check if present? 
    # The plan says "Pre-remap value checks". heart_raw shouldn't have NaNs in these fields yet based on earlier inspection.
    diff = unique_vals - allowed_set
    if diff:
        print(f"[FATAL] Column '{col_name}' contains unexpected values: {diff}")
        print(f"        Allowed: {allowed_set}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Fix heart_raw.csv using UCI dataset as reference.")
    parser.add_argument("--uci-path", type=str, default="data/cleveland_clean.csv", help="Path to UCI reference CSV")
    args = parser.parse_args()

    # 1. Resolve Paths
    raw_path = repo_root / cfg.paths["raw_data"]
    out_path = repo_root / cfg.paths["data"]
    uci_path = repo_root / args.uci_path

    print("========================================================")
    print(" HEART ATTACK ANALYSIS - DATASET FIX SCRIPT")
    print("========================================================")
    print(f"Repo Root: {repo_root}")
    print(f"Raw Input: {raw_path}")
    print(f"UCI Ref  : {uci_path}")
    print(f"Output   : {out_path}")
    print("--------------------------------------------------------")

    if raw_path == out_path:
        print("[FATAL] Input and output paths are the same. Aborting to prevent overwrite loop.")
        sys.exit(1)
    
    if not raw_path.exists():
        print(f"[FATAL] Raw data not found at {raw_path}")
        sys.exit(1)

    # 2. Load heart_raw
    df = pd.read_csv(raw_path)
    initial_rows = len(df)
    print(f"Loaded heart_raw.csv: {initial_rows} rows")

    required_cols = [
        'age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 
        'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output'
    ]
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        print(f"[FATAL] Missing columns in heart_raw: {missing_cols}")
        sys.exit(1)
        
    # Pre-remap validation
    print("Validating raw value ranges...")
    validate_subset(df['cp'], {0, 1, 2, 3}, 'cp')
    validate_subset(df['restecg'], {0, 1, 2}, 'restecg')
    validate_subset(df['slp'], {0, 1, 2}, 'slp')
    validate_subset(df['thall'], {0, 1, 2, 3}, 'thall')
    validate_subset(df['caa'], {0, 1, 2, 3, 4}, 'caa')
    validate_subset(df['output'], {0, 1}, 'output')
    print("... Validation passed.")

    # 3. Apply remaps
    print("\nApplying value remaps (Kaggle -> UCI semantics)...")

    # Mappings
    # cp: {0: 4, 1: 2, 2: 3, 3: 1}
    # restecg: {0: 2, 1: 0, 2: 1}
    # slp: {0: 3, 1: 2, 2: 1}
    # thall: {0: NaN, 1: 6, 2: 3, 3: 7}
    # caa: 4 -> NaN
    # output: 1 - output

    def report_remap(col, changes_dict, series):
        print(f"  [{col}] Value counts before: {series.value_counts().to_dict()}")
        # Check if we introduce NaNs
        nans_before = series.isna().sum()
        return nans_before

    # -- Output --
    print("  [output] Flipping 0<->1")
    df['output'] = 1 - df['output']

    # -- CP --
    print(f"  [cp] Remapping {{0:4, 1:2, 2:3, 3:1}}")
    df['cp'] = df['cp'].map({0: 4, 1: 2, 2: 3, 3: 1})

    # -- RestECG --
    print(f"  [restecg] Remapping {{0:2, 1:0, 2:1}}")
    df['restecg'] = df['restecg'].map({0: 2, 1: 0, 2: 1})

    # -- Slp --
    print(f"  [slp] Remapping {{0:3, 1:2, 2:1}}")
    df['slp'] = df['slp'].map({0: 3, 1: 2, 2: 1})

    # -- Thall --
    print(f"  [thall] Remapping {{0:NaN, 1:6, 2:3, 3:7}}")
    # Use replace to handle NaN mapping gracefully or map
    # Since map with dict produces NaN for unmatched, and we matched all, 
    # but we explicitly want 0 -> NaN.
    # Note: Using map on int column with NaNs will convert to float.
    df['thall'] = df['thall'].map({0: np.nan, 1: 6, 2: 3, 3: 7})

    # -- Caa --
    print(f"  [caa] Treating 4 as NaN")
    df['caa'] = df['caa'].replace(4, np.nan)

    # 4. Drop NaNs and Duplicates
    print("\nCleaning data...")
    
    n_nan_rows = df.isna().any(axis=1).sum()
    if n_nan_rows > 0:
        print(f"  Dropping {n_nan_rows} rows containing NaN values.")
        df = df.dropna()
    else:
        print("  No NaN values generated.")

    n_rows_after_nan = len(df)
    
    # Check exact duplicates
    n_dupes = df.duplicated(keep='first').sum()
    if n_dupes > 0:
        print(f"  Dropping {n_dupes} exact duplicate rows.")
        df = df.drop_duplicates(keep='first')
    else:
        print("  No exact duplicates found.")

    n_final = len(df)
    print(f"Final row count: {n_final} (Removed {initial_rows - n_final} rows total)")

    # 5. Cross-check vs UCI
    print("\nVerifying against UCI reference...")
    if not uci_path.exists():
        print(f"[WARN] UCI path {uci_path} does not exist. Skipping verification.")
    else:
        uci = pd.read_csv(uci_path)
        # Prepare UCI for matching
        # UCI target: 0=healthy, 1-4=sick. Binary: >0 is 1.
        uci_target_bin = (uci['target'] > 0).astype(int)
        
        # We need to match feature-by-feature. 
        # Rename heart_raw columns to match UCI for the join/merge
        # heart_raw: age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall
        # UCI:       age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        
        rename_map = {
            'trtbps': 'trestbps',
            'thalachh': 'thalach',
            'exng': 'exang',
            'slp': 'slope',
            'caa': 'ca',
            'thall': 'thal'
        }
        
        # Create a view of df with UCI names
        df_to_check = df.rename(columns=rename_map)
        
        # Columns to join on (all features)
        join_cols = [c for c in df_to_check.columns if c != 'output']
        
        # Perform merge to find matches
        # We assume UCI also might have NaNs but the clean version usually has them dropped or imputed?
        # The user provided 'cleveland_clean.csv', assuming it's clean.
        # But let's be safe and ensure types match (float vs int).
        
        # Enforce same dtypes for join keys
        for col in join_cols:
            if col in uci.columns:
                # Cast both to float to avoid int/float mismatch issues during merge
                df_to_check[col] = df_to_check[col].astype(float)
                uci[col] = uci[col].astype(float)
        
        # Add index to track which rows matched
        df_to_check['__raw_idx__'] = df_to_check.index
        uci['__uci_idx__'] = uci.index
        uci['__uci_target_bin__'] = uci_target_bin
        
        merged = pd.merge(
            df_to_check, 
            uci[join_cols + ['__uci_idx__', '__uci_target_bin__']], 
            on=join_cols, 
            how='left'
        )
        
        # Check 1: Any unmatched rows?
        unmatched = merged[merged['__uci_idx__'].isna()]
        if len(unmatched) > 0:
            print(f"[FATAL] Found {len(unmatched)} rows in fixed heart_raw that do NOT match any row in UCI reference.")
            print("First 5 unmatched rows:")
            print(unmatched[join_cols].head(5))
            sys.exit(1)
        else:
            print("  [OK] All rows match a corresponding row in UCI dataset.")
            
        # Check 2: Label consistency
        # output vs __uci_target_bin__
        label_mismatch = merged[merged['output'] != merged['__uci_target_bin__']]
        if len(label_mismatch) > 0:
            print(f"[FATAL] Found {len(label_mismatch)} rows where features match but labels disagree.")
            print("First 5 mismatches:")
            print(label_mismatch[join_cols + ['output', '__uci_target_bin__']].head(5))
            sys.exit(1)
        else:
            print("  [OK] Labels align perfectly (UCI target>0 == output).")

    # 6. Write Output
    print(f"\nWriting fixed dataset to {out_path} ...")
    # Ensure directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write without index, using original column names (df has original names)
    df.to_csv(out_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()



