import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from src import config
from src.preprocess import extract_epochs
from src.model import build_model

def load_labels():
    if os.path.exists(config.LABELS_FILE):
        print(f"Loading labels from {config.LABELS_FILE}...")
        lbl_df = pd.read_csv(config.LABELS_FILE)
        return pd.Series(lbl_df.Prediction.values, index=lbl_df.IdFeedBack).to_dict()
    print("WARNING: Labels file not found. Training might fail or use defaults.")
    return None

def main():
    # 1. Load Data
    labels_map = load_labels()
    train_files = glob.glob(os.path.join(config.TRAIN_DIR, "*.csv"))
    
    print(f"Found {len(train_files)} training files.")
    
    all_X, all_y = [], []
    
    for f in train_files:
        try:
            df = pd.read_csv(f)
            X_chunk, y_chunk, _ = extract_epochs(df, f, labels_map=labels_map, mode='train')
            if len(X_chunk) > 0:
                all_X.append(X_chunk)
                all_y.append(y_chunk)
                print(f"Processed {os.path.basename(f)}: {len(X_chunk)} epochs")
        except Exception as e:
            print(f"Error processing {f}: {e}")

    if not all_X:
        print("No data extracted. Exiting.")
        return

    # 2. Stack & Scale
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    print(f"Dataset Shape: {X.shape}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 4. Train Model
    print("Initializing Model...")
    clf = build_model()
    clf.fit(X_train, y_train)
    
    # 5. Evaluate
    val_probs = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs)
    print(f"\n==============================")
    print(f"Validation AUC Score: {auc:.4f}")
    print(f"==============================")
    print(classification_report(y_val, clf.predict(X_val)))

    # 6. Save Artifacts
    joblib.dump(clf, config.MODEL_SAVE_PATH)
    joblib.dump(scaler, os.path.join(config.BASE_DIR, 'scaler.pkl'))
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()