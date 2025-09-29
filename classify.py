#!/usr/bin/env python3
"""
classify_shubh.py
A cleaner, configurable classifier script with:
 - 5-fold CV (weighted F1)
 - Feature ablation (MFCC only / scalars only / both)
 - Optional hyperparameter search (GridSearchCV)
 - Held-out split report + confusion matrix

Usage examples:
  python classify_shubh.py
  python classify_shubh.py --no-grid --cv-only
  python classify_shubh.py --features pitch meanf1 meanf2 --seed 7
  python classify_shubh.py --save-cm cm.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from sklearn.model_selection import (
    StratifiedKFold, train_test_split, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, accuracy_score
)
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt


# --------------------------
# MFCC loader (12 x N -> 12 x target_frames -> flatten)
# --------------------------
def load_and_interpolate_mfcc(mfcc_path: Path, target_frames: int = 25) -> np.ndarray:
    mfcc = pd.read_csv(mfcc_path, header=None, sep=r"\s+")
    mfcc_np = mfcc.to_numpy()
    coeffs, frames = mfcc_np.shape
    old_idx = np.arange(frames)
    new_idx = np.linspace(0, frames - 1, target_frames)
    out = np.zeros((coeffs, target_frames), dtype=float)
    for c in range(coeffs):
        f = interp1d(old_idx, mfcc_np[c, :], kind="linear", fill_value="extrapolate")
        out[c, :] = f(new_idx)
    return out.flatten()


def build_dataset(csv_path: Path, mfcc_dir: Path, chosen_features, target_frames=25):
    df = pd.read_csv(csv_path)

    # Scalar features
    X_scalars = df[chosen_features].to_numpy(dtype=float)
    y = df.iloc[:, -1].to_numpy()

    # MFCC features: expect either 'mfcc' (stem) or 'file' (stem)
    if "mfcc" in df.columns:
        stems = df["mfcc"].astype(str)
    elif "file" in df.columns:
        stems = df["file"].astype(str)
    else:
        raise ValueError(
            "CSV must contain a 'mfcc' or 'file' column with filename stems "
            "used in MFCC/<stem>MFCC.csv"
        )

    mfcc_list = []
    for stem in stems:
        mfcc_path = mfcc_dir / f"{stem}MFCC.csv"
        if not mfcc_path.exists():
            raise FileNotFoundError(f"Missing MFCC file: {mfcc_path}")
        mfcc_list.append(load_and_interpolate_mfcc(mfcc_path, target_frames=target_frames))
    X_mfcc = np.vstack(mfcc_list)

    return X_scalars, X_mfcc, y


def cv_f1(pipe, X, y, cv):
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
    return scores.mean(), scores.std()


def plot_and_save_cm(y_true, y_pred, out_png=None, title="Held-out Confusion Matrix"):
    print("\nConfusion matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    try:
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap="Blues")
        plt.title(title)
        plt.tight_layout()
        if out_png:
            plt.savefig(out_png, dpi=200)
            print(f"Saved confusion matrix to {out_png}")
        else:
            plt.show()
    except Exception as e:
        print(f"(Skipping CM plot — {e})")


def heldout_report(X, y, name, base_pipe, seed):
    """Run a consistent held-out split for a given feature set and report metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    model = base_pipe.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n=== Held-out ({name}) ===")
    print(classification_report(y_test, y_pred, digits=2))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    return (
        f1_score(y_test, y_pred, average="weighted"),
        accuracy_score(y_test, y_pred)
    )


def main():
    parser = argparse.ArgumentParser(description="Binary speech variety classification (Shubh version)")
    parser.add_argument("--csv", type=Path, default=Path("dialectsdataoutput.csv"),
                        help="Path to feature CSV (default: dialectsdataoutput.csv)")
    parser.add_argument("--mfcc-dir", type=Path, default=Path("MFCC"),
                        help="Directory containing per-file MFCC CSVs (default: MFCC/)")
    parser.add_argument("--features", nargs="+", default=["pitch", "meanf1", "meanf2"],
                        help="Scalar feature names from the CSV to include")
    parser.add_argument("--frames", type=int, default=25, help="Target MFCC frames (default: 25)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-grid", action="store_true", help="Skip GridSearchCV")
    parser.add_argument("--cv-only", action="store_true", help="Run only CV (no held-out split)")
    parser.add_argument("--save-cm", type=Path, default=None, help="Save confusion matrix PNG to this path")
    args = parser.parse_args()

    print(f"CSV: {args.csv}")
    print(f"MFCC dir: {args.mfcc_dir}")
    print(f"Chosen scalar features: {args.features}")
    print(f"Target MFCC frames: {args.frames}")
    print(f"Seed: {args.seed}")

    X_scalars, X_mfcc, y = build_dataset(args.csv, args.mfcc_dir, args.features, target_frames=args.frames)
    X_both = np.hstack([X_scalars, X_mfcc])

    print(f"\nShapes → scalars: {X_scalars.shape}, mfcc: {X_mfcc.shape}, both: {X_both.shape}")
    print(f"Class balance: {np.bincount(y.astype(int))}")

    # Common pipeline (scaling inside CV and held-out)
    base_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(400, 100, 48, 12),
            activation="relu",
            learning_rate_init=1e-2,
            solver="adam",
            early_stopping=True,
            n_iter_no_change=5,
            max_iter=200,
            random_state=args.seed,
            verbose=False
        ))
    ])

    # --- Held-out reports for all three feature sets (same split) ---
    f1_mfcc, acc_mfcc   = heldout_report(X_mfcc,   y, "MFCC only",        base_pipe, args.seed)
    f1_scal, acc_scal   = heldout_report(X_scalars, y, "Scalars only",     base_pipe, args.seed)
    f1_both, acc_both   = heldout_report(X_both,    y, "MFCC + Scalars",   base_pipe, args.seed)

    print("\nSummary (held-out):")
    print(f"MFCC only        -> F1={f1_mfcc:.2f},  Acc={acc_mfcc:.2f}")
    print(f"Scalars only     -> F1={f1_scal:.2f},  Acc={acc_scal:.2f}")
    print(f"MFCC + Scalars   -> F1={f1_both:.2f},  Acc={acc_both:.2f}")

    # --- 5-FOLD CV (weighted F1) ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    print("\n=== 5-fold CV (weighted F1) ===")
    for name, X in [("MFCC only", X_mfcc), ("Scalars only", X_scalars), ("MFCC + Scalars", X_both)]:
        m, s = cv_f1(base_pipe, X, y, cv)
        print(f"{name:16s}: {m:.3f} ± {s:.3f}")

    # --- Optional GridSearchCV on combined features ---
    if not args.no_grid:
        print("\n=== GridSearchCV on MFCC + Scalars (weighted F1) ===")
        grid = {
            "mlp__hidden_layer_sizes": [(256, 64, 16), (400, 100, 48, 12)],
            "mlp__alpha": [1e-5, 1e-4, 1e-3],
            "mlp__learning_rate_init": [1e-3, 1e-2]
        }
        gs = GridSearchCV(base_pipe, grid, cv=cv, scoring="f1_weighted", n_jobs=-1)
        gs.fit(X_both, y)
        print(f"Best CV F1: {gs.best_score_:.3f}")
        print(f"Best params: {gs.best_params_}")

    if args.cv_only:
        return

    # --- Final held-out split (reportable like Andrew's) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_both, y, test_size=0.25, random_state=args.seed, stratify=y
    )
    print(f"\nHeld-out split → train: {X_train.shape[0]}, test: {X_test.shape[0]}")

    pipe = base_pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("\n=== Held-out Classification Report ===")
    print(classification_report(y_test, y_pred, digits=2))
    plot_and_save_cm(y_test, y_pred, out_png=args.save_cm)


if __name__ == "__main__":
    main()
