"""Check freshly generated results against the released CSVs in data/.

Usage (from the repository root), after running the smoke test in
generation/README.md:

    python generation/compare_to_release.py \
        --simlex-summary results/simlex/<timestamp>/summary.csv \
        --anisotropy results/anisotropy_smoke_test/anisotropy.csv \
        --dim-reduction results/anisotropy_smoke_test/dim_reduction_r2.csv

Any of the three flags may be omitted. For each file given, every new row is
matched to the released row for the same model, checkpoint, layer, and
condition, and the largest absolute difference is reported. Small differences
(well under the tolerance) can come from hardware or library versions.
"""
import argparse
from pathlib import Path

import pandas as pd

from assemble_csvs import build_geometry, build_simlex_results

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def compare(new_df, released_df, keys, value_col, label, tolerance):
    new_df = new_df.copy()
    released_df = released_df.copy()
    for df in (new_df, released_df):
        df["step"] = df["step"].astype("Float64")
    merged = pd.merge(new_df[keys + [value_col]], released_df[keys + [value_col]],
                      on=keys, how="left", suffixes=("_new", "_released"))
    missing = merged[f"{value_col}_released"].isna().sum()
    diffs = (merged[f"{value_col}_new"] - merged[f"{value_col}_released"]).abs()
    max_diff = diffs.max()
    ok = missing == 0 and max_diff <= tolerance
    print(f"{label}: {len(merged)} rows compared, {missing} with no released match, "
          f"max abs difference {max_diff:.2e} -> {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        worst = merged.assign(abs_diff=diffs).sort_values("abs_diff", ascending=False).head(5)
        print(worst.to_string(index=False))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--simlex-summary", nargs="+", type=Path)
    parser.add_argument("--anisotropy", nargs="+", type=Path)
    parser.add_argument("--dim-reduction", nargs="+", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    results = []
    if args.simlex_summary:
        new = build_simlex_results(args.simlex_summary)
        released = pd.read_csv(DATA_DIR / "simlex_results.csv")
        results.append(compare(new, released, ["name", "layer", "step", "standardization"],
                               "spearman", "SimLex", args.tolerance))
    if args.anisotropy or args.dim_reduction:
        if not (args.anisotropy and args.dim_reduction):
            parser.error("--anisotropy and --dim-reduction must be given together")
        aniso, dims = build_geometry(args.anisotropy, args.dim_reduction)
        results.append(compare(aniso, pd.read_csv(DATA_DIR / "anisotropy.csv"),
                               ["name", "layer", "step"], "anisotropy", "Anisotropy", args.tolerance))
        results.append(compare(dims, pd.read_csv(DATA_DIR / "dim_reduction.csv"),
                               ["name", "layer", "step", "k"], "r2", "Rogue dimensionality", args.tolerance))
    if not results:
        parser.error("give at least one of --simlex-summary, --anisotropy/--dim-reduction")
    print("\nAll checks passed." if all(results) else "\nSome checks did not match; see above.")


if __name__ == "__main__":
    main()
