"""
Consolidate two pipeline run XLSXs into one, picking the best row per venue.

Selection rules per venue:
1. Venue only in one run -> use that row
2. Both runs exist, one is_measurable=Yes the other not -> use measurable row
3. Both is_measurable=Yes -> pick less problematic Focus_severity (Ok > Warning > Error > NaN); tie -> run1
4. Neither measurable -> use run1
"""

import pandas as pd

RUN1_PATH = "output_dir/focus_clude_suggestion_gemmini_NA_on_data_17_2/concat_data/laplacian_th_with_blur_measurable_issues.xlsx"
RUN2_PATH = "output_dir/clude_suggestions_for_th_NA_gemmini-dont-toch/2026-02-16_03-48/concat_data/laplacian_th_with_blur_measurable_issues.xlsx"
OUTPUT_PATH = "output_dir/consolidated_best_per_venue.xlsx"

# Lower = better
SEVERITY_RANK = {"Ok": 0, "Warning": 1, "Error": 2}
# NaN gets rank 3 (worst)
DEFAULT_SEVERITY_RANK = 3


def is_measurable_yes(val):
    return val == "Yes"


def severity_rank(val):
    return SEVERITY_RANK.get(val, DEFAULT_SEVERITY_RANK)


def main():
    df1 = pd.read_excel(RUN1_PATH)
    df2 = pd.read_excel(RUN2_PATH)

    # Deduplicate each by venue_id (keep first)
    df1 = df1.drop_duplicates(subset="venue_id", keep="first")
    df2 = df2.drop_duplicates(subset="venue_id", keep="first")

    print(f"Run1: {len(df1)} unique venues")
    print(f"Run2: {len(df2)} unique venues")

    # Index by venue_id for fast lookup
    df1_idx = df1.set_index("venue_id")
    df2_idx = df2.set_index("venue_id")

    all_venues = df1_idx.index.union(df2_idx.index)
    print(f"Total unique venues: {len(all_venues)}")

    chosen_rows = []
    stats = {"only_run1": 0, "only_run2": 0, "measurable_wins": 0, "severity_wins": 0, "tie_run1": 0, "neither_run1": 0}

    for venue_id in all_venues:
        in1 = venue_id in df1_idx.index
        in2 = venue_id in df2_idx.index

        if in1 and not in2:
            chosen_rows.append(df1_idx.loc[venue_id])
            stats["only_run1"] += 1
        elif in2 and not in1:
            chosen_rows.append(df2_idx.loc[venue_id])
            stats["only_run2"] += 1
        else:
            row1 = df1_idx.loc[venue_id]
            row2 = df2_idx.loc[venue_id]
            m1 = is_measurable_yes(row1["is_measurable"])
            m2 = is_measurable_yes(row2["is_measurable"])

            if m1 and not m2:
                chosen_rows.append(row1)
                stats["measurable_wins"] += 1
            elif m2 and not m1:
                chosen_rows.append(row2)
                stats["measurable_wins"] += 1
            elif m1 and m2:
                s1 = severity_rank(row1["Focus_severity"])
                s2 = severity_rank(row2["Focus_severity"])
                if s1 <= s2:
                    chosen_rows.append(row1)
                    stats["tie_run1" if s1 == s2 else "severity_wins"] += 1
                else:
                    chosen_rows.append(row2)
                    stats["severity_wins"] += 1
            else:
                # Neither measurable
                chosen_rows.append(row1)
                stats["neither_run1"] += 1

    result = pd.DataFrame(chosen_rows)
    result.index.name = "venue_id"
    result = result.reset_index()
    result.to_excel(OUTPUT_PATH, index=False)

    print(f"\nWrote {len(result)} rows to {OUTPUT_PATH}")
    print(f"\nSelection summary:")
    print(f"  Only in run1:                  {stats['only_run1']}")
    print(f"  Only in run2:                  {stats['only_run2']}")
    print(f"  Measurable wins over non-meas: {stats['measurable_wins']}")
    print(f"  Better severity wins:          {stats['severity_wins']}")
    print(f"  Equal severity (run1 default): {stats['tie_run1']}")
    print(f"  Neither measurable (run1):     {stats['neither_run1']}")


if __name__ == "__main__":
    main()
