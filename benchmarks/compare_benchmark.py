import json
from pathlib import Path
from statistics import mean, median, stdev
import csv
import math

def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def pick_time_value(entry: dict):
    """Robustly pick a numeric timing from an entry.
    Prefer mean_time_ms, then median_time_ms, then max_time_ms, then any numeric field.
    Returns (value_ms: float or None, size: int or None).
    """
    for key in ("mean_time_ms", "median_time_ms", "max_time_ms"):
        if key in entry:
            try:
                return float(entry[key]), entry.get("size")
            except Exception:
                pass
    # fallback: search for first numeric value
    for k, v in entry.items():
        if isinstance(v, (int, float)):
            # ignore small integer metadata like iteration counts if name-like keys present
            if k in ("iterations", "size", "runs"):
                continue
            try:
                return float(v), entry.get("size")
            except Exception:
                pass
    return None, entry.get("size")

def build_map(entries):
    # allow entries that might already be a dict of results
    out = {}
    for e in entries:
        name = e.get("name") or e.get("id") or e.get("test") or None
        if not name:
            # generate fallback unique name if missing
            name = json.dumps(e, sort_keys=True)
        out[name] = e
    return out

def compare_category(fastLowess_entries, base_R_entries):
    fastLowess_map = build_map(fastLowess_entries)
    base_R_map = build_map(base_R_entries)
    common = sorted(set(fastLowess_map.keys()) & set(base_R_map.keys()))
    rows = []
    speedups = []
    for name in common:
        r_entry = fastLowess_map[name]
        b_entry = base_R_map[name]
        r_val, r_size = pick_time_value(r_entry)
        b_val, b_size = pick_time_value(b_entry)

        row = {
            "name": name,
            "fastLowess_value_ms": r_val,
            "base_R_value_ms": b_val,
            "fastLowess_size": r_size,
            "base_R_size": b_size,
            "notes": []
        }

        if r_val is None or b_val is None:
            row["notes"].append("missing_metric")
            rows.append(row)
            continue

        # core comparisons
        if r_val == 0 or b_val == 0:
            speedup = None
        else:
            speedup = b_val / r_val  # >1 => fastLowess faster by this factor
        row["speedup_base_R_over_fastLowess"] = speedup
        if speedup is not None:
            row["log2_speedup"] = math.log2(speedup) if speedup > 0 else None
            row["percent_change_base_R_vs_fastLowess"] = ((b_val - r_val) / r_val) * 100.0
            speedups.append(speedup)

        # absolute diffs
        row["absolute_diff_ms"] = None if r_val is None or b_val is None else (b_val - r_val)
        row["abs_percent_vs_fastLowess"] = None if r_val == 0 else abs(row["absolute_diff_ms"]) / r_val * 100.0

        # per-point normalization if size available and >0
        size = r_size or b_size
        if size:
            try:
                size_i = int(size)
                row["fastLowess_ms_per_point"] = r_val / size_i
                row["base_R_ms_per_point"] = b_val / size_i
                row["speedup_per_point"] = None if row["fastLowess_ms_per_point"] == 0 else row["base_R_ms_per_point"] / row["fastLowess_ms_per_point"]
            except Exception:
                row["notes"].append("bad_size")

        rows.append(row)
    summary = {
        "compared": len(common),
        "mean_speedup": mean(speedups) if speedups else None,
        "median_speedup": median(speedups) if speedups else None,
        "count_with_metrics": len(speedups),
    }
    return rows, summary

def main():
    repo_root = Path(__file__).resolve().parent
    # walk up to workspace root (same heuristic as other scripts)
    workspace = repo_root
    for _ in range(6):
        if (workspace / "output").exists():
            break
        if workspace.parent == workspace:
            break
        workspace = workspace.parent
    out_dir = workspace / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    fastLowess_path = out_dir / "fastLowess_benchmark.json"
    base_R_path = out_dir / "base_R_benchmark.json"

    if not fastLowess_path.exists() or not base_R_path.exists():
        missing = []
        if not fastLowess_path.exists():
            missing.append(str(fastLowess_path))
        if not base_R_path.exists():
            missing.append(str(base_R_path))
        print("Missing files:", ", ".join(missing))
        return

    fastLowess = load_json(fastLowess_path)
    base_R = load_json(base_R_path)

    all_keys = sorted(set(fastLowess.keys()) | set(base_R.keys()))
    comparison = {}
    overall_speedups = []

    # detailed rows for CSV
    csv_rows = []
    csv_fieldnames = [
        "category","name","fastLowess_value_ms","base_R_value_ms","speedup_base_R_over_fastLowess",
        "log2_speedup","percent_change_base_R_vs_fastLowess","absolute_diff_ms","abs_percent_vs_fastLowess",
        "fastLowess_size","base_R_size","fastLowess_ms_per_point","base_R_ms_per_point","speedup_per_point","notes"
    ]

    for key in all_keys:
        r_entries = fastLowess.get(key, [])
        s_entries = base_R.get(key, [])
        rows, summary = compare_category(r_entries, s_entries)
        comparison[key] = {"rows": rows, "summary": summary}
        if summary["median_speedup"] is not None:
            overall_speedups.append(summary["median_speedup"])
        for row in rows:
            csv_rows.append({
                "category": key,
                "name": row.get("name"),
                "fastLowess_value_ms": row.get("fastLowess_value_ms"),
                "base_R_value_ms": row.get("base_R_value_ms"),
                "speedup_base_R_over_fastLowess": row.get("speedup_base_R_over_fastLowess"),
                "log2_speedup": row.get("log2_speedup"),
                "percent_change_base_R_vs_fastLowess": row.get("percent_change_base_R_vs_fastLowess"),
                "absolute_diff_ms": row.get("absolute_diff_ms"),
                "abs_percent_vs_fastLowess": row.get("abs_percent_vs_fastLowess"),
                "fastLowess_size": row.get("fastLowess_size"),
                "base_R_size": row.get("base_R_size"),
                "fastLowess_ms_per_point": row.get("fastLowess_ms_per_point"),
                "base_R_ms_per_point": row.get("base_R_ms_per_point"),
                "speedup_per_point": row.get("speedup_per_point"),
                "notes": ";".join(row.get("notes", []))
            })

    print("\nBenchmark comparison (base_R_ms / fastLowess_ms):")
    for key, data in comparison.items():
        s = data["summary"]
        print(f"- {key}: compared={s['compared']}, median_speedup={s['median_speedup']}, mean_speedup={s['mean_speedup']}")

    # Top wins and regressions across all categories
    all_rows = [r for cat in comparison.values() for r in cat["rows"] if r.get("speedup_base_R_over_fastLowess") is not None]
    if all_rows:
        sorted_by_speed = sorted(all_rows, key=lambda r: r["speedup_base_R_over_fastLowess"] or 0, reverse=True)
        sorted_by_regression = sorted(all_rows, key=lambda r: r["speedup_base_R_over_fastLowess"] or 0)

        print("\nTop 10 fastLowess wins (largest base_R_ms / fastLowess_ms):")
        for r in sorted_by_speed[:10]:
            print(f"  {r['name']}: base_R={r['base_R_value_ms']:.4f}ms, fastLowess={r['fastLowess_value_ms']:.4f}ms, speedup={r['speedup_base_R_over_fastLowess']:.2f}x")

        print("\nTop 10 regressions (base_R faster than fastLowess):")
        for r in sorted_by_regression[:10]:
            if r["speedup_base_R_over_fastLowess"] < 1.0:
                print(f"  {r['name']}: base_R={r['base_R_value_ms']:.4f}ms, fastLowess={r['fastLowess_value_ms']:.4f}ms, speedup={r['speedup_base_R_over_fastLowess']:.2f}x")

    # Print detailed per-category rows to console
    print("\nDetailed per-category results:")
    for cat, data in comparison.items():
        rows = data["rows"]
        if not rows:
            continue
        print(f"\nCategory: {cat} (compared={data['summary']['compared']})")
        # header
        print(f"{'name':60} {'fastLowess_ms':>10} {'base_R_ms':>10} {'speedup':>8} {'%chg':>8} {'notes'}")
        for r in rows:
            name = (r.get("name") or "")[:60].ljust(60)
            fastLowess_v = r.get("fastLowess_value_ms")
            base_R_v = r.get("base_R_value_ms")
            sp = r.get("speedup_base_R_over_fastLowess")
            pct = r.get("percent_change_base_R_vs_fastLowess")
            notes = ";".join(r.get("notes", []))
            fastLowess_s = f"{fastLowess_v:.4f}" if isinstance(fastLowess_v, (int, float)) else "N/A"
            base_R_s = f"{base_R_v:.4f}" if isinstance(base_R_v, (int, float)) else "N/A"
            sp_s = f"{sp:.2f}x" if isinstance(sp, (int, float)) else "N/A"
            pct_s = f"{pct:.1f}%" if isinstance(pct, (int, float)) else "N/A"
            print(f"{name} {fastLowess_s:>10} {base_R_s:>10} {sp_s:>8} {pct_s:>8} {notes}")

if __name__ == "__main__":
    main()
