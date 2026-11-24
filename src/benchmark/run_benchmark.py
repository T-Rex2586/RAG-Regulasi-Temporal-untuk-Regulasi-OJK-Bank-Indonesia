import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import yaml

from src.benchmark.curator import BenchmarkCurator
from src.benchmark.splitter import BenchmarkSplitter, compute_sha256
from src.benchmark.queries_data import get_curated_queries


def load_config(path: str = "configs/benchmark.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["benchmark"]


def main():
    parser = argparse.ArgumentParser(description="Build, curate, deduplicate, and split Temporal Regulatory QA Benchmark")
    parser.add_argument("--config", default="configs/benchmark.yaml", help="Path to benchmark config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg.get("output_dir", "benchmark"))
    splits_dir = Path(cfg.get("splits_dir", "benchmark/splits"))
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    print("=========================================================================")
    print("           TEMPORAL REGULATORY QA BENCHMARK BUILDER (FASE 4)             ")
    print("=========================================================================")

    # 1. Load queries
    raw_queries = get_curated_queries()
    print(f"Loaded {len(raw_queries)} curated queries across {len(cfg['slices'])} slices.")

    # 2. Curate and audit
    curator = BenchmarkCurator(
        near_dup_threshold=cfg["dedup"]["near_dup_threshold"],
        ngram_size=cfg["dedup"]["ngram_size"],
        parsed_dir=cfg.get("parsed_dir", "data/parsed"),
    )
    audit_res = curator.audit_benchmark(raw_queries)

    print("\n--- AUDIT RESULTS ---")
    print(f"Total Queries: {audit_res['total_queries']}")
    print(f"Slice Distribution: {audit_res['slice_distribution']}")
    print(f"Regulator Distribution: {audit_res['regulator_distribution']}")
    print(f"Schema Errors: {len(audit_res['schema_errors'])}")
    if audit_res["schema_errors"]:
        for err in audit_res["schema_errors"][:5]:
            print(f"  - ERROR: {err}")
    print(f"Exact Duplicates: {len(audit_res['exact_duplicates'])}")
    print(f"Near-Duplicates (threshold >= {cfg['dedup']['near_dup_threshold']}): {len(audit_res['near_duplicates'])}")

    if not audit_res["passed"]:
        print("\n>>> AUDIT FAILED! Please resolve errors before proceeding.")
        exit(1)

    print(">>> AUDIT PASSED: All schema validations, target doc references, and dedup checks passed cleanly.")

    # 3. Save master queries.jsonl
    master_queries_file = output_dir / "queries.jsonl"
    with open(master_queries_file, "w", encoding="utf-8") as f:
        for q in raw_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    master_sha256 = compute_sha256(master_queries_file)
    print(f"\nMaster queries saved to: {master_queries_file} ({master_sha256[:16]}...)")

    # 4. Stratified Split (Balanced per slice)
    splitter = BenchmarkSplitter(seed=cfg["seed"])
    train_set, dev_set, test_set = splitter.split_by_slice_exact(
        raw_queries,
        train_per_slice=15,
        dev_per_slice=5,
        test_per_slice=5,
    )

    split_manifest = splitter.save_splits(train_set, dev_set, test_set, splits_dir)

    print("\n--- SPLITS SUMMARY ---")
    print(f"{'Split':<10} | {'Total':<6} | {'easy':<6} | {'multi_hop':<10} | {'konflik':<8} | {'temporal':<9} | {'unanswer':<9} | {'SHA-256'}")
    print("-" * 90)
    for s_name, s_data, s_hash in [
        ("Train", train_set, split_manifest["hashes"]["train_sha256"]),
        ("Dev", dev_set, split_manifest["hashes"]["dev_sha256"]),
        ("Test", test_set, split_manifest["hashes"]["test_sha256"]),
    ]:
        counts = {s: sum(1 for q in s_data if q["slice"] == s) for s in cfg["slices"]}
        print(
            f"{s_name:<10} | {len(s_data):<6} | {counts.get('easy',0):<6} | {counts.get('multi_hop',0):<10} | {counts.get('konflik_versi',0):<8} | {counts.get('temporal_point_in_time',0):<9} | {counts.get('unanswerable',0):<9} | {s_hash[:16]}..."
        )
    print("=" * 90)

    # 5. Write Run Manifest
    run_id = f"benchmark_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 4,
        "config": cfg,
        "acceptance": {
            "total_queries_ge_100": len(raw_queries) >= 100,
            "slices_balanced": all(v == 25 for v in audit_res["slice_distribution"].values()),
            "dedup_passed": len(audit_res["exact_duplicates"]) == 0 and len(audit_res["near_duplicates"]) == 0,
            "test_split_locked_sha256": split_manifest["hashes"]["test_sha256"],
        },
        "audit": audit_res,
        "splits_manifest": split_manifest,
        "master_sha256": master_sha256,
    }

    manifest_file = run_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print(f"\nRun manifest written to: {manifest_file}")
    print(f"Splits manifest written to: {splits_dir / 'manifest.json'}")
    print("\n>>> ACCEPTANCE CRITERIA MET (Fase 4: Benchmark Dataset).")


if __name__ == "__main__":
    main()
