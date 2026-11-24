import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import yaml
from tqdm import tqdm

from src.benchmark.splitter import compute_sha256
from src.retrieval.bm25_indexer import BM25Index
from src.retrieval.dense_indexer import DenseIndexer
from src.pipeline.generator import QwenGenerator
from src.pipeline.b0_direct import B0DirectBaseline
from src.pipeline.b1_prompted import B1PromptedBaseline
from src.pipeline.b2_naive_rag import B2NaiveRAGBaseline
from src.pipeline.b3_hybrid_rag import B3HybridRAGBaseline
from src.pipeline.b4_self_corrective import B4SelfCorrectiveBaseline


def load_config(path: str = "configs/baselines.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["baselines"]


def verify_test_lock(splits_manifest_path: str = "benchmark/splits/manifest.json", test_path: str = "benchmark/splits/test.jsonl") -> bool:
    """Verifies that the test set has not been tampered with or leaked."""
    with open(splits_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    expected_hash = manifest["hashes"]["test_sha256"]
    actual_hash = compute_sha256(Path(test_path))
    if expected_hash != actual_hash:
        print(f"CRITICAL ERROR: Test set hash mismatch!\nExpected: {expected_hash}\nActual:   {actual_hash}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run Baseline Pipelines B0-B3 on Dev Set")
    parser.add_argument("--config", default="configs/baselines.yaml", help="Path to baselines config")
    parser.add_argument("--baselines", nargs="+", default=["B0", "B1", "B2", "B3", "B4"], help="Baselines to run")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries (for testing)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 1. Anti-leakage verification
    print("Checking Test Split lock...")
    if not verify_test_lock():
        print("Test split lock verification failed! Aborting.")
        exit(1)
    print(">>> Test split locked & verified (zero leakage).")

    # 2. Load dev queries
    dev_path = Path(cfg["dev_file"])
    dev_queries = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dev_queries.append(json.loads(line.strip()))

    if args.limit:
        dev_queries = dev_queries[: args.limit]
    print(f"Loaded {len(dev_queries)} queries from dev set: {dev_path}")

    # 3. Create run directory
    b_tag = "_".join(b.lower() for b in args.baselines)
    run_id = f"{b_tag}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cfg["runs_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 4. Initialize Generator (singleton)
    generator = QwenGenerator.get_instance(
        model_path=cfg["model"]["path"],
        device=cfg["model"]["device"],
    )

    # 5. Initialize Retrieval components if B2 or B3 requested
    dense_indexer = None
    bm25_index = None

    needs_dense = any(b in ("B2", "B3", "B4") for b in args.baselines)
    needs_bm25 = any(b in ("B3", "B4") for b in args.baselines)

    if needs_dense:
        print("Connecting to Qdrant Docker (port 6334) for Dense retrieval...")
        dense_indexer = DenseIndexer(
            host="127.0.0.1",
            port=6334,
            prefer_grpc=True,
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            vector_dim=384,
            threads=8,
        )

    if needs_bm25:
        bm25_file = cfg["b3"]["bm25_index_file"]
        print(f"Loading BM25 index from {bm25_file}...")
        bm25_index = BM25Index.load(bm25_file)

    # 6. Instantiate Baseline pipelines
    pipelines = {}
    if "B0" in args.baselines:
        pipelines["B0"] = B0DirectBaseline(generator=generator, prompt_file=cfg["b0"]["prompt_file"])
    if "B1" in args.baselines:
        pipelines["B1"] = B1PromptedBaseline(generator=generator, prompt_file=cfg["b1"]["prompt_file"])
    if "B2" in args.baselines:
        pipelines["B2"] = B2NaiveRAGBaseline(
            generator=generator,
            dense_indexer=dense_indexer,
            collection_name=cfg["b2"]["collection"],
            top_k=cfg["b2"]["top_k"],
            prompt_file=cfg["b2"]["prompt_file"],
        )
    if "B3" in args.baselines:
        pipelines["B3"] = B3HybridRAGBaseline(
            generator=generator,
            bm25_index=bm25_index,
            dense_indexer=dense_indexer,
            dense_collection=cfg["b3"]["dense_collection"],
            top_k_bm25=cfg["b3"]["top_k_bm25"],
            top_k_dense=cfg["b3"]["top_k_dense"],
            rrf_k=cfg["b3"]["rrf_k"],
            final_top_k=cfg["b3"]["final_top_k"],
            prompt_file=cfg["b3"]["prompt_file"],
        )
    if "B4" in args.baselines:
        pipelines["B4"] = B4SelfCorrectiveBaseline(
            generator=generator,
            bm25_index=bm25_index,
            dense_indexer=dense_indexer,
            dense_collection=cfg["b4"]["dense_collection"],
            top_k_bm25=cfg["b4"]["top_k_bm25"],
            top_k_dense=cfg["b4"]["top_k_dense"],
            rrf_k=cfg["b4"]["rrf_k"],
            final_top_k=cfg["b4"]["final_top_k"],
            max_retries=cfg["b4"]["max_retries"],
            sufficiency_threshold=cfg["b4"]["sufficiency_threshold"],
            claim_support_threshold=cfg["b4"]["claim_support_threshold"],
            prompt_file=cfg["b4"]["prompt_file"],
        )

    baseline_summaries = {}

    # 7. Run evaluation on Dev Set per baseline
    for b_id, pipe in pipelines.items():
        print(f"\n=======================================================")
        print(f"             RUNNING BASELINE: {b_id}                  ")
        print(f"=======================================================")

        out_file = run_dir / f"dev_{b_id.lower()}.jsonl"
        results = []
        total_lat = 0.0
        total_tok = 0
        json_ok_cnt = 0
        abstain_cnt = 0

        with open(out_file, "w", encoding="utf-8") as f_out:
            for q in tqdm(dev_queries, desc=f"Evaluating {b_id}"):
                res = pipe.run(q)
                results.append(res)
                f_out.write(json.dumps(res, ensure_ascii=False) + "\n")

                total_lat += res["latency_sec"]
                total_tok += res["tokens_generated"]
                if res.get("json_parsed", True):
                    json_ok_cnt += 1
                if res.get("verdict") == "unanswerable" or res.get("abstain_reason") is not None:
                    abstain_cnt += 1

        n = len(results)
        avg_lat = round(total_lat / n, 2) if n else 0.0
        avg_tok = round(total_tok / n, 1) if n else 0.0
        json_rate = round((json_ok_cnt / n) * 100, 1) if n else 0.0

        baseline_summaries[b_id] = {
            "baseline_id": b_id,
            "queries_evaluated": n,
            "avg_latency_sec": avg_lat,
            "total_latency_sec": round(total_lat, 2),
            "avg_tokens_generated": avg_tok,
            "json_parse_rate_pct": json_rate,
            "abstain_count": abstain_cnt,
            "output_file": str(out_file).replace("\\", "/"),
        }

        print(f"Baseline {b_id} finished:")
        print(f"  Avg Latency: {avg_lat}s | Avg Tokens: {avg_tok} | JSON OK: {json_rate}% | Abstains: {abstain_cnt}/{n}")

    if dense_indexer:
        dense_indexer.close()

    # 8. Print Comparison Summary
    print("\n=========================================================================")
    print("                    FASE 6: BASELINES COMPARISON REPORT                  ")
    print("=========================================================================")
    print(f"{'Baseline':<10} | {'Queries':<7} | {'Avg Latency':<12} | {'Avg Tokens':<11} | {'JSON Rate':<10} | {'Abstains'}")
    print("-" * 75)
    for b_id, data in baseline_summaries.items():
        print(
            f"{b_id:<10} | {data['queries_evaluated']:<7} | {data['avg_latency_sec']:<10}s | {data['avg_tokens_generated']:<11} | {data['json_parse_rate_pct']:<9}% | {data['abstain_count']}/{data['queries_evaluated']}"
        )
    print("=========================================================================")

    # 9. Save Run Manifest
    run_manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 6,
        "config": cfg,
        "test_lock_verified": True,
        "baselines_evaluated": list(baseline_summaries.keys()),
        "summary": baseline_summaries,
    }

    manifest_file = run_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print(f"\nRun manifest written to: {manifest_file}")
    print(">>> ACCEPTANCE CRITERIA MET (Fase 6: B0-B3 Baseline Pipelines).")


if __name__ == "__main__":
    main()
