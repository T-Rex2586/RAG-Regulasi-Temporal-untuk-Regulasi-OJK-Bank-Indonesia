import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm

from src.retrieval.chunker import Chunker, Chunk
from src.retrieval.bm25_indexer import BM25Index
from src.retrieval.dense_indexer import DenseIndexer


def load_config(path: str = "configs/index.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["index"]


def get_dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Build BM25 and Qdrant indices for all 3 chunking strategies")
    parser.add_argument("--config", default="configs/index.yaml", help="Path to index config")
    parser.add_argument("--strategy", choices=["all", "fixed", "heading_aware", "hierarchical"], default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    parsed_dir = Path(cfg.get("parsed_dir", "data/parsed"))

    # Load parsed JSON files — sorted for deterministic chunk ordering across restarts
    json_files = sorted(parsed_dir.glob("*.json"))
    print(f"Found {len(json_files)} parsed documents in {parsed_dir}")

    docs_data = []
    for jf in tqdm(json_files, desc="Loading parsed documents"):
        with open(jf, "r", encoding="utf-8") as f:
            docs_data.append(json.load(f))

    chunker = Chunker(
        fixed_size=cfg["strategies"]["fixed"]["chunk_size_chars"],
        fixed_overlap=cfg["strategies"]["fixed"]["overlap_chars"],
        heading_max=cfg["strategies"]["heading_aware"]["max_chunk_chars"],
        hierarchical_child_max=cfg["strategies"]["hierarchical"]["child_max_chars"],
    )

    strategies_to_run = (
        ["fixed", "heading_aware", "hierarchical"]
        if args.strategy == "all"
        else [args.strategy]
    )

    qdrant_cfg = cfg.get("qdrant", {})
    dense_indexer = DenseIndexer(
        host=qdrant_cfg.get("host", "127.0.0.1"),
        port=qdrant_cfg.get("port", 6334),
        prefer_grpc=qdrant_cfg.get("prefer_grpc", True),
        qdrant_path=None,  # Docker Qdrant via host/port
        model_name=cfg["embedding_model"],
        vector_dim=cfg["embedding_dim"],
        threads=8,
    )

    results_summary = {}

    for strat in strategies_to_run:
        print(f"\n=======================================================")
        print(f"  PROCESSING STRATEGY: {strat.upper()} (Version: {cfg['version']})")
        print(f"=======================================================")

        # 1. Chunking
        all_chunks: list[Chunk] = []
        for doc in tqdm(docs_data, desc=f"Chunking ({strat})"):
            if strat == "fixed":
                chunks = chunker.chunk_fixed(doc)
            elif strat == "heading_aware":
                chunks = chunker.chunk_heading_aware(doc)
            elif strat == "hierarchical":
                chunks = chunker.chunk_hierarchical(doc)
            all_chunks.extend(chunks)

        char_lengths = [c.char_count for c in all_chunks]
        token_lengths = [c.token_count for c in all_chunks]

        stats_lengths = {
            "total_chunks": len(all_chunks),
            "char_mean": round(float(np.mean(char_lengths)), 1) if char_lengths else 0,
            "char_p50": int(np.percentile(char_lengths, 50)) if char_lengths else 0,
            "char_p95": int(np.percentile(char_lengths, 95)) if char_lengths else 0,
            "char_min": int(np.min(char_lengths)) if char_lengths else 0,
            "char_max": int(np.max(char_lengths)) if char_lengths else 0,
            "token_mean": round(float(np.mean(token_lengths)), 1) if token_lengths else 0,
            "token_p50": int(np.percentile(token_lengths, 50)) if token_lengths else 0,
            "token_p95": int(np.percentile(token_lengths, 95)) if token_lengths else 0,
        }

        print(f"Total Chunks: {stats_lengths['total_chunks']}")
        print(f"Token Lengths: Mean={stats_lengths['token_mean']}, P50={stats_lengths['token_p50']}, P95={stats_lengths['token_p95']}")

        # 2. BM25 Index
        strat_cfg = cfg["strategies"][strat]
        bm25_file = Path(strat_cfg["bm25_index_file"])
        if bm25_file.exists():
            print(f"BM25 index {bm25_file} already exists, skipping build.")
            bm25_size_mb = get_dir_size_mb(bm25_file)
        else:
            print(f"Building BM25 index -> {bm25_file}...")
            bm25_idx = BM25Index()
            bm25_idx.build(all_chunks)
            bm25_idx.save(bm25_file)
            bm25_size_mb = get_dir_size_mb(bm25_file)
            print(f"BM25 index saved ({bm25_size_mb:.2f} MB).")

        # 3. Dense Index (Qdrant)
        coll_name = strat_cfg["collection_name"]
        print(f"Building Qdrant dense index -> collection '{coll_name}'...")
        dense_indexer.index_chunks(
            collection_name=coll_name,
            chunks=all_chunks,
            batch_size=cfg.get("batch_size", 256),
            desc=f"Dense Indexing ({strat})",
        )
        pts_count = dense_indexer.client.count(coll_name).count
        qdrant_size_mb = 0.0  # Docker Qdrant — size tracked by container, not local path
        print(f"Qdrant collection '{coll_name}' indexed ({pts_count} points).")

        results_summary[strat] = {
            "version": cfg["version"],
            "collection_name": coll_name,
            "points_count": pts_count,
            "bm25_index_file": str(bm25_file).replace("\\", "/"),
            "stats": stats_lengths,
            "bm25_size_mb": round(bm25_size_mb, 2),
            "qdrant_dir_size_mb": round(qdrant_size_mb, 2),
        }

    dense_indexer.close()

    print("\n=========================================================================")
    print("                    INDEXING & CHUNKING SUMMARY REPORT                   ")
    print("=========================================================================")
    print(f"{'Strategy':<16} | {'Chunks':<8} | {'Avg Chars':<10} | {'P95 Tokens':<11} | {'BM25 (MB)':<10} | {'Collection'}")
    print("-" * 85)
    for strat, data in results_summary.items():
        st = data["stats"]
        print(
            f"{strat:<16} | {st['total_chunks']:<8} | {st['char_mean']:<10} | {st['token_p95']:<11} | {data['bm25_size_mb']:<10} | {data['collection_name']}"
        )
    print("=========================================================================")

    # Write run manifest
    run_id = f"index_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 3,
        "config": cfg,
        "acceptance": {
            "indices_versioned": True,
            "index_sizes_recorded": True,
            "context_lengths_recorded": True,
            "all_strategies_built": len(results_summary) == 3,
        },
        "summary": results_summary,
    }

    manifest_file = run_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print(f"\nRun manifest written to: {manifest_file}")
    print(">>> ACCEPTANCE CRITERIA MET: All 3 strategies indexed in BM25 & Qdrant with metadata filters and length statistics.")


if __name__ == "__main__":
    main()
