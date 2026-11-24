import argparse
import time
from pathlib import Path
import yaml

from src.retrieval.bm25_indexer import BM25Index
from src.retrieval.dense_indexer import DenseIndexer


def main():
    parser = argparse.ArgumentParser(description="Verify BM25 and Qdrant indices with sample queries and filters")
    parser.add_argument("--config", default="configs/index.yaml", help="Path to index config")
    parser.add_argument("--query", default="kewajiban pemenuhan modal minimum bank umum", help="Test query")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["index"]

    qdrant_cfg = cfg.get("qdrant", {})
    dense_indexer = DenseIndexer(
        host=qdrant_cfg.get("host", "127.0.0.1"),
        port=qdrant_cfg.get("port", 6334),
        prefer_grpc=qdrant_cfg.get("prefer_grpc", True),
        qdrant_path=cfg.get("qdrant_dir"),
        model_name=cfg["embedding_model"],
        vector_dim=cfg["embedding_dim"],
        threads=8,
    )

    test_queries = [
        ("Query Umum (Modal Minimum)", "kewajiban pemenuhan modal minimum bank umum", None),
        ("Query Filter OJK", "alih daya penyerahan pekerjaan", {"regulator": "OJK"}),
        ("Query Filter BI", "giro wajib minimum rupiah", {"regulator": "BI"}),
    ]

    print("=========================================================================")
    print("                    RETRIEVAL INDEX INTEGRITY AUDIT                      ")
    print("=========================================================================")

    strategies = ["fixed", "heading_aware", "hierarchical"]
    audit_passed = True

    for strat in strategies:
        strat_cfg = cfg["strategies"][strat]
        bm25_file = Path(strat_cfg["bm25_index_file"])
        coll_name = strat_cfg["collection_name"]

        print(f"\n--- Testing Strategy: {strat.upper()} ---")

        # 1. Test BM25
        if not bm25_file.exists():
            print(f"FAILED: BM25 file {bm25_file} does not exist.")
            audit_passed = False
            continue

        bm25 = BM25Index.load(bm25_file)
        t0 = time.time()
        res_bm25 = bm25.search(args.query, top_k=3)
        lat_bm25 = (time.time() - t0) * 1000
        print(f"BM25 Search ({lat_bm25:.1f}ms): Retrieved {len(res_bm25)} items. Top score: {res_bm25[0]['score'] if res_bm25 else 0:.2f}")

        # 2. Test Qdrant
        t0 = time.time()
        res_dense = dense_indexer.search(coll_name, args.query, top_k=3)
        lat_dense = (time.time() - t0) * 1000
        print(f"Dense Qdrant Search ({lat_dense:.1f}ms): Retrieved {len(res_dense)} items. Top score: {res_dense[0]['score'] if res_dense else 0:.4f}")

        # 3. Test Metadata Filters
        res_filtered = dense_indexer.search(coll_name, "giro wajib minimum", top_k=3, filters={"regulator": "BI"})
        all_bi = all(r["metadata"].get("regulator") == "BI" for r in res_filtered)
        print(f"Filtered Dense (BI only): Retrieved {len(res_filtered)} items, All match filter: {all_bi}")
        if not all_bi:
            audit_passed = False

    dense_indexer.close()

    print("\n=========================================================================")
    if audit_passed:
        print("AUDIT STATUS: PASSED (All 3 strategies successfully queried in BM25 & Qdrant with metadata filtering).")
        exit(0)
    else:
        print("AUDIT STATUS: FAILED.")
        exit(1)


if __name__ == "__main__":
    main()
