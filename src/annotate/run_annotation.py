import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from src.annotate.qrels_generator import QrelsGenerator


def main():
    parser = argparse.ArgumentParser(description="Run Phase 5 Annotation: Dual Blind Annotation, Cohen's Kappa, and Qrels Generation")
    parser.add_argument("--queries", default="benchmark/queries.jsonl", help="Path to master queries JSONL")
    parser.add_argument("--bm25_index", default="index/bm25/heading_aware_v1.pkl", help="Path to BM25 index")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=========================================================================")
    print("      TEMPORAL REGULATORY QA ANNOTATION & QRELS PIPELINE (FASE 5)        ")
    print("=========================================================================")

    qgen = QrelsGenerator(
        queries_path=args.queries,
        bm25_index_path=args.bm25_index,
        seed=args.seed,
    )

    results = qgen.run_pipeline(
        output_qrels_path="benchmark/qrels.tsv",
        disagreement_log_path="benchmark/disagreement_log.csv",
        agreement_json_path="benchmark/agreement.json",
    )

    # Acceptance verification
    iaa = results["iaa"]
    kappa = iaa["cohen_kappa"]
    passed_kappa = kappa >= 0.70
    passed_disagreement_logged = results["disagreements_count"] > 0
    passed_qrels_generated = results["total_pairs"] >= 400

    overall_passed = passed_kappa and passed_disagreement_logged and passed_qrels_generated

    # Write run manifest
    run_id = f"annotation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 5,
        "config": {
            "queries_file": args.queries,
            "bm25_index_file": args.bm25_index,
            "seed": args.seed,
            "relevance_scale": [0, 1, 2],
        },
        "acceptance": {
            "agreement_calculated": True,
            "cohen_kappa_substantial": passed_kappa,
            "cohen_kappa_score": kappa,
            "disagreements_adjudicated": passed_disagreement_logged,
            "qrels_tsv_complete": passed_qrels_generated,
        },
        "metrics": iaa,
        "summary": {
            "total_pairs_labeled": results["total_pairs"],
            "disagreements_count": results["disagreements_count"],
            "raw_agreement_pct": iaa["raw_agreement_pct"],
            "cohen_kappa": kappa,
            "interpretation": iaa["interpretation"],
        },
        "artifacts": {
            "qrels_file": "benchmark/qrels.tsv",
            "agreement_file": "benchmark/agreement.json",
            "disagreement_log_file": "benchmark/disagreement_log.csv",
            "guidelines_file": "benchmark/annotation_guidelines.md",
        },
    }

    manifest_file = run_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print("\n=========================================================================")
    print("                    FASE 5 ACCEPTANCE VERIFICATION                       ")
    print("=========================================================================")
    print(f"Total Query-Passage Pairs Labeled : {results['total_pairs']}")
    print(f"Inter-Annotator Agreement (Raw)   : {iaa['raw_agreement_pct']}%")
    print(f"Cohen's Kappa (Unweighted)        : {kappa} ({iaa['interpretation']})")
    print(f"Disagreements Logged & Adjudicated: {results['disagreements_count']} cases")
    print(f"Master Qrels File                 : {results['qrels_file']}")
    print(f"Run Manifest File                 : {manifest_file}")
    print("-" * 73)
    if overall_passed:
        print(">>> ACCEPTANCE CRITERIA MET (Fase 5: Anotasi & Qrels).")
    else:
        print(">>> ACCEPTANCE CRITERIA FAILED!")
        exit(1)


if __name__ == "__main__":
    main()
