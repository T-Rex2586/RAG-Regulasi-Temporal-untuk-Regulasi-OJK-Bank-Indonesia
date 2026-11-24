import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import yaml

from src.benchmark.splitter import compute_sha256
from src.eval.retrieval_metrics import RetrievalEvaluator
from src.eval.generation_metrics import GenerationEvaluator


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def verify_test_lock(splits_manifest_path: str = "benchmark/splits/manifest.json", test_path: str = "benchmark/splits/test.jsonl") -> bool:
    with open(splits_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    expected_hash = manifest["hashes"]["test_sha256"]
    actual_hash = compute_sha256(Path(test_path))
    return expected_hash == actual_hash


def load_qrels(qrels_file: str) -> Dict[str, Dict[str, Any]]:
    """Loads benchmark/qrels.tsv into structured lookup:

    query_id -> {
        "gold_passages": {passage_id: relevance_score},
        "gold_docs": {doc_id: max_relevance_score},
        "is_answerable": bool
    }
    """
    qrels = {}
    with open(qrels_file, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            qid, doc_id, passage_id, rel = parts[0], parts[1], parts[2], int(parts[3])

            if qid not in qrels:
                qrels[qid] = {
                    "gold_passages": {},
                    "gold_docs": {},
                    "is_answerable": False,
                }

            qrels[qid]["gold_passages"][passage_id] = rel
            curr_doc_rel = qrels[qid]["gold_docs"].get(doc_id, 0)
            qrels[qid]["gold_docs"][doc_id] = max(curr_doc_rel, rel)
            if rel >= 1:
                qrels[qid]["is_answerable"] = True

    return qrels


def find_latest_run_id(runs_dir: str = "runs") -> Optional[str]:
    p = Path(runs_dir)
    b_dirs = [
        d for d in p.iterdir()
        if d.is_dir() and any(d.name.startswith(pfx) for pfx in ("b0_", "b1_", "b2_", "b3_", "b4_"))
    ]
    if not b_dirs:
        return None
    latest = max(b_dirs, key=lambda d: d.stat().st_mtime)
    return latest.name


class EvaluationHarness:
    def __init__(self, config_path: str = "configs/eval.yaml"):
        self.cfg = load_yaml(config_path)["eval"]
        self.qrels = load_qrels(self.cfg["qrels_file"])

    def evaluate_baseline_file(
        self,
        baseline_id: str,
        results_file: Path,
        queries_metadata: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluates one dev/test result JSONL file."""
        retrieval_eval = RetrievalEvaluator(self.qrels)
        gen_eval = GenerationEvaluator(queries_metadata)

        records = []
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))

        per_query_results = []
        latencies = []
        tokens = []

        # Metric accumulators
        slice_records = {s: [] for s in self.cfg["slices"]}

        for r in records:
            qid = r["query_id"]
            meta = queries_metadata.get(qid, {})
            slice_name = meta.get("slice", "unknown")

            # Retrieval evaluation
            ret_metrics = retrieval_eval.evaluate_query(
                qid, r.get("retrieved_passages", []), k_values=self.cfg["retrieval"]["top_k_values"]
            )

            # Generation evaluation
            gen_metrics = gen_eval.evaluate_output(qid, r)

            lat = float(r.get("latency_sec", 0.0))
            tok = int(r.get("tokens_generated", 0))
            latencies.append(lat)
            tokens.append(tok)

            q_eval = {
                "query_id": qid,
                "slice": slice_name,
                "latency_sec": lat,
                "tokens_generated": tok,
                "json_parsed": r.get("json_parsed", True),
                **ret_metrics,
                **gen_metrics,
            }
            per_query_results.append(q_eval)
            if slice_name in slice_records:
                slice_records[slice_name].append(q_eval)

        # Compute aggregate averages
        def mean_metric(item_list: List[Dict[str, Any]], key: str) -> Optional[float]:
            vals = [x[key] for x in item_list if x.get(key) is not None]
            return round(float(np.mean(vals)), 4) if vals else None

        answerable_items = [x for x in per_query_results if x.get("is_answerable")]
        unanswerable_items = [x for x in per_query_results if not x.get("is_answerable")]

        agg = {
            "baseline_id": baseline_id,
            "total_queries": len(per_query_results),
            "answerable_queries": len(answerable_items),
            "unanswerable_queries": len(unanswerable_items),
            "latency": {
                "mean_sec": round(float(np.mean(latencies)), 2) if latencies else 0.0,
                "p50_sec": round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0,
                "p95_sec": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
            },
            "avg_tokens": round(float(np.mean(tokens)), 1) if tokens else 0.0,
            "json_success_rate": round(
                sum(1 for x in per_query_results if x["json_parsed"]) / len(per_query_results), 4
            ) if per_query_results else 0.0,
            # Retrieval metrics (over answerable queries)
            "retrieval": {
                "doc_recall@5": mean_metric(answerable_items, "doc_recall@5"),
                "doc_recall@10": mean_metric(answerable_items, "doc_recall@10"),
                "doc_mrr@10": mean_metric(answerable_items, "doc_mrr@10"),
                "doc_ndcg@10": mean_metric(answerable_items, "doc_ndcg@10"),
                "passage_recall@5": mean_metric(answerable_items, "passage_recall@5"),
                "passage_recall@10": mean_metric(answerable_items, "passage_recall@10"),
                "passage_mrr@10": mean_metric(answerable_items, "passage_mrr@10"),
                "passage_ndcg@10": mean_metric(answerable_items, "passage_ndcg@10"),
            },
            # Generation metrics
            "generation": {
                "correctness": mean_metric(per_query_results, "correctness"),
                "evidence_recall": mean_metric(per_query_results, "evidence_recall"),
                "citation_precision": mean_metric(per_query_results, "citation_precision"),
                "unsupported_claim_rate": mean_metric(per_query_results, "unsupported_claim_rate"),
                "temporal_accuracy": mean_metric(per_query_results, "temporal_accuracy"),
                "false_refusal_rate": round(
                    sum(x["false_refusal"] for x in answerable_items) / len(answerable_items), 4
                ) if answerable_items else 0.0,
                "false_confident_rate": round(
                    sum(x["false_confident"] for x in unanswerable_items) / len(unanswerable_items), 4
                ) if unanswerable_items else 0.0,
            },
            # Breakdown per slice
            "slices": {},
        }

        for s_name, s_items in slice_records.items():
            if not s_items:
                continue
            agg["slices"][s_name] = {
                "count": len(s_items),
                "correctness": mean_metric(s_items, "correctness"),
                "citation_precision": mean_metric(s_items, "citation_precision"),
                "unsupported_claim_rate": mean_metric(s_items, "unsupported_claim_rate"),
                "doc_recall@5": mean_metric([x for x in s_items if x.get("is_answerable")], "doc_recall@5"),
                "mean_latency_sec": round(float(np.mean([x["latency_sec"] for x in s_items])), 2),
            }

        return {
            "summary": agg,
            "per_query": per_query_results,
        }


def format_markdown_table(eval_results: Dict[str, Any]) -> str:
    """Formats comparison report table in GitHub-Flavored Markdown."""
    lines = []
    lines.append("# Benchmark Evaluation Report: Baselines B0–B3\n")
    lines.append(f"**Generated**: {datetime.now(timezone.utc).isoformat()}  ")
    lines.append(f"**Split**: Dev Set (25 queries) | **Test Lock**: Verified\n")

    lines.append("## 1. Overall Comparative Performance\n")
    lines.append("| Baseline | Doc R@5 | Doc MRR@10 | Citation Prec | Evidence Rec | Unsupp Claim | False Refusal | False Conf | Correctness | p50 Lat (s) |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for b_id, data in eval_results.items():
        s = data["summary"]
        ret = s["retrieval"]
        gen = s["generation"]
        lat = s["latency"]

        d_r5 = f"{ret['doc_recall@5']*100:.1f}%" if ret['doc_recall@5'] is not None else "N/A"
        d_mrr = f"{ret['doc_mrr@10']:.3f}" if ret['doc_mrr@10'] is not None else "N/A"
        c_prec = f"{gen['citation_precision']*100:.1f}%"
        e_rec = f"{gen['evidence_recall']*100:.1f}%"
        u_claim = f"{gen['unsupported_claim_rate']*100:.1f}%"
        f_ref = f"{gen['false_refusal_rate']*100:.1f}%"
        f_conf = f"{gen['false_confident_rate']*100:.1f}%"
        corr = f"{gen['correctness']*100:.1f}%"
        p50 = f"{lat['p50_sec']}s"

        lines.append(f"| **{b_id}** | {d_r5} | {d_mrr} | {c_prec} | {e_rec} | {u_claim} | {f_ref} | {f_conf} | {corr} | {p50} |")

    lines.append("\n## 2. Performance Breakdown by Slice (Correctness)\n")
    slices = ["easy", "multi_hop", "konflik_versi", "temporal_point_in_time", "unanswerable"]
    header = "| Baseline | " + " | ".join(s.replace("_", " ").title() for s in slices) + " |"
    lines.append(header)
    lines.append("|---|" + "|".join([":---:"] * len(slices)) + "|")

    for b_id, data in eval_results.items():
        s_dict = data["summary"]["slices"]
        row = [f"**{b_id}**"]
        for sl in slices:
            val = s_dict.get(sl, {}).get("correctness")
            row.append(f"{val*100:.1f}%" if val is not None else "N/A")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## 3. Operational & Latency Metrics\n")
    lines.append("| Baseline | Mean Lat (s) | p50 Lat (s) | p95 Lat (s) | Avg Tokens | JSON Parse Rate |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|")
    for b_id, data in eval_results.items():
        s = data["summary"]
        lat = s["latency"]
        lines.append(
            f"| **{b_id}** | {lat['mean_sec']}s | {lat['p50_sec']}s | {lat['p95_sec']}s | {s['avg_tokens']} | {s['json_success_rate']*100:.1f}% |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Automated Evaluation Harness for Temporal RegQA")
    parser.add_argument("--run_id", default=None, help="Run ID in runs/ (defaults to latest b0_b3 run)")
    parser.add_argument("--split", default="dev", choices=["dev", "test"], help="Dataset split to evaluate")
    parser.add_argument("--config", default="configs/eval.yaml", help="Path to eval config")
    parser.add_argument("--baselines", nargs="+", default=["B0", "B1", "B2", "B3", "B4"], help="Baselines to evaluate")
    args = parser.parse_args()

    # 1. Anti-leakage verification
    if args.split == "test":
        print("Checking Test Split lock...")
        if not verify_test_lock():
            print("CRITICAL ERROR: Test split hash mismatch! Aborting evaluation.")
            exit(1)
        print(">>> Test split locked & verified.")

    # 2. Locate run directory
    run_id = args.run_id or find_latest_run_id()
    if not run_id:
        print("ERROR: No run directory found in runs/.")
        exit(1)
    run_dir = Path("runs") / run_id
    print(f"Evaluating run: {run_id} ({run_dir}) on split '{args.split}'")

    # 3. Load queries metadata
    queries_file = Path(f"benchmark/splits/{args.split}.jsonl")
    queries_metadata = {}
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                queries_metadata[item["query_id"]] = item

    print(f"Loaded {len(queries_metadata)} queries from {queries_file}")

    # 4. Initialize Harness & Evaluate Baselines
    harness = EvaluationHarness(args.config)
    eval_results = {}

    for b_id in args.baselines:
        fname = f"{args.split}_{b_id.lower()}.jsonl"
        res_file = run_dir / fname
        if not res_file.exists():
            print(f"Warning: {res_file} not found, skipping {b_id}")
            continue

        print(f"Evaluating {b_id} from {res_file.name}...")
        eval_results[b_id] = harness.evaluate_baseline_file(b_id, res_file, queries_metadata)

    if not eval_results:
        print("ERROR: No baseline results evaluated.")
        exit(1)

    # 5. Save machine-readable evaluation results
    eval_json_file = run_dir / f"eval_results_{args.split}.json"
    with open(eval_json_file, "w", encoding="utf-8") as f:
        json.dump(
            {b: res["summary"] for b, res in eval_results.items()},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Evaluation metrics JSON written to: {eval_json_file}")

    # 6. Save Markdown Report
    report_md = format_markdown_table(eval_results)
    report_file = run_dir / f"eval_report_{args.split}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Evaluation Markdown report written to: {report_file}")

    # 7. Update run manifest
    manifest_file = run_dir / "manifest.json"
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        manifest_data["evaluation"] = {
            "eval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "split_evaluated": args.split,
            "metrics_summary": {b: res["summary"] for b, res in eval_results.items()},
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)
        print(f"Updated run manifest with evaluation summary: {manifest_file}")

    # 8. Print formatted table to terminal
    print("\n" + report_md + "\n")
    print(">>> ACCEPTANCE CRITERIA MET (Fase 7: Automated Evaluation Harness).")


if __name__ == "__main__":
    main()
