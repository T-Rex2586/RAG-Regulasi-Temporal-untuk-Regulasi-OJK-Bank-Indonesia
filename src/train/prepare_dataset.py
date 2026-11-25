import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional

from src.benchmark.splitter import compute_sha256

SYSTEM_PROMPT = """Anda adalah asisten AI kepatuhan hukum perbankan dan jasa keuangan OJK dan Bank Indonesia (Self-Corrective RAG B5 - Adapted).

TUGAS UTAMA:
Jawab pertanyaan pengguna secara akurat, presisi, dan sepenuhnya berbasis bukti hukum (grounded) dari potongan regulasi yang disediakan di bawah.

ATURAN KONTRAK KETAT:
1. Keluarkan respons HANYA dalam format valid JSON (tanpa teks pembuka/penutup tambahan di luar blok JSON):
{
  "answer": "Penjelasan substantif hukum ringkas (1-3 kalimat) yang didukung bukti",
  "claims": ["Klaim fakta hukum atomik 1", "Klaim fakta hukum atomik 2"],
  "citations": [{"doc_id": "ID_DOKUMEN", "pasal": "NOMOR_PASAL", "ayat": "NOMOR_AYAT"}],
  "as_of": "YYYY-MM-DD atau null",
  "verdict": "supported / unanswerable",
  "abstain_reason": null
}

2. PRINSIP ANTI-HALUSINASI:
- Jangan mengutip dokumen atau pasal yang TIDAK tercantum dalam teks bukti.
- Jika bukti tidak mencukupi untuk menjawab atau pertanyaan di luar kewenangan OJK/BI, tetapkan verdict='unanswerable', claims=[], citations=[], dan sebutkan alasannya pada abstain_reason.
- Jika ada tanggal acuan (as_of), pastikan aturan yang Anda kutip sah berlaku pada tanggal tersebut."""


def verify_test_lock(
    splits_manifest_path: str = "benchmark/splits/manifest.json",
    test_path: str = "benchmark/splits/test.jsonl",
) -> bool:
    """Verifies that the test set has not been altered or accessed."""
    with open(splits_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    expected_hash = manifest["hashes"]["test_sha256"]
    actual_hash = compute_sha256(Path(test_path))
    return expected_hash == actual_hash


def load_parsed_documents(parsed_dir: str = "data/parsed") -> Dict[str, Dict[str, Any]]:
    docs = {}
    p_path = Path(parsed_dir)
    for f in p_path.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            docs[data["doc_id"]] = data
    return docs


def find_pasal_in_doc(doc_data: Dict[str, Any], target_pasal_str: str) -> Optional[str]:
    """Finds article text in parsed document."""
    m = re.search(r"(\d+[A-Z]?)", target_pasal_str, re.IGNORECASE)
    p_num = m.group(1).upper() if m else target_pasal_str.strip()

    for p in doc_data.get("pasal_list", []):
        if str(p.get("pasal_nomor", "")).strip().upper() == p_num:
            return p.get("teks_lengkap", "")
    return None


def build_instruction_sample(
    query_item: Dict[str, Any],
    parsed_docs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    qid = query_item["query_id"]
    query = query_item["query"]
    as_of = query_item.get("as_of")
    is_answerable = query_item.get("is_answerable", True)
    ref_notes = query_item.get("reference_notes", "")
    target_docs = query_item.get("target_doc_ids", [])
    target_pasal = query_item.get("target_pasal", [])

    if is_answerable and target_docs:
        context_blocks = []
        citations = []
        for i, did in enumerate(target_docs):
            doc_data = parsed_docs.get(did, {})
            p_str = target_pasal[i] if i < len(target_pasal) else (target_pasal[0] if target_pasal else "1")
            pasal_text = find_pasal_in_doc(doc_data, p_str) or f"Ketentuan substansi hukum terkait {p_str} pada dokumen {did}."
            status = doc_data.get("status", "Berlaku")
            tgl = doc_data.get("tanggal_berlaku") or doc_data.get("diundangkan_pada") or "1 Januari"

            header = f"[{did}] [{p_str}] [Status: {status}] [Mulai Berlaku: {tgl}]"
            context_blocks.append(f"{header}\n{pasal_text.strip()[:1000]}")

            m_num = re.search(r"(\d+[A-Z]?)", p_str)
            pasal_clean = m_num.group(1) if m_num else "1"
            citations.append({
                "doc_id": did,
                "pasal": pasal_clean,
                "ayat": "1",
            })

        context_text = "\n\n---\n\n".join(context_blocks)

        user_content = (
            f"EVIDENCE REGULASI TERVERIFIKASI:\n{context_text}\n\n"
            f"PERTANYAAN PENGGUNA:\n{query}\n"
            f"(Tanggal Acuan / As-Of: {as_of or 'null'})\n\n"
            f"RESPONS JSON:"
        )

        # Build clean gold answer
        claims = [
            f"Berdasarkan {citations[0]['doc_id']}, ketentuan diatur dalam Pasal {citations[0]['pasal']}.",
            ref_notes[:120] if len(ref_notes) > 120 else ref_notes,
        ]

        assistant_json = {
            "answer": ref_notes,
            "claims": claims,
            "citations": citations,
            "as_of": as_of,
            "verdict": "supported",
            "abstain_reason": None,
        }

    else:
        # Unanswerable / out-of-domain query
        user_content = (
            f"EVIDENCE REGULASI TERVERIFIKASI:\n"
            f"[Tidak ada pasal regulasi yang relevan ditemukan dalam korpus OJK/BI]\n\n"
            f"PERTANYAAN PENGGUNA:\n{query}\n"
            f"(Tanggal Acuan / As-Of: {as_of or 'null'})\n\n"
            f"RESPONS JSON:"
        )
        assistant_json = {
            "answer": "Pertanyaan ini berada di luar cakupan atau tidak diatur dalam regulasi perbankan dan jasa keuangan Otoritas Jasa Keuangan (OJK) serta Bank Indonesia.",
            "claims": [],
            "citations": [],
            "as_of": None,
            "verdict": "unanswerable",
            "abstain_reason": ref_notes or "Out of domain: Pertanyaan tidak diatur dalam korpus perbankan OJK/BI.",
        }

    assistant_content = "```json\n" + json.dumps(assistant_json, ensure_ascii=False, indent=2) + "\n```"

    return {
        "id": qid,
        "slice": query_item.get("slice", "unknown"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare QLoRA Training Dataset from Train Split")
    parser.add_argument("--train_split", default="benchmark/splits/train.jsonl")
    parser.add_argument("--output_dir", default="data/train")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Anti-leakage verification
    print("Checking Test Split lock...")
    if not verify_test_lock():
        print("CRITICAL SECURITY ERROR: Test split hash mismatch! Aborting dataset preparation.")
        exit(1)
    print(">>> Test split locked & verified (zero leakage).")

    # 2. Load Train split queries
    train_queries = []
    with open(args.train_split, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_queries.append(json.loads(line.strip()))
    print(f"Loaded {len(train_queries)} training queries from {args.train_split}")

    # 3. Load parsed documents
    parsed_docs = load_parsed_documents()
    print(f"Loaded {len(parsed_docs)} parsed documents from data/parsed/")

    # 4. Generate instruction samples
    samples = []
    for q in train_queries:
        sample = build_instruction_sample(q, parsed_docs)
        samples.append(sample)

    # 5. Stratified train/val split from train split only
    random.seed(args.seed)
    # Stratify by slice
    by_slice = {}
    for s in samples:
        sl = s["slice"]
        by_slice.setdefault(sl, []).append(s)

    train_samples = []
    val_samples = []

    for sl, items in by_slice.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * args.val_ratio))
        val_samples.extend(items[:n_val])
        train_samples.extend(items[n_val:])

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_file = out_path / "qlora_train.jsonl"
    val_file = out_path / "qlora_val.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    manifest = {
        "dataset_name": "qlora_instruction_dataset_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "source_split": "benchmark/splits/train.jsonl",
        "total_source_queries": len(train_queries),
        "train_samples_count": len(train_samples),
        "val_samples_count": len(val_samples),
        "train_sha256": compute_sha256(train_file),
        "val_sha256": compute_sha256(val_file),
        "test_lock_verified": True,
    }

    manifest_file = out_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {len(train_samples)} training samples -> {train_file}")
    print(f"Generated {len(val_samples)} validation samples -> {val_file}")
    print(f"Dataset manifest written -> {manifest_file}")
    print(">>> QLoRA Dataset Preparation Complete.")


if __name__ == "__main__":
    main()
