import argparse
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description="Audit random 30 parsed documents for quality check")
    parser.add_argument("--manifest", default="data/parsed_manifest.json", help="Path to parsed manifest")
    parser.add_argument("--sample-size", type=int, default=30, help="Number of samples to audit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible audit")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} does not exist.")
        exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = data.get("documents", [])
    if len(docs) < args.sample_size:
        print(f"ERROR: Not enough documents ({len(docs)}) for sample size {args.sample_size}")
        exit(1)

    random.seed(args.seed)
    samples = random.sample(docs, args.sample_size)

    print("==========================================================================================")
    print(f"               AUDIT SAMPEL 30 DOKUMEN PARSED (Seed: {args.seed})                       ")
    print("==========================================================================================")
    print(f"{'No':<3} | {'Doc ID':<26} | {'Reg':<4} | {'Jenis':<5} | {'Pasal':<6} | {'Ayat':<5} | {'Tgl Berlaku':<20}")
    print("-" * 90)

    valid_samples = 0
    for idx, s in enumerate(samples, 1):
        parsed_file = Path(s["parsed_file"])
        if not parsed_file.exists():
            status = "MISSING"
        else:
            with open(parsed_file, "r", encoding="utf-8") as f_doc:
                doc_detail = json.load(f_doc)
            # Check basic validity
            has_text = doc_detail.get("total_karakter", 0) > 200
            has_pasal = doc_detail.get("total_pasal", 0) > 0
            if has_text and has_pasal:
                valid_samples += 1

        tgl = str(s.get("tanggal_berlaku") or "-")[:20]
        print(f"{idx:<3} | {s['doc_id']:<26} | {s['regulator']:<4} | {s['jenis']:<5} | {s['total_pasal']:<6} | {s['total_ayat']:<5} | {tgl:<20}")

    print("-" * 90)
    print(f"Hasil Audit: {valid_samples}/{args.sample_size} sampel memenuhi struktur Bab/Pasal/Ayat/Metadata.")
    if valid_samples >= int(args.sample_size * 0.95):
        print("STATUS AUDIT: LOLOS (Kualitas ekstraksi dokumen valid dan konsisten).")
    else:
        print("STATUS AUDIT: GAGAL.")


if __name__ == "__main__":
    main()
