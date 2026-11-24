import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from tqdm import tqdm

from src.ingest.parser import LegalDocParser


def main():
    parser = argparse.ArgumentParser(description="Parse all regulatory PDFs into structured JSON")
    parser.add_argument("--register", default="data/source_register.csv", help="Source register path")
    parser.add_argument("--out-dir", default="data/parsed", help="Output directory for parsed JSONs")
    args = parser.parse_args()

    reg_path = Path(args.register)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(reg_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    print(f"Total documents to parse: {len(records)}")
    doc_parser = LegalDocParser()

    parsed_manifest_items = []
    success_count = 0
    total_pasal_all = 0
    total_ayat_all = 0
    pasal_distribution = {"0": 0, "1-10": 0, "11-50": 0, "51-100": 0, ">100": 0}

    for rec in tqdm(records, desc="Parsing PDFs"):
        pdf_file = Path(rec["file_path"])
        if not pdf_file.exists():
            print(f"WARNING: File {pdf_file} missing on disk.")
            continue

        try:
            parsed = doc_parser.parse(
                pdf_path=pdf_file,
                doc_id=rec["doc_id"],
                regulator=rec["regulator"],
                jenis=rec["jenis"],
                nomor=rec["nomor"],
                tahun=rec["tahun"],
                judul=rec["judul"],
                status=rec["status"],
            )

            out_file = out_dir / f"{rec['doc_id']}.json"
            with open(out_file, "w", encoding="utf-8") as f_out:
                json.dump(parsed.to_dict(), f_out, ensure_ascii=False, indent=2)

            if parsed.parsed_successfully:
                success_count += 1

            p_count = parsed.total_pasal
            total_pasal_all += p_count
            total_ayat_all += parsed.total_ayat

            if p_count == 0 or (p_count == 1 and parsed.pasal_list[0].pasal_nomor == "Naratif"):
                pasal_distribution["0"] += 1
            elif 1 <= p_count <= 10:
                pasal_distribution["1-10"] += 1
            elif 11 <= p_count <= 50:
                pasal_distribution["11-50"] += 1
            elif 51 <= p_count <= 100:
                pasal_distribution["51-100"] += 1
            else:
                pasal_distribution[">100"] += 1

            parsed_manifest_items.append({
                "doc_id": parsed.doc_id,
                "regulator": parsed.regulator,
                "jenis": parsed.jenis,
                "nomor": parsed.nomor,
                "tahun": parsed.tahun,
                "total_pasal": parsed.total_pasal,
                "total_ayat": parsed.total_ayat,
                "tanggal_berlaku": parsed.tanggal_berlaku,
                "tanggal_ditetapkan": parsed.tanggal_ditetapkan,
                "mencabut_count": len(parsed.mencabut),
                "mengubah_count": len(parsed.mengubah),
                "parsed_successfully": parsed.parsed_successfully,
                "parsed_file": str(out_file).replace("\\", "/"),
            })
        except Exception as e:
            print(f"ERROR parsing {rec['doc_id']}: {e}")

    total_docs = len(records)
    success_rate = (success_count / total_docs * 100) if total_docs > 0 else 0

    print("\n========================================")
    print("        PARSING SUMMARY REPORT          ")
    print("========================================")
    print(f"Total Documents Processed   : {total_docs}")
    print(f"Successfully Parsed         : {success_count} ({success_rate:.2f}%)")
    print(f"Total Pasals Extracted      : {total_pasal_all}")
    print(f"Total Ayats Extracted       : {total_ayat_all}")
    print("Pasal Distribution per Document:")
    for bucket, count in pasal_distribution.items():
        print(f"  - {bucket:<10}: {count} documents")
    print("----------------------------------------")

    # Save parsed manifest
    manifest_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_documents": total_docs,
        "successfully_parsed": success_count,
        "success_rate_percent": round(success_rate, 2),
        "total_pasal": total_pasal_all,
        "total_ayat": total_ayat_all,
        "pasal_distribution": pasal_distribution,
        "documents": parsed_manifest_items,
    }

    parsed_manifest_path = Path("data/parsed_manifest.json")
    with open(parsed_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, ensure_ascii=False, indent=2)

    # Save run manifest
    run_id = f"parser_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 2,
        "acceptance": {
            "target_success_rate": 95.0,
            "achieved_success_rate": round(success_rate, 2),
            "is_accepted": success_rate >= 95.0,
        },
        "stats": {
            "total_documents": total_docs,
            "successfully_parsed": success_count,
            "total_pasal": total_pasal_all,
            "total_ayat": total_ayat_all,
            "pasal_distribution": pasal_distribution,
        },
    }

    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print(f"Parsed manifest written to: {parsed_manifest_path}")
    print(f"Run manifest written to: {run_dir / 'manifest.json'}")

    if success_rate >= 95.0:
        print(">>> ACCEPTANCE CRITERIA MET: >=95% documents successfully parsed.")
    else:
        print(">>> WARNING: Acceptance criteria NOT met (<95%).")


if __name__ == "__main__":
    main()
