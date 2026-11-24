import argparse
import json
from pathlib import Path

from src.ingest.source_register import SourceRegister


def main():
    parser = argparse.ArgumentParser(description="Verify integrity of downloaded regulatory corpus")
    parser.add_argument("--register", default="data/source_register.csv", help="Path to source register csv")
    args = parser.parse_args()

    reg_path = Path(args.register)
    if not reg_path.exists():
        print(f"ERROR: Register file {reg_path} does not exist.")
        exit(1)

    register = SourceRegister(reg_path)
    res = register.verify_integrity()

    print("========================================")
    print("      CORPUS INTEGRITY AUDIT REPORT     ")
    print("========================================")
    print(f"Total Registered Documents : {res['total_records']}")
    print(f"Unique SHA-256 Hashes      : {res['unique_hashes']}")
    print(f"Missing Files on Disk      : {res['missing_files_count']}")
    print(f"Corrupted Hashes           : {res['corrupted_hashes_count']}")
    print(f"Duplicate Files Detected   : {res['duplicates_count']}")
    print("----------------------------------------")

    breakdown = {}
    total_bytes = 0
    for rec in register.records.values():
        k = f"{rec.regulator} ({rec.jenis})"
        breakdown[k] = breakdown.get(k, 0) + 1
        total_bytes += rec.file_size_bytes

    print("Distribution:")
    for k, v in sorted(breakdown.items()):
        print(f"  - {k:<25}: {v:>4} documents")
    print(f"Total Size on Disk         : {total_bytes / (1024 * 1024):.2f} MB")
    print("----------------------------------------")

    if res["is_valid"]:
        print("STATUS: ACCEPTED (Meets all Phase 1 requirements: >=300 unique docs, SHA-256 verified)")
        exit(0)
    else:
        print("STATUS: REJECTED / INCOMPLETE (Target >=300 unique docs, 0 missing, 0 corrupt)")
        exit(2)


if __name__ == "__main__":
    main()
