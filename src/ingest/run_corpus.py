import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import yaml

from src.ingest.source_register import SourceRegister
from src.ingest.scraper_bi import BIScraper
from src.ingest.scraper_ojk import OJKScraper


def load_config(config_path: str = "configs/corpus.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Ingest regulatory corpus from OJK & BI")
    parser.add_argument("--config", default="configs/corpus.yaml", help="Path to config yaml")
    parser.add_argument("--target", type=int, default=None, help="Total target count override")
    parser.add_argument("--ojk-target", type=int, default=None, help="OJK target count override")
    parser.add_argument("--bi-target", type=int, default=None, help="BI target count override")
    args = parser.parse_args()

    cfg = load_config(args.config)["corpus"]

    target_total = args.target or cfg.get("target_total", 320)
    target_ojk = args.ojk_target or cfg.get("target_ojk", target_total // 2)
    target_bi = args.bi_target or cfg.get("target_bi", target_total // 2)

    raw_dir = Path(cfg.get("raw_dir", "data/raw"))
    register_path = Path(cfg.get("register_path", "data/source_register.csv"))

    register = SourceRegister(register_path)
    print(f"Loaded existing register with {len(register.records)} records.")

    # 1. Scrape BI
    print(f"\n[1/2] Starting BI Scraper (Target: {target_bi})...")
    bi_scraper = BIScraper(
        register=register,
        raw_dir=raw_dir,
        base_url=cfg["bi"]["base_url"],
        timeout=cfg.get("request_timeout_seconds", 45.0),
        delay=cfg.get("request_delay_seconds", 0.5),
        user_agent=cfg.get("user_agent"),
    )
    bi_count = bi_scraper.scrape(target_count=target_bi)
    print(f"[BI] Collected {bi_count} documents.")

    # 2. Scrape OJK
    print(f"\n[2/2] Starting OJK Scraper (Target: {target_ojk})...")
    ojk_scraper = OJKScraper(
        register=register,
        raw_dir=raw_dir,
        base_url=cfg["ojk"]["base_url"],
        timeout=cfg.get("request_timeout_seconds", 45.0),
        delay=cfg.get("request_delay_seconds", 0.5),
        user_agent=cfg.get("user_agent"),
    )
    ojk_count = ojk_scraper.scrape(
        target_count=target_ojk,
        sektors=cfg["ojk"].get("sektors"),
        jenis_list=cfg["ojk"].get("jenis"),
    )
    print(f"[OJK] Collected {ojk_count} documents.")

    # 3. Final verification & manifest
    register.save()
    verification = register.verify_integrity()
    print("\n=== Corpus Verification Results ===")
    for k, v in verification.items():
        print(f"  {k}: {v}")

    # Breakdown statistics
    breakdown = {}
    total_bytes = 0
    for rec in register.records.values():
        key = f"{rec.regulator}_{rec.jenis}"
        breakdown[key] = breakdown.get(key, 0) + 1
        total_bytes += rec.file_size_bytes

    print("\n=== Breakdown by Regulator & Jenis ===")
    for k, v in sorted(breakdown.items()):
        print(f"  {k}: {v} documents")
    print(f"Total Storage: {total_bytes / (1024 * 1024):.2f} MB")

    # Record run manifest
    run_id = f"corpus_init_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 1,
        "config": cfg,
        "verification": verification,
        "breakdown": breakdown,
        "total_storage_mb": round(total_bytes / (1024 * 1024), 2),
    }

    manifest_file = run_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nRun manifest written to: {manifest_file}")
    if verification["is_valid"]:
        print(">>> ACCEPTANCE CRITERIA MET: >=300 documents collected, verified SHA-256 integrity, 0 duplicates.")
    else:
        print(f">>> WARNING: Current count {verification['total_records']}/300.")


if __name__ == "__main__":
    main()
