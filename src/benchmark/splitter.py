from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple


def compute_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


class BenchmarkSplitter:
    def __init__(
        self,
        seed: int = 42,
        train_ratio: float = 0.60,
        dev_ratio: float = 0.20,
        test_ratio: float = 0.20,
    ):
        self.seed = seed
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio

    def split_stratified(
        self, queries: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        rng = random.Random(self.seed)

        # Group by (slice, regulator)
        strata = defaultdict(list)
        for q in queries:
            strata[(q["slice"], q["regulator"])].append(q)

        train_set, dev_set, test_set = [], [], []

        for key, group in strata.items():
            rng.shuffle(group)
            n = len(group)
            n_test = max(1, round(n * self.test_ratio)) if n >= 4 else (1 if n >= 3 else 0)
            n_dev = max(1, round(n * self.dev_ratio)) if n >= 4 else (1 if n >= 2 else 0)
            
            # Ensure we don't exceed total
            if n_test + n_dev >= n:
                if n == 1:
                    n_test, n_dev = 0, 0
                elif n == 2:
                    n_test, n_dev = 0, 1
                elif n == 3:
                    n_test, n_dev = 1, 1

            n_train = n - n_test - n_dev

            test_set.extend(group[:n_test])
            dev_set.extend(group[n_test : n_test + n_dev])
            train_set.extend(group[n_test + n_dev :])

        # If strict target count is needed (e.g. 75, 25, 25 for 125 total),
        # balance across slices
        # Let's verify per-slice stratification
        train_set.sort(key=lambda x: x["query_id"])
        dev_set.sort(key=lambda x: x["query_id"])
        test_set.sort(key=lambda x: x["query_id"])

        return train_set, dev_set, test_set

    def split_by_slice_exact(
        self, queries: List[Dict[str, Any]], train_per_slice: int = 15, dev_per_slice: int = 5, test_per_slice: int = 5
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Strictly balances each slice to exact quotas (e.g. 15 train, 5 dev, 5 test)."""
        rng = random.Random(self.seed)
        slice_groups = defaultdict(list)
        for q in queries:
            slice_groups[q["slice"]].append(q)

        train_set, dev_set, test_set = [], [], []

        for s_name, group in slice_groups.items():
            # Sort first for determinism before shuffling
            group_sorted = sorted(group, key=lambda x: x["query_id"])
            rng.shuffle(group_sorted)

            test_subset = group_sorted[:test_per_slice]
            dev_subset = group_sorted[test_per_slice : test_per_slice + dev_per_slice]
            train_subset = group_sorted[test_per_slice + dev_per_slice : test_per_slice + dev_per_slice + train_per_slice]

            test_set.extend(test_subset)
            dev_set.extend(dev_subset)
            train_set.extend(train_subset)

        train_set.sort(key=lambda x: x["query_id"])
        dev_set.sort(key=lambda x: x["query_id"])
        test_set.sort(key=lambda x: x["query_id"])

        return train_set, dev_set, test_set

    def save_splits(
        self,
        train_set: List[Dict[str, Any]],
        dev_set: List[Dict[str, Any]],
        test_set: List[Dict[str, Any]],
        splits_dir: Path,
    ) -> Dict[str, Any]:
        splits_dir.mkdir(parents=True, exist_ok=True)

        train_path = splits_dir / "train.jsonl"
        dev_path = splits_dir / "dev.jsonl"
        test_path = splits_dir / "test.jsonl"

        for p, d in [(train_path, train_set), (dev_path, dev_set), (test_path, test_set)]:
            with open(p, "w", encoding="utf-8") as f:
                for item in d:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        manifest = {
            "seed": self.seed,
            "counts": {
                "total": len(train_set) + len(dev_set) + len(test_set),
                "train": len(train_set),
                "dev": len(dev_set),
                "test": len(test_set),
            },
            "slices": {
                "train": dict(defaultdict(int, {s: sum(1 for q in train_set if q["slice"] == s) for s in {q["slice"] for q in train_set}})),
                "dev": dict(defaultdict(int, {s: sum(1 for q in dev_set if q["slice"] == s) for s in {q["slice"] for q in dev_set}})),
                "test": dict(defaultdict(int, {s: sum(1 for q in test_set if q["slice"] == s) for s in {q["slice"] for q in test_set}})),
            },
            "regulators": {
                "train": dict(defaultdict(int, {r: sum(1 for q in train_set if q["regulator"] == r) for r in {q["regulator"] for q in train_set}})),
                "dev": dict(defaultdict(int, {r: sum(1 for q in dev_set if q["regulator"] == r) for r in {q["regulator"] for q in dev_set}})),
                "test": dict(defaultdict(int, {r: sum(1 for q in test_set if q["regulator"] == r) for r in {q["regulator"] for q in test_set}})),
            },
            "hashes": {
                "train_sha256": compute_sha256(train_path),
                "dev_sha256": compute_sha256(dev_path),
                "test_sha256": compute_sha256(test_path),
            },
        }

        manifest_path = splits_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest
