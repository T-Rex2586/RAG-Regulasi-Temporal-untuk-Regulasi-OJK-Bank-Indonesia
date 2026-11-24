# Benchmark Evaluation Report: Baselines B0–B3

**Generated**: 2026-11-19T10:34:48.091616+00:00  
**Split**: Dev Set (25 queries) | **Test Lock**: Verified

## 1. Overall Comparative Performance

| Baseline | Doc R@5 | Doc MRR@10 | Citation Prec | Evidence Rec | Unsupp Claim | False Refusal | False Conf | Correctness | p50 Lat (s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0** | 0.0% | 0.000 | 20.0% | 0.0% | 100.0% | 0.0% | 100.0% | 6.9% | 58.8s |
| **B1** | 0.0% | 0.000 | 28.0% | 12.0% | 80.0% | 20.0% | 100.0% | 13.8% | 66.94s |
| **B2** | 30.0% | 0.227 | 16.0% | 8.0% | 76.0% | 65.0% | 80.0% | 12.4% | 55.45s |
| **B3** | 57.5% | 0.467 | 44.0% | 50.0% | 48.0% | 15.0% | 60.0% | 37.6% | 102.12s |
| **B4** | 57.5% | 0.454 | 36.0% | 32.0% | 40.0% | 35.0% | 20.0% | 32.0% | 83.61s |

## 2. Performance Breakdown by Slice (Correctness)

| Baseline | Easy | Multi Hop | Konflik Versi | Temporal Point In Time | Unanswerable |
|---|:---:|:---:|:---:|:---:|:---:|
| **B0** | 8.5% | 9.3% | 5.8% | 10.7% | 0.0% |
| **B1** | 13.4% | 19.5% | 18.3% | 18.0% | 0.0% |
| **B2** | 24.8% | 6.7% | 3.3% | 7.2% | 20.0% |
| **B3** | 47.4% | 40.5% | 26.5% | 33.5% | 40.0% |
| **B4** | 37.4% | 8.9% | 5.9% | 27.7% | 80.0% |

## 3. Operational & Latency Metrics

| Baseline | Mean Lat (s) | p50 Lat (s) | p95 Lat (s) | Avg Tokens | JSON Parse Rate |
|---|:---:|:---:|:---:|:---:|:---:|
| **B0** | 48.9s | 58.8s | 69.26s | 206.8 | 100.0% |
| **B1** | 76.97s | 66.94s | 151.45s | 214.0 | 64.0% |
| **B2** | 57.7s | 55.45s | 84.69s | 188.4 | 92.0% |
| **B3** | 123.65s | 102.12s | 239.68s | 290.9 | 48.0% |
| **B4** | 108.36s | 83.61s | 266.54s | 191.8 | 60.0% |