# Systematic Ablation Studies Report (A1–A7)

**Generated**: 2026-11-25T14:38:25.390873+00:00  
**Dataset**: Development Set (25 Queries, 5 Slices) | **Test Lock**: SHA-256 Verified (Zero Leakage)

## A1: Chunking Strategy (Fixed vs Heading-Aware vs Hierarchical)
| Strategy | Doc Recall@5 | Doc Recall@10 | Doc MRR@10 | Doc nDCG@10 | Avg Context Chars | Total Chunks | Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FIXED** | 66.2% | 70.4% | 0.7239 | 0.5729 | 2984.9 | 50813 | 425.7ms |
| **HEADING_AWARE** | 80.8% | 86.7% | 0.9500 | 0.5302 | 5322.8 | 35828 | 334.3ms |
| **HIERARCHICAL** | 70.4% | 83.8% | 0.8333 | 0.4722 | 4331.8 | 54143 | 417.6ms |

## A2: Retrieval Channels (BM25 vs Dense vs Hybrid RRF vs Filtered)
| Channel | Doc Recall@5 | Doc Recall@10 | Doc MRR@10 | Doc nDCG@10 | Mean Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|
| **bm25_only** | 87.5% | 90.0% | 1.0000 | 0.5750 | 204.2ms |
| **dense_only** | 26.7% | 28.3% | 0.3572 | 0.3096 | 86.3ms |
| **hybrid_rrf** | 80.8% | 86.7% | 0.9500 | 0.5302 | 347.7ms |
| **hybrid_filtered** | 79.6% | 82.1% | 0.9417 | 0.5263 | 334.4ms |

## A3: Verification Loop & Self-Correction (B3 vs B4)
| Configuration | Unsupported Claim Rate | False Confident Rate | False Refusal Rate | Correctness |
|---|:---:|:---:|:---:|:---:|
| **Without Verifier (B3)** | 48.0% | 60.0% | 15.0% | 37.6% |
| **With Verifier (B4)** | 40.0% | 20.0% | 35.0% | 32.0% |
| **Delta (Δ)** | **-8.0%** | **-40.0%** | **+20.0%** | — |

## A4: QLoRA Domain Adaptation (B4 Base vs B5 QLoRA)
| Configuration | JSON Success Rate | Mean Latency (s) | Avg Tokens | Unsupported Claim Rate |
|---|:---:|:---:|:---:|:---:|
| **Base Model (B4)** | 60.0% | 108.36s | 191.8 | 40.0% |
| **QLoRA Adapted (B5)** | 92.0% | 60.2s | 138.2 | 72.0% |
| **Delta (Δ)** | **+32.0%** | **-44.4%** | **-27.9%** | — |

## A5: Agentic Tools Execution (B5 without Tools vs B6 with Tools)
| Configuration | False Refusal Rate | Correctness | JSON Success | Tool Correctness | Unsafe Call Rate | Budget Overrun |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Without Tools (B5)** | 15.0% | 24.7% | 92.0% | N/A | N/A | N/A |
| **With Tools (B6)** | 10.0% | 17.5% | 96.0% | 98.0% | 0.0% | 0.0% |

## A6: Retrieval Policy (Fixed Top-K vs Slice-Adaptive Top-K)
| Policy | Overall Doc Recall | Complex Slice Recall (Multi-Hop / Conflict) | Mean Retrieval Latency |
|---|:---:|:---:|:---:|
| **fixed_k5** | 71.7% | 70.8% | 330.8ms |
| **adaptive_by_slice** | 76.7% | 86.7% | 337.8ms |

## A7: Context Window / Token Budget (2,048 vs 6,000 Tokens)
| Budget Setting | Max Token Budget | Truncation Rate | Retained Doc Recall |
|---|:---:|:---:|:---:|
| **2048_tokens** | 2048 | 75.0% | 79.6% |
| **6000_tokens** | 6000 | 0.0% | 80.8% |