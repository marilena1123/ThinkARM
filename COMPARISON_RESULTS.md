# Framework Comparison Results

## Summary

Comparing two reasoning annotation frameworks:
- **ThinkARM**: Episode-based analysis with 8 functional episodes (Read, Analyze, Plan, Implement, Explore, Verify, Monitor, Answer)
- **BLOOM**: Cognitive taxonomy with 6 levels (Remember, Understand, Apply, Analyze, Evaluate, Create)

## Current Status

### ✅ ThinkARM Baseline (Complete)
A baseline classifier using **only ThinkARM thinking traces** (excluding final answers).

**Results on 384 samples from 4 models:**
- Models: deepseekR1, Phi4R, DeepSeek-R1-Distill-Qwen-7B, QwQ32B
- Accuracy: **67.53%**
- AUC: **0.7387**
- F1: **0.6988**
- Features: 75 (episode intensities + transition patterns + dynamics)

**Script:** `compare_frameworks_thinkarm_baseline.py`

```bash
python compare_frameworks_thinkarm_baseline.py \
    --thinkarm_label_dir data/label \
    --correctness_dir data/correct \
    --models deepseekR1,Phi4R,QwQ32B,DeepSeek-R1-Distill-Qwen-7B \
    --output_dir comparison_thinking_results
```

**Report:** `comparison_thinking_results/thinking_baseline_report.txt`

### Key Findings from ThinkARM Baseline

**Positive Contributors (Help Correctness):**
1. Read → Verify transitions (0.40)
2. Explore → Monitor transitions (0.33)
3. Explore → Read transitions (0.31)
4. Monitor frequency (0.20)
5. Planning ratio (0.18)

**Negative Contributors (Hurt Correctness):**
1. High Explore frequency (-0.67) ← **Strongest negative predictor**
2. Implement → Analyze transitions (-0.33)
3. Explore → Answer transitions (-0.32)

**Interpretation:**
- Models that explore excessively in thinking traces tend to produce incorrect answers
- Clean transitions from exploration back to reading/verification help correctness
- Good planning and verification episodes support correctness

---

### ⏳ Full BLOOM Comparison (In Progress)

To complete the full framework comparison, we need BLOOM annotations.

**Scripts Available:**
1. `convert_thinkarm_to_bloom_fixed.py` - Converts ThinkARM data to BLOOM input format
2. `annotate_bloom.py` - Annotates with BLOOM cognitive levels using Llama-3.3-70B-Instruct
3. `compare_frameworks_thinking_only.py` - Compares both frameworks when BLOOM annotations exist

**Why Not Yet Generated:**
- Requires vLLM and Llama-3.3-70B-Instruct model
- Not available in local environment
- Should be run on Leonardo HPC cluster with GPU

**How to Generate BLOOM Annotations:**

#### Step 1: Verify Raw Data
```bash
ls data/label/deepseekR1/  # Should see numbered JSON files (1.json, 2.json, etc.)
ls data/raw/deepseekR1.json  # Should have corresponding raw model outputs
```

#### Step 2: Convert to BLOOM Format
```bash
python convert_thinkarm_to_bloom_fixed.py \
    --thinkarm_dir data/label \
    --raw_dir data/raw \
    --output_dir outputs_bloom_from_thinkarm \
    --models deepseekR1,Phi4R,QwQ32B,DeepSeek-R1-Distill-Qwen-7B \
    --dataset_name math
```

**Output:** `outputs_bloom_from_thinkarm/bloom_input_*.json`
- Each file contains: question, reasoning trace, ground truth
- Preserves original ThinkARM annotations for comparison

#### Step 3: Annotate with BLOOM on Leonardo
```bash
# Load vLLM environment and run annotation
sbatch job_annotate_bloom.sh outputs_bloom_from_thinkarm "bloom_input_*.json"
```

**Output:** `outputs_bloom_annotated/bloom_input_*_math.json`
- Adds: `bloom_labels` (full trace) and `thinking_bloom_labels` (thinking only)
- Each label includes cognitive level and reasoning

#### Step 4: Run Full Comparison
Once BLOOM annotations are generated:

```bash
python compare_frameworks_thinking_only.py \
    --thinkarm_label_dir data/label \
    --bloom_annotated_dir outputs_bloom_annotated \
    --correctness_dir data/correct \
    --models deepseekR1,Phi4R,QwQ32B,DeepSeek-R1-Distill-Qwen-7B \
    --output_dir comparison_thinking_results
```

This will generate:
- ThinkARM vs BLOOM performance comparison
- Feature importance for each framework
- Combined model analysis

---

## Architecture Overview

### Data Flow

```
ThinkARM Annotated Data (data/label/)
         ↓
   [extract_thinkarm_thinking_features]
         ↓
     75 Features
   (episode ratios + transitions + dynamics)
         ↓
   ┌─────────────────────────────┐
   │   BASELINE CLASSIFIER       │
   │  (Currently Implemented)    │
   │                             │
   │  - Accuracy: 67.53%         │
   │  - AUC: 0.7387              │
   │  - F1: 0.6988               │
   └─────────────────────────────┘
         ↓
   comparison_thinking_results/
   thinking_baseline_report.txt


ThinkARM + BLOOM Annotated Data
         ↓
   ┌──────────────────┬──────────────────┐
   ↓                  ↓
ThinkARM          BLOOM
75 Features       45 Features
   ↓                  ↓
   └──────────────────┬──────────────────┘
         ↓
   ┌─────────────────────────────────────┐
   │   FULL COMPARISON CLASSIFIER        │
   │  (When BLOOM Annotations Ready)     │
   │                                     │
   │  - ThinkARM only                    │
   │  - BLOOM only                       │
   │  - Combined                         │
   └─────────────────────────────────────┘
```

### Feature Definitions

#### ThinkARM Thinking Features (75 total)
- **Global**: Total tokens in thinking traces
- **Intensity** (8 features): Ratio of each episode type
- **Transitions** (64 features): 8×8 transition matrix
- **Dynamics** (2 features): Explore and Monitor frequency

#### BLOOM Thinking Features (45 total, when available)
- **Global**: Total tokens in thinking
- **Intensity** (6 features): Ratio of each cognitive level
- **Transitions** (36 features): 6×6 transition matrix
- **Dynamics** (2 features): Evaluate and Understand frequency

---

## Files

### Classifiers
- `compare_frameworks_thinkarm_baseline.py` ✅ (Working)
- `compare_frameworks_thinking_only.py` (Requires BLOOM annotations)
- `compare_frameworks_classifier.py` (Full traces, including answers)

### Data Conversion
- `convert_thinkarm_to_bloom_fixed.py` (Convert data format)
- `annotate_bloom.py` (Run BLOOM annotation with Llama)
- `job_annotate_bloom.sh` (SLURM job for Leonardo)

### Documentation
- `BLOOM_FROM_THINKARM.md` (Detailed workflow)
- `COMPARISON_RESULTS.md` (This file)

### Results
- `comparison_thinking_results/thinking_baseline_report.txt`

---

## Next Steps

1. **To get BLOOM annotations:**
   - SSH into Leonardo HPC cluster
   - Run conversion and annotation steps above
   - Download annotated results

2. **To run full comparison:**
   - Place BLOOM annotated files in `outputs_bloom_annotated/`
   - Run `compare_frameworks_thinking_only.py`

3. **To extend analysis:**
   - Try `compare_frameworks_classifier.py` for full traces (thinking + answers)
   - Analyze per-model differences
   - Correlate ThinkARM episodes with BLOOM cognitive levels

---

## Questions & Troubleshooting

**Q: Why are BLOOM annotations not generated locally?**
A: They require vLLM + Llama-3.3-70B-Instruct, which need GPU and ~141GB model memory.

**Q: What if I only want to use ThinkARM?**
A: The baseline classifier is already complete. It achieves 67.53% accuracy and shows which thinking episode patterns predict correctness.

**Q: How do I compare frameworks?**
A: Once BLOOM annotations exist, run `compare_frameworks_thinking_only.py` to directly compare ThinkARM vs BLOOM vs Combined on the same test set.

**Q: Can I run this on a subset of data first?**
A: Yes, use `--max_samples 50` to test with fewer samples before running full dataset.

---

## Paper-Ready Results

The baseline results show:
- **ThinkARM thinking traces can predict correctness** with reasonable accuracy (67.53%)
- **Episode transitions matter more than absolute duration**: Specific transition patterns (Read→Verify) strongly support correctness
- **Excessive exploration hurts performance**: Models that spend too much time in the Explore episode tend to produce incorrect answers
- **Cognitive patterns are learnable**: The classifier identifies interpretable patterns that match intuition about problem-solving

When BLOOM annotations are ready, we'll compare whether cognitive level transitions (Remember→Understand→Apply→Analyze→Evaluate) align with episode transitions in the thinking process.
