# EpiTagger

## Setup

```bash
pip install -r requirements.txt
```

## Data Organization

The data is organized into the following directories within `data/`:

- `raw/`: Raw model outputs. Each file (e.g., `DSQwen32B.json`) contains the problems and model responses.
- `ground_truth/`: Human-annotated labels. Organized by model name. Each JSON file (e.g., `QwQ32B/1.json`) corresponds to a problem and contains the reasoning steps with `human_label`.
- `label/`: Automatically generated labels. Organized by model name. Each JSON file (e.g., `DSQwen32B/1.json`) contains the reasoning steps with `sentence-category` (the assigned label) and `sentence-category-reason`.
- `correct/`: Correctness evaluation results. JSON files mapping problem indices to boolean values indicating if the answer was correct.

## Automatic Labeling

```bash
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
python -m method.label --annotate_model [ANNOTATE_MODEL] --response_model [RESPONSE_MODEL]
```

## Correctness Evaluation

```bash
python -m analysis.correctness_eval --model [MODEL] --evaluator_model [EVALUATE_MODEL]
```

## Fine-grained Analysis

### Temporal Dynamics

```bash
python -m analysis.temporal 
```

### Word Cloud Analysis

```bash
python -m analysis.word_cloud
```

### Diagnostic Analysis

```bash
python -m analysis.diagnostic 
```

### Episode-level N-gram Analysis

```bash
python -m analysis.episode_ngram_preprocess
python -m analysis.episode_ngram_discriminate [FILE1] [FILE2]

```
