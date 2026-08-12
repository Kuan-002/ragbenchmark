# PaperPilot Interview Demo Script

## Goal

Show the evolution from retrieval benchmark to a traceable evidence-grounded
RAG product.

## Setup

```bash
python start.py
```

Open:

```text
http://localhost:8000
```

Use a committed demo paper so the demo does not depend on GROBID or network
availability.

## Demo Flow

1. Load a demo paper.

   Explain that the app indexes paragraph chunks and table chunks separately.

2. Ask a factual question.

   Example:

   ```text
   What architecture does the model use?
   ```

   Show the answer and cited sources. Explain that this is the base RAG path.

3. Ask a numerical/table question.

   ```text
   Which model has the highest F1 score?
   ```

   Show the cited table or numeric source. Explain that query routing can
   prefer table evidence for numerical questions.

4. Ask a comparison question.

   ```text
   How does this model compare with the baseline?
   ```

   Show that the QueryPlan asks for two-sided evidence and the answer is
   cautious when one side is missing.

5. Submit negative feedback.

   Pick one reason, such as citation mismatch or numeric/table error. Explain
   that feedback is written to `data/feedback.jsonl` and can be aggregated into
   the same badcase taxonomy used in benchmark analysis.

## Closing Pitch

PaperPilot started as a RAG retrieval benchmark, then became a product loop:
measure retrieval failures, encode the fixes into query routing and retrieval
strategies, and collect feedback for the next iteration.
