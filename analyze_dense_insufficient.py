import json
import random

with open("data/results_latest.json") as f:
    data = json.load(f)

dense_results = data["Dense"]["results"]
insufficient = [r for r in dense_results if r.get("sufficiency") == "INSUFFICIENT"]
print(f"Dense INSUFFICIENT total: {len(insufficient)}\n")

random.seed(42)
sampled = random.sample(insufficient, min(5, len(insufficient)))

for i, r in enumerate(sampled):
    query_paper = r["qid"].split("__")[0]
    print(f"=== Case {i+1} ===")
    print(f"Question   : {r['question']}")
    print(f"Query paper: {query_paper}")
    print(f"NDCG@5     : {r['ndcg5']:.4f}")
    print()

    print("Gold chunks:")
    for cid in r["gold_ids"]:
        gold_paper = cid.split("__")[0]
        tag = "SAME" if gold_paper == query_paper else "DIFF"
        print(f"  [{tag}] {cid}")

    print()
    print("Retrieved chunks (top-5):")
    for rank, cid in enumerate(r["retrieved_ids"]):
        ret_paper = cid.split("__")[0]
        in_gold = "★" if cid in r["gold_ids"] else " "
        tag = "SAME-paper" if ret_paper == query_paper else "DIFF-paper"
        print(f"  {in_gold} Rank{rank+1} [{tag}] {cid}")
    print()
