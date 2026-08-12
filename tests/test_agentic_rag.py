import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bench.agentic_eval import evaluate_agentic_qa
from paperpilot.core.agentic_rag import run_agentic_rag


def _tool_call(call_id, name, args):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(args),
        ),
    )


def _response(*tool_calls, content=None):
    message = SimpleNamespace(
        content=content,
        tool_calls=list(tool_calls),
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            return _response(content="done")
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


class AgenticRagLoopTest(unittest.TestCase):
    def test_model_tool_calls_drive_retrieve_inspect_finish_loop(self):
        client = FakeClient([
            _response(_tool_call("c1", "retrieve_evidence", {
                "query": "What architecture does the model use?",
                "top_k": 3,
            })),
            _response(_tool_call("c2", "inspect_context", {})),
            _response(_tool_call("c3", "finish", {
                "reason": "the evidence answers the factual question",
            })),
        ])

        def retrieve(_query, _top_k):
            return [{
                "score": 1.0,
                "chunk": {
                    "chunk_id": "p1",
                    "chunk_type": "paragraph",
                    "section": "Model",
                    "text": "The model uses a Transformer encoder architecture.",
                },
            }]

        with patch("paperpilot.core.agentic_rag.generate_answer", return_value="final answer"):
            result = run_agentic_rag(
                "What architecture does the model use?",
                {"query_type": "factual", "evidence_need": "single_chunk"},
                retrieve,
                client,
                top_k=3,
            )

        self.assertEqual(result["answer"], "final answer")
        names = [step["name"] for step in result["trace"]]
        self.assertIn("tool:retrieve_evidence", names)
        self.assertIn("tool:inspect_context", names)
        self.assertIn("tool:finish", names)
        self.assertEqual(result["retrieved"][0]["chunk"]["chunk_id"], "p1")
        self.assertEqual(len(client.chat.completions.requests), 3)
        self.assertIn("tools", client.chat.completions.requests[0])

    def test_numeric_agent_can_extract_table_data_before_stopping(self):
        client = FakeClient([
            _response(_tool_call("c1", "retrieve_numeric_or_table", {
                "query": "Which model has the highest F1 score?",
                "top_k": 5,
            })),
            _response(_tool_call("c2", "extract_table_data", {})),
            _response(_tool_call("c3", "inspect_context", {})),
            _response(_tool_call("c4", "finish", {
                "reason": "table rows and numeric facts are available",
            })),
        ])

        def retrieve(_query, _top_k):
            return [{
                "score": 2.0,
                "chunk": {
                    "chunk_id": "t1",
                    "chunk_type": "table",
                    "section": "Results",
                    "caption": "F1 scores",
                    "text": "| Model | F1 |\n|---|---|\n| Small | 79.1 |\n| Large | 83.5 |",
                    "table_columns": ["Model", "F1"],
                    "table_rows": [["Small", "79.1"], ["Large", "83.5"]],
                },
            }]

        with patch("paperpilot.core.agentic_rag.generate_answer", return_value="Large has 83.5 F1 [1]."):
            result = run_agentic_rag(
                "Which model has the highest F1 score?",
                {
                    "query_type": "numerical",
                    "needs_calculation": True,
                    "source_preference": ["table"],
                },
                retrieve,
                client,
                top_k=5,
            )

        names = [step["name"] for step in result["trace"]]
        self.assertIn("tool:retrieve_numeric_or_table", names)
        self.assertIn("tool:extract_table_data", names)
        inspect_step = next(step for step in result["trace"] if step["name"] == "tool:inspect_context")
        self.assertGreater(inspect_step["data"]["output"]["numeric_fact_count"], 0)
        self.assertEqual(result["retrieved"][0]["chunk"]["chunk_id"], "t1")

    def test_agent_can_list_neighbor_chunks_for_retrieved_evidence(self):
        client = FakeClient([
            _response(_tool_call("c1", "retrieve_evidence", {
                "query": "language pairs",
                "top_k": 1,
            })),
            _response(_tool_call("c2", "list_neighbor_chunks", {
                "chunk_id": "p1",
                "window": 1,
            })),
            _response(_tool_call("c3", "inspect_context", {})),
            _response(_tool_call("c4", "finish", {
                "reason": "neighboring chunks complete the evidence",
            })),
        ])

        def retrieve(_query, _top_k):
            return [{
                "score": 1.0,
                "chunk": {
                    "chunk_id": "p1",
                    "chunk_type": "paragraph",
                    "section": "Experiments",
                    "text": "The experiment setup refers to the following language pairs.",
                },
            }]

        def neighbors(chunk_id, _window):
            self.assertEqual(chunk_id, "p1")
            return [{
                "score": 0.99,
                "chunk": {
                    "chunk_id": "p2",
                    "chunk_type": "paragraph",
                    "section": "Experiments",
                    "text": "The explored language pairs are English-French and English-German.",
                },
            }]

        with patch("paperpilot.core.agentic_rag.generate_answer", return_value="final answer"):
            result = run_agentic_rag(
                "What language pairs are explored?",
                {"query_type": "factual", "evidence_need": "single_chunk"},
                retrieve,
                client,
                top_k=5,
                neighbor_fn=neighbors,
            )

        names = [step["name"] for step in result["trace"]]
        self.assertIn("tool:list_neighbor_chunks", names)
        self.assertEqual([row["chunk"]["chunk_id"] for row in result["retrieved"][:2]], ["p1", "p2"])
        first_request_tools = [
            tool["function"]["name"]
            for tool in client.chat.completions.requests[0]["tools"]
        ]
        second_request_tools = [
            tool["function"]["name"]
            for tool in client.chat.completions.requests[1]["tools"]
        ]
        self.assertNotIn("list_neighbor_chunks", first_request_tools)
        self.assertIn("list_neighbor_chunks", second_request_tools)

    def test_requires_tool_calling_client(self):
        with self.assertRaises(ValueError):
            run_agentic_rag("question", {}, lambda _q, _k: [], None)

    def test_parallel_tool_calls_are_corrected_to_observation_loop(self):
        client = FakeClient([
            _response(
                _tool_call("c1", "retrieve_evidence", {
                    "reason": "need initial evidence",
                    "query": "architecture",
                    "top_k": 1,
                }),
                _tool_call("c2", "list_neighbor_chunks", {
                    "reason": "premature neighbor call",
                    "chunk_id": "p1",
                    "window": 1,
                }),
            ),
            _response(_tool_call("c3", "finish", {
                "reason": "the first observation is enough",
            })),
        ])

        def retrieve(_query, _top_k):
            return [{
                "score": 1.0,
                "chunk": {
                    "chunk_id": "p1",
                    "chunk_type": "paragraph",
                    "section": "Model",
                    "text": "The model uses a Transformer encoder architecture.",
                },
            }]

        def neighbors(_chunk_id, _window):
            raise AssertionError("second same-turn tool call should not execute")

        with patch("paperpilot.core.agentic_rag.generate_answer", return_value="final answer"):
            result = run_agentic_rag(
                "What architecture does the model use?",
                {"query_type": "factual"},
                retrieve,
                client,
                neighbor_fn=neighbors,
            )

        first_round = result["rounds"][0]
        self.assertEqual(first_round["tool_results"][0]["output"]["ok"], True)
        self.assertEqual(first_round["tool_results"][1]["output"]["ok"], False)
        self.assertIn("Only one tool call", first_round["tool_results"][1]["output"]["error"])
        self.assertFalse(client.chat.completions.requests[0].get("parallel_tool_calls", True))

    def test_extract_table_tool_is_hidden_until_table_evidence_exists(self):
        client = FakeClient([
            _response(_tool_call("c1", "retrieve_evidence", {
                "reason": "need paragraph evidence",
                "query": "datasets",
                "top_k": 1,
            })),
            _response(_tool_call("c2", "finish", {
                "reason": "paragraph evidence is enough",
            })),
        ])

        def retrieve(_query, _top_k):
            return [{
                "score": 1.0,
                "chunk": {
                    "chunk_id": "p1",
                    "chunk_type": "paragraph",
                    "section": "Data",
                    "text": "The experiments use the Europarl dataset.",
                },
            }]

        with patch("paperpilot.core.agentic_rag.generate_answer", return_value="final answer"):
            run_agentic_rag(
                "Which datasets did they use?",
                {"query_type": "factual"},
                retrieve,
                client,
            )

        first_tools = [tool["function"]["name"] for tool in client.chat.completions.requests[0]["tools"]]
        second_tools = [tool["function"]["name"] for tool in client.chat.completions.requests[1]["tools"]]
        self.assertNotIn("finish", first_tools)
        self.assertNotIn("extract_table_data", second_tools)
        self.assertIn("finish", second_tools)

    def test_rewrite_query_is_an_independent_observation_tool(self):
        client = FakeClient([
            _response(_tool_call("c1", "rewrite_query", {
                "reason": "need search variants before retrieving",
                "focus": "best F1 score",
            })),
            _response(_tool_call("c2", "retrieve_numeric_or_table", {
                "reason": "rewrite suggests metric evidence",
                "query": "best F1 score table",
                "top_k": 1,
            })),
            _response(_tool_call("c3", "finish", {
                "reason": "evidence is sufficient",
            })),
        ])

        def retrieve(_query, _top_k):
            return [{
                "score": 1.0,
                "chunk": {
                    "chunk_id": "t1",
                    "chunk_type": "paragraph",
                    "section": "Results",
                    "text": "The best F1 score is 83.5.",
                },
            }]

        with patch("paperpilot.core.agentic_rag.generate_answer", return_value="final answer"):
            result = run_agentic_rag(
                "Which model has the best F1 score?",
                {"query_type": "numerical", "needs_calculation": True},
                retrieve,
                client,
            )

        rewrite_step = next(step for step in result["trace"] if step["name"] == "tool:rewrite_query")
        self.assertIn("rewrites", rewrite_step["data"]["output"])
        self.assertIn(
            "rewrite_query",
            [tool["function"]["name"] for tool in client.chat.completions.requests[0]["tools"]],
        )

    def test_verified_candidate_answer_is_used_as_final_answer(self):
        client = FakeClient([
            _response(_tool_call("c1", "retrieve_evidence", {
                "reason": "need source evidence",
                "query": "architecture",
                "top_k": 1,
            })),
            _response(_tool_call("c2", "verify_answer_support", {
                "reason": "verify the candidate before stopping",
                "candidate_answer": "The model uses a Transformer encoder architecture [1].",
            })),
            _response(_tool_call("c3", "finish", {
                "reason": "verified answer is supported",
            })),
        ])

        def retrieve(_query, _top_k):
            return [{
                "score": 1.0,
                "chunk": {
                    "chunk_id": "p1",
                    "chunk_type": "paragraph",
                    "section": "Model",
                    "text": "The model uses a Transformer encoder architecture.",
                },
            }]

        with patch("paperpilot.core.agentic_rag.generate_answer", return_value="unverified generated answer"):
            result = run_agentic_rag(
                "What architecture does the model use?",
                {"query_type": "factual"},
                retrieve,
                client,
            )

        self.assertEqual(result["answer"], "The model uses a Transformer encoder architecture [1].")
        verify_step = next(step for step in result["trace"] if step["name"] == "tool:verify_answer_support")
        self.assertTrue(verify_step["data"]["output"]["verified"])

    def test_two_failed_verifications_force_finish_only_and_final_answer_override(self):
        client = FakeClient([
            _response(_tool_call("c1", "retrieve_evidence", {
                "reason": "need source evidence",
                "query": "architecture",
                "top_k": 1,
            })),
            _response(_tool_call("c2", "verify_answer_support", {
                "reason": "first verification",
                "candidate_answer": "The model uses a Transformer encoder. It also uses a graph decoder.",
            })),
            _response(_tool_call("c3", "verify_answer_support", {
                "reason": "second verification",
                "candidate_answer": "The model uses a Transformer encoder. It also uses a graph decoder.",
            })),
            _response(_tool_call("c4", "finish", {
                "reason": "use only supported evidence",
                "final_answer": "The model uses a Transformer encoder.",
            })),
        ])

        def retrieve(_query, _top_k):
            return [{
                "score": 1.0,
                "chunk": {
                    "chunk_id": "p1",
                    "chunk_type": "paragraph",
                    "section": "Model",
                    "text": "The model uses a Transformer encoder architecture.",
                },
            }]

        with patch("paperpilot.core.agentic_rag.generate_answer", return_value="generated answer"):
            result = run_agentic_rag(
                "What architecture does the model use?",
                {"query_type": "factual"},
                retrieve,
                client,
            )

        fourth_tools = [
            tool["function"]["name"]
            for tool in client.chat.completions.requests[3]["tools"]
        ]
        self.assertEqual(fourth_tools, ["finish"])
        self.assertEqual(result["answer"], "The model uses a Transformer encoder.")


class AgenticQaEvalTest(unittest.TestCase):
    def test_evaluates_against_gold_evidence_and_writes_rounds_jsonl(self):
        queries = [{
            "qid": "q1",
            "question": "What architecture does the model use?",
            "reference_answer": "Transformer encoder",
        }]
        qrels = {"q1": {"gold"}}

        def agent_runner(_query):
            return {
                "answer": "It uses a Transformer encoder [1].",
                "retrieved": [{
                    "score": 1.0,
                    "chunk": {
                        "chunk_id": "gold",
                        "section": "Model",
                        "text": "It uses a Transformer encoder.",
                    },
                }],
                "trace": [{"name": "tool:retrieve_evidence", "observation": "added"}],
                "rounds": [{
                    "round": 1,
                    "assistant_message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "function": {
                                "name": "retrieve_evidence",
                                "arguments": "{\"query\":\"architecture\"}",
                            },
                        }],
                    },
                    "tool_results": [{
                        "tool_name": "retrieve_evidence",
                        "output": {"added_chunk_ids": ["gold"]},
                    }],
                }],
            }

        out = Path("data/test_runs/agentic_runs_test.jsonl")
        metrics = evaluate_agentic_qa(agent_runner, queries, qrels, trace_jsonl=out)
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(metrics.n, 1)
        self.assertEqual(metrics.hit1, 1.0)
        self.assertEqual(metrics.mrr, 1.0)
        self.assertEqual(rows[0]["rounds"][0]["tool_results"][0]["output"]["added_chunk_ids"], ["gold"])


if __name__ == "__main__":
    unittest.main()
