"""Verify bounded prompts, retry on genuine LiteRT input-length errors,
and preservation of existing independent-agent reviews.
No network requests or local Gemma inference required.
"""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error

MODULE = Path(__file__).with_name("run_litert_agents.py")
spec = importlib.util.spec_from_file_location("litert_review_runner", MODULE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class TestAgentRunner(unittest.TestCase):
    def setUp(self):
        self.task = {
            "id": "06", "specialty": "Quantization",
            "assignment": "Review tensor per-channel metadata accurately.",
            "sources": ["native/litert_acfq_shadow.h",
                        "native/gemma_int4_kernel.c"],
            "required_output": "Evidence; uncertainty; one falsifiable check."
        }

    def test_all_sixteen_missions_fit_character_budget(self):
        jobs = json.loads(
            (MODULE.parent / "litert_agent_missions.json").read_text()
        )["missions"]
        self.assertEqual(len(jobs), 16)
        for task in jobs:
            prompt, hashes = runner.prompt_for(task)
            self.assertLessEqual(len(prompt), runner.PROMPT_CHARS_LIMIT)
            self.assertEqual(len(hashes), len(task["sources"]))
        self.assertEqual(jobs[11]["sources"][0].split("@")[1], "1120:1212")

    def test_retry_only_for_context_failure(self):
        length_body = (
            b"RuntimeError: INVALID_ARGUMENT: Input token ids are too long. "
            b"Exceeding the maximum number of tokens allowed: 4196 >= 4096"
        )
        def bad_request():
            return urllib.error.HTTPError("http://127.0.0.1:9379", 500,
                                          "backend rejected", None,
                                          io.BytesIO(length_body))
        good = {"choices": [{"message": {"content": "BESLUT: NEEDS_EVIDENCE"}}]}
        mock_calls = [bad_request(), good]
        with mock.patch.object(runner, "http_json", side_effect=mock_calls) as mocked:
            resp, hashes, digest, limit = runner.chat_with_retry(
                self.task, "gemma4-e2b", "http://127.0.0.1:9379", 600)
        self.assertEqual(resp, good)
        self.assertEqual(limit, 750)
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(len(digest), 64)

    def test_existing_reports_survive_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            existing = out_dir / "mission-06.md"
            existing.write_text("prior review", encoding="utf8")
            missions = {"model": "gemma4-e2b", "missions": [self.task]}
            def respond(url, **kwargs):
                if url.endswith("/v1/models"):
                    return {"data": [{"id": "gemma4-e2b"}]}
                self.fail("Unexpected chat call: existing report must be skipped")
            with mock.patch.object(runner, "http_json", side_effect=respond):
                rc = runner.run(missions, "http://127.0.0.1:9379",
                                out_dir, 600)
            self.assertEqual(rc, 0)
            self.assertEqual(existing.read_text(), "prior review")


if __name__ == "__main__":
    unittest.main()
