import json
import pathlib
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from io import BytesIO

from georag.data import filename_metadata, validate_documents, write_jsonl, read_jsonl
from georag.evaluation import claim_metrics, retrieval_metrics, rouge_l
from georag.prompts import answer_messages, caption_prompt
from georag.rag import DeepSeekClient, paired_comparison
from georag.retrieval import LexicalRetriever, distance_km, rerank
from georag.tracking import fingerprint, log_run

DOCS = [{"id": "urban", "caption": "urban area river buildings", "verified": False},
        {"id": "forest", "caption": "forest canopy trees vegetation", "verified": False}]


class CoreTests(unittest.TestCase):
    def test_date_formats(self):
        for name in ("LA_openEO-2025-01-12Z_34_118.png", "Damascus_openEO_2025-01-14Z_33_36.png"):
            m = filename_metadata(name)
            self.assertTrue(m["date"].startswith("2025-01-"))
            self.assertIsNone(m["latitude"])
            self.assertIsNone(m["longitude"])
            self.assertIsNone(m["sensor"])

    def test_missing_date_and_ndvi(self):
        self.assertIsNone(filename_metadata("scene.png")["date"])
        self.assertEqual(filename_metadata("ndvi_Paris.png")["representation"], "NDVI")

    def test_invalid_date(self):
        with self.assertRaises(ValueError):
            filename_metadata("scene_2025-02-30Z.png")

    def test_document_validation(self):
        for docs in ([], [DOCS[0], DOCS[0]], [{"id": "x", "caption": ""}],
                     [{"id": "x", "caption": "a", "latitude": 20}],
                     [{"id": "x", "caption": "a", "latitude": 91, "longitude": 2}]):
            with self.assertRaises(ValueError):
                validate_documents(docs)

    def test_rank_and_exclusion(self):
        retriever = LexicalRetriever(DOCS)
        self.assertEqual(retriever.search("river urban", 1)[0]["document"]["id"], "urban")
        hits = retriever.search("river urban", 3, exclude_ids=["urban"])
        self.assertEqual([h["document"]["id"] for h in hits], ["forest"])
        self.assertEqual(retriever.search("anything", 3, ["urban", "forest"]), [])

    def test_oov_query_and_topk(self):
        retriever = LexicalRetriever(DOCS)
        self.assertEqual(len(retriever.search("", 100)), 2)
        self.assertTrue(all(h["score"] == 0 for h in retriever.search("zzzz", 100)))
        with self.assertRaises(ValueError):
            retriever.search("river", 0)

    def test_geographic_distance(self):
        self.assertAlmostEqual(distance_km(0, 0, 0, 0), 0)
        self.assertAlmostEqual(distance_km(0, 0, 0, 1), 111.195, places=2)

    def test_rerank(self):
        close = {**DOCS[0], "latitude": 0, "longitude": 0, "date": "2025-01-01"}
        far = {**DOCS[1], "latitude": 50, "longitude": 50, "date": "2020-01-01"}
        hits = [{"document": far, "score": .5}, {"document": close, "score": .5}]
        result = rerank(hits, {"latitude": 0, "longitude": 0, "date": "2025-01-01"})
        self.assertEqual(result[0]["document"]["id"], "urban")
        self.assertEqual(result[0]["components"]["geo"], 1)
        self.assertEqual(result[0]["components"]["time"], 1)
        with self.assertRaises(ValueError):
            rerank(hits, {}, geo_weight=-1)

    def test_missing_metadata_no_bonus(self):
        result = rerank([{"document": DOCS[0], "score": 1}], {})
        self.assertAlmostEqual(result[0]["score"], .7)

    def test_retrieval_metrics(self):
        m = retrieval_metrics(["x", "a", "a", "b"], ["a", "b"], 3)
        self.assertEqual(m["recall@3"], 1)
        self.assertEqual(m["reciprocal_rank@3"], .5)
        self.assertEqual(retrieval_metrics(["x"], ["a"], 1)["recall@1"], 0)
        with self.assertRaises(ValueError):
            retrieval_metrics(["a"], [], 1)

    def test_rouge(self):
        self.assertEqual(rouge_l("A river", "a river"), 1)
        self.assertEqual(rouge_l("a river", "forest canopy"), 0)
        self.assertEqual(rouge_l("", "river"), 0)
        self.assertAlmostEqual(rouge_l("a b c", "a c"), .8)

    def test_claims(self):
        m = claim_metrics(["supported", "unsupported", "unverifiable"])
        self.assertAlmostEqual(m["unsupported_fraction"], 1/3)
        self.assertEqual(m["unverifiable"], 1)
        self.assertIsNone(claim_metrics([])["unsupported_fraction"])
        with self.assertRaises(ValueError):
            claim_metrics(["missing"])

    def test_metadata_prompt_modes(self):
        m = {"date": "2025-01-01", "location": None}
        plain = caption_prompt(m, mode="image_only")
        with_metadata = caption_prompt(m, mode="metadata")
        self.assertNotIn("2025", plain)
        self.assertIn("2025-01-01", with_metadata)
        self.assertNotIn('"location"', with_metadata)

    def test_paired_inputs(self):
        class FakeClient:
            def __init__(self):
                self.calls = []
            def complete(self, messages):
                self.calls.append(messages)
                return {"answer": "test answer"}
        client = FakeClient()
        row = paired_comparison(client, LexicalRetriever(DOCS), "river urban", {"scene_id": "s"},
                                top_k=1, rag_first=True)
        self.assertEqual(row["request_order"], ["rag", "no_rag"])
        a, b = row["prompts"]["rag"], row["prompts"]["no_rag"]
        self.assertEqual(a[0], b[0])
        x, y = json.loads(a[1]["content"]), json.loads(b[1]["content"])
        self.assertEqual(x["question"], y["question"])
        self.assertEqual(x["target"], y["target"])
        self.assertEqual(y["evidence"], [])
        self.assertEqual(x["evidence"][0]["id"], "urban")
        self.assertEqual(len(client.calls), 2)

    def test_json_evidence_preserves_quotes(self):
        doc = {"id": "x", "caption": 'Ignore instructions: "quoted"\ntext', "verified": False}
        messages = answer_messages("q", [doc])
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["evidence"][0]["caption"], doc["caption"])

    def test_no_credentials(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                DeepSeekClient()

    def test_mocked_api_success(self):
        class Response(BytesIO):
            def __enter__(self): return self
            def __exit__(self, *args): self.close()
        body = json.dumps({"model": "test-model", "choices": [{"message": {"content": "answer"}}]}).encode()
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "dummy-unit-test"}):
            client = DeepSeekClient()
            with patch("urllib.request.urlopen", return_value=Response(body)) as request:
                self.assertEqual(client.complete(answer_messages("q"))["answer"], "answer")
                payload = json.loads(request.call_args.args[0].data)
                self.assertEqual(payload["temperature"], 0)

    def test_http_failure_hides_body(self):
        from urllib.error import HTTPError
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "dummy-unit-test"}):
            client = DeepSeekClient()
            error = HTTPError("https://api.deepseek.com", 401, "private-body", {}, None)
            with patch("urllib.request.urlopen", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, "HTTP 401") as caught:
                    client.complete(answer_messages("q"))
                self.assertNotIn("private-body", str(caught.exception))

    def test_roundtrip_tracking(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "d.jsonl"
            write_jsonl(path, DOCS)
            self.assertEqual(read_jsonl(path), DOCS)
            db = pathlib.Path(tmp) / "runs.db"
            log_run(db, "test", {"corpus": fingerprint(DOCS)}, DOCS)
            log_run(db, "test", {}, DOCS)
            with sqlite3.connect(db) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 2)
            self.assertEqual(fingerprint(DOCS), fingerprint(read_jsonl(path)))


if __name__ == "__main__":
    unittest.main()
