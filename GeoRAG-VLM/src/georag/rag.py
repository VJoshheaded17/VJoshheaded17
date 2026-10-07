"""DeepSeek integration with environment credentials and controlled comparisons."""
import json
import os
import urllib.request
import urllib.error
from .prompts import answer_messages


class DeepSeekClient:
    def __init__(self, model=None, temperature=0.0, max_tokens=512, timeout=90):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("Set DEEPSEEK_API_KEY in your environment")
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout

    def complete(self, messages):
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        request = urllib.request.Request("https://api.deepseek.com/chat/completions",
                    data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
                    "Authorization": "Bearer " + self.api_key}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            # Never print request headers or API response bodies that could contain secrets.
            raise RuntimeError(f"DeepSeek returned HTTP {exc.code}") from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError("DeepSeek request failed or timed out") from None
        try:
            answer = result["choices"][0]["message"]["content"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError
            return {"answer": answer, "served_model": result.get("model"),
                    "usage": result.get("usage"), "response_id": result.get("id")}
        except (KeyError, IndexError, TypeError, ValueError):
            raise RuntimeError("DeepSeek returned an invalid answer structure") from None


def paired_comparison(client, retriever, question, target=None, top_k=3, exclude_ids=(), rag_first=False):
    hits = retriever.search(question, top_k=top_k, exclude_ids=exclude_ids)
    evidence = [hit["document"] for hit in hits]
    prompts = {"no_rag": answer_messages(question, target=target),
               "rag": answer_messages(question, evidence, target)}
    order = ("rag", "no_rag") if rag_first else ("no_rag", "rag")
    answers = {condition: client.complete(prompts[condition]) for condition in order}
    return {"question": question, "target": target, "retrieved_ids": [d["id"] for d in evidence],
            "retrieval_scores": [h["score"] for h in hits], "prompts": prompts, "answers": answers,
            "request_order": list(order)}
