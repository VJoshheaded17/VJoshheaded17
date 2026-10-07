"""Run the local demo, vector retrieval, captions, and paired LLM experiments."""
import argparse
import json
from pathlib import Path
from .data import read_jsonl, write_jsonl, validate_documents
from .evaluation import retrieval_metrics, rouge_l, bleu
from .retrieval import LexicalRetriever
from .tracking import fingerprint, log_run


def make_retriever(args, documents):
    if args.backend == "lexical":
        if args.representation != "text":
            raise ValueError("Lexical retrieval only supports text")
        return LexicalRetriever(documents)
    from .embeddings import MiniLMEncoder, CLIPEncoder
    from .vector_store import VectorRetriever
    encoder = CLIPEncoder() if args.encoder == "clip" else MiniLMEncoder()
    return VectorRetriever(documents, encoder, args.backend, args.representation, args.image_root)


def retrieval_options(parser):
    parser.add_argument("--corpus", default="data/sample/toy_corpus.jsonl")
    parser.add_argument("--backend", choices=["lexical", "faiss", "qdrant"], default="lexical")
    parser.add_argument("--encoder", choices=["minilm", "clip"], default="minilm")
    parser.add_argument("--representation", choices=["text", "image"], default="text")
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--top-k", type=int, default=3)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    demo = subs.add_parser("demo", help="Offline retrieval evaluation on synthetic records")
    retrieval_options(demo)
    demo.add_argument("--queries", default="data/sample/toy_queries.jsonl")
    demo.add_argument("--output", default="results/runs/toy_retrieval.json")
    search = subs.add_parser("retrieve")
    retrieval_options(search)
    search.add_argument("question")
    compare = subs.add_parser("compare", help="Makes two paid API calls per query")
    retrieval_options(compare)
    compare.add_argument("--queries", required=True)
    compare.add_argument("--output", default="results/runs/paired_answers.jsonl")
    compare.add_argument("--db", default="results/runs/experiments.db")
    compare.add_argument("--bleu", action="store_true")
    cap = subs.add_parser("caption")
    cap.add_argument("--manifest", required=True)
    cap.add_argument("--image-root", default=".")
    cap.add_argument("--mode", choices=["image_only", "metadata", "legacy_tags"], default="metadata")
    cap.add_argument("--output", default="results/runs/captions.jsonl")
    cap.add_argument("--db", default="results/runs/experiments.db")
    cap.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    retriever = None
    try:
        if args.command == "caption":
            from .caption import KosmosCaptioner
            records = read_jsonl(args.manifest)
            model = KosmosCaptioner(seed=args.seed)
            results = []
            for scene in records:
                result = model.describe(Path(args.image_root) / scene["image_path"], scene, args.mode,
                                        scene.get("question", "Describe the visible land-cover features."))
                results.append({**scene, **result})
            validate_documents(results)
            write_jsonl(args.output, results)
            log_run(args.db, "caption", {"seed": args.seed, "mode": args.mode,
                    "manifest_sha256": fingerprint(records), "model": model.model_name}, results)
            print(f"Saved {len(results)} captions to {args.output}")
            return
        documents = validate_documents(read_jsonl(args.corpus))
        queries = read_jsonl(args.queries) if args.command in {"demo", "compare"} else None
        if queries is not None and not queries:
            raise ValueError("Query set is empty")
        if args.command == "compare":
            # Validate all evaluation inputs before initiating paid requests.
            known = {d["id"] for d in documents}
            for q in queries:
                if not q.get("id") or not q.get("question") or not q.get("reference", "").strip():
                    raise ValueError("Each comparison needs id, question, and a human reference")
                if q["reference"].startswith("REPLACE"):
                    raise ValueError("Replace the example reference with a human-verified answer")
                if not set(q.get("exclude_ids", ())).issubset(known):
                    raise ValueError("Unknown exclusion document id")
            if args.bleu:
                import nltk  # Check optional dependency before any paid call.
            from .rag import DeepSeekClient, paired_comparison
            client = DeepSeekClient()
        retriever = make_retriever(args, documents)
        if args.command == "retrieve":
            print(json.dumps(retriever.search(args.question, args.top_k), indent=2))
        elif args.command == "demo":
            rows = []
            known = {d["id"] for d in documents}
            for q in queries:
                if not set(q["relevant_ids"]).issubset(known):
                    raise ValueError("Relevance label names an unknown document")
                hits = retriever.search(q["question"], args.top_k, q.get("exclude_ids", ()))
                ids = [h["document"]["id"] for h in hits]
                rows.append({"query_id": q["id"], "retrieved_ids": ids,
                             **retrieval_metrics(ids, q["relevant_ids"], args.top_k)})
            keys = [f"recall@{args.top_k}", f"reciprocal_rank@{args.top_k}"]
            result = {"dataset": "synthetic toy fixtures; not satellite validation", "backend": args.backend,
                      "encoder": args.encoder if args.backend != "lexical" else None,
                      "representation": args.representation, "corpus_sha256": fingerprint(documents),
                      "queries_sha256": fingerprint(queries), "queries": rows,
                      "mean_metrics": {k: sum(r[k] for r in rows) / len(rows) for k in keys}}
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result["mean_metrics"], indent=2))
            print(f"Saved retrieval fixture results to {args.output}")
        else:
            config = {"corpus_sha256": fingerprint(documents), "queries_sha256": fingerprint(queries),
                      "backend": args.backend, "encoder": args.encoder if args.backend != "lexical" else None,
                      "representation": args.representation, "top_k": args.top_k, "model": client.model,
                      "temperature": client.temperature, "max_tokens": client.max_tokens,
                      "note": "temperature=0 does not guarantee deterministic hosted responses"}
            results = []
            for i, q in enumerate(queries):
                row = paired_comparison(client, retriever, q["question"], q.get("target"), args.top_k,
                                        q.get("exclude_ids", ()), rag_first=bool(i % 2))
                row.update(query_id=q["id"], reference=q["reference"], config=config)
                row["metrics"] = {condition: {"rouge_l_f1": rouge_l(q["reference"], answer["answer"])}
                                  for condition, answer in row["answers"].items()}
                if args.bleu:
                    for condition, answer in row["answers"].items():
                        row["metrics"][condition]["bleu4_smoothed"] = bleu(q["reference"], answer["answer"])
                results.append(row)
                # Keep a checkpoint after each successful pair.
                write_jsonl(args.output, results)
                log_run(args.db, "paired_rag", config, [row])
            print(f"Saved {len(results)} paired comparisons to {args.output}")
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    finally:
        if retriever is not None and hasattr(retriever, "close"):
            retriever.close()


if __name__ == "__main__":
    main()
