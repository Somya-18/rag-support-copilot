from .providers import ModelProvider
from .retrieval import Retriever, SearchHit

ANSWER_PROMPT = """Answer the question using only the supplied evidence chunks. \
Each chunk is numbered; end every factual claim with its citation like [1] or [2]. \
If the evidence does not contain enough information to answer, say so clearly. \
Return {answer}."""


def _build_contexts(hits: list[SearchHit]) -> list[dict]:
    return [
        {
            "citation": i,
            "title": hit.title,
            "heading": " > ".join(hit.heading_path) if hit.heading_path else "",
            "content": hit.content[:1500],
        }
        for i, hit in enumerate(hits, start=1)
    ]


class RagPipeline:
    def __init__(self, provider: ModelProvider, retriever: Retriever):
        self.provider = provider
        self.retriever = retriever

    def query(self, question: str, product_version: str = "latest") -> dict:
        hits = self.retriever.search(question, product_version=product_version)

        if not hits:
            return {
                "answer": "No relevant content found in the ingested documents for this question.",
                "sources": [],
                "hit_count": 0,
            }

        contexts = _build_contexts(hits)
        result = self.provider.structured(
            ANSWER_PROMPT,
            {"question": question, "evidence": contexts},
        )
        answer = result.data.get("answer", "").strip()

        cited_indices = {i for i, _ in enumerate(hits, start=1) if f"[{i}]" in answer}
        sources = [
            {
                "citation": i,
                "title": hit.title,
                "heading": " > ".join(hit.heading_path) if hit.heading_path else "",
                "excerpt": hit.content[:300],
                "score": round(hit.fused_score, 4),
                "url": hit.canonical_url,
            }
            for i, hit in enumerate(hits, start=1)
            if i in cited_indices
        ]

        return {"answer": answer, "sources": sources, "hit_count": len(hits)}
