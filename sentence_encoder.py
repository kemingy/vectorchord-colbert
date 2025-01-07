from __future__ import annotations

from sentence_transformers import SentenceTransformer


class SentenceEmbedding:
    def __init__(self):
        self.model = SentenceTransformer(
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True
        )
        # In case you want to reduce the maximum length:
        self.model.max_seq_length = 8192

    def encode_docs(self, documents: list[str]):
        return self.model.encode(documents)

    def encode_doc(self, doc: str):
        return self.model.encode([doc])[0]

    def encode_queries(self, queries: list[str]):
        return self.model.encode(queries, prompt_name="query")

    def encode_query(self, query: str):
        return self.model.encode([query], prompt_name="query")[0]


if __name__ == "__main__":
    se = SentenceEmbedding()
    text = "the quick brown fox jumps over the lazy dog"
    doc_emb = se.encode_doc(text)
    query_emb = se.encode_query(text)
    print(doc_emb.shape, query_emb.shape)
    doc_emb.dump("doc_sentence_emb.np")
    query_emb.dump("query_sentence_emb.np")

