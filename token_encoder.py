from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint


class TokenEmbedding:
    def __init__(self):
        self.config = ColBERTConfig(doc_maxlen=220, query_maxlen=32)
        self.checkpoint = Checkpoint(
            "colbert-ir/colbertv2.0", colbert_config=self.config, verbose=0
        )

    def encode_doc(self, doc: str):
        return self.checkpoint.docFromText([doc], keep_dims=False)[0].numpy()

    def encode_docs(self, documents: list[str]):
        return self.checkpoint.docFromText(documents, keep_dims=False)

    def encode_query(self, query: str):
        return self.checkpoint.queryFromText([query])[0].numpy()

    def encode_queries(self, queries: list[str]):
        return self.checkpoint.queryFromText(queries)


if __name__ == "__main__":
    te = TokenEmbedding()
    text = "the quick brown fox jumps over the lazy dog"
    doc_emb = te.encode_doc(text)
    query_emb = te.encode_query(text)
    print(doc_emb.shape, query_emb.shape)
    doc_emb.dump("doc_token_emb.np")
    query_emb.dump("query_token_emb.np")

