from main import PgClient, BASE_URL, logger, download_and_unzip
from loader import GenericDataLoader

from pathlib import Path
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    dataset = "quora"
    data_path = download_and_unzip(BASE_URL.format(dataset), "datasets")
    corpus, query, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    num_doc = len(corpus)

    client = PgClient(
        "postgresql://postgres:postgres@172.17.0.1:5432/", dataset, num_doc
    )

    corpus_ids, corpus_text = [], []
    for key, val in corpus.items():
        corpus_ids.append(key)
        corpus_text.append(val["title"] + " " + val["text"])
    del corpus

    doc_emb = np.load(Path("datasets") / dataset / "doc_sentence_emb.npy")

    for cid, text, emb in tqdm(zip(corpus_ids, corpus_text, doc_emb)):
        embs = [e for e in client.token_encoder.encode_doc(text)]
        client.conn.execute(
            f"INSERT INTO {dataset}_corpus (id, text, emb, embs) VALUES (%s, %s, %s, %s)",
            (cid, text, emb, embs),
        )

    qids, query_text = [], []
    for key, val in query.items():
        qids.append(key)
        query_text.append(val)
    del query

    logger.info("Corpus: %d, query: %d", num_doc, len(qids))

    query_emb = np.load(Path("datasets") / dataset / "query_sentence_emb.npy")

    for qid, text, emb in tqdm(zip(qids, query_text, query_emb)):
        embs = [e for e in client.token_encoder.encode_query(text)]
        client.conn.execute(
            f"INSERT INTO {dataset}_query (id, text, emb, embs) VALUES (%s, %s, %s, %s)",
            (qid, text, emb, embs),
        )
