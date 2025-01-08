from main import download_and_unzip, BASE_URL, build_parser
from loader import GenericDataLoader
from sentence_encoder import SentenceEmbedding

from itertools import islice

import numpy as np
from tqdm import tqdm


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def main(dataset):
    data_path = download_and_unzip(BASE_URL.format(dataset), "datasets")
    split = "dev" if dataset == "msmarco" else "test"
    corpus, query, _ = GenericDataLoader(data_folder=data_path).load(split=split)

    corpus_ids, corpus_text = [], []
    for key, val in corpus.items():
        corpus_ids.append(key)
        corpus_text.append(val["title"] + " " + val["text"])
    del corpus

    qids, query_text = [], []
    for key, val in query.items():
        qids.append(key)
        query_text.append(val)
    del query

    se = SentenceEmbedding()

    doc_sentence_embs = []
    for docs in tqdm(batched(corpus_text, 100)):
        doc_sentence_embs.extend(se.encode_docs(docs))
    np.save(f"{dataset}_doc_sentence", np.array(doc_sentence_embs), allow_pickle=False)
    del doc_sentence_embs

    query_sentence_embs = []
    for queries in tqdm(batched(query_text, 100)):
        query_sentence_embs.extend(se.encode_queries(queries))
    np.save(
        f"{dataset}_query_sentence", np.array(query_sentence_embs), allow_pickle=False
    )
    del query_sentence_embs


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args.dataset)
