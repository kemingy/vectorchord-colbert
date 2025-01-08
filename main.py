from loader import GenericDataLoader
from evaluate import EvaluateRetrieval
from sentence_encoder import SentenceEmbedding
from token_encoder import TokenEmbedding

import argparse
import os
import logging
import json
from time import perf_counter
import zipfile
import requests
from tqdm.autonotebook import tqdm
import psycopg
from pgvector.psycopg import register_vector


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(process)d - %(levelname)s - %(filename)s:%(lineno)s - %(message)s"
)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


def download_and_unzip(url: str, out_dir: str, chunk_size: int = 1024) -> str:
    os.makedirs(out_dir, exist_ok=True)
    dataset = url.split("/")[-1]
    zip_file = os.path.join(out_dir, dataset)

    if not os.path.isfile(zip_file):
        logger.info("Downloading {} ...".format(dataset))
        download_url(url, zip_file, chunk_size)

    if not os.path.isdir(zip_file.replace(".zip", "")):
        logger.info("Unzipping {} ...".format(dataset))
        unzip(zip_file, out_dir)

    return os.path.join(out_dir, dataset.replace(".zip", ""))


def unzip(zip_file: str, out_dir: str):
    zip_ = zipfile.ZipFile(zip_file, "r")
    zip_.extractall(path=out_dir)
    zip_.close()


def download_url(url: str, save_path: str, chunk_size: int = 1024):
    """Download url with progress bar using tqdm
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads

    Args:
        url (str): downloadable url
        save_path (str): local path to save the downloaded file
        chunk_size (int, optional): chunking of files. Defaults to 1024.
    """
    r = requests.get(url, stream=True)
    total = int(r.headers.get("Content-Length", 0))
    with (
        open(save_path, "wb") as fd,
        tqdm(
            desc=save_path,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=chunk_size,
        ) as bar,
    ):
        for data in r.iter_content(chunk_size=chunk_size):
            size = fd.write(data)
            bar.update(size)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", default="fiqa", choices=["fiqa", "msmarco", "quora"]
    )
    parser.add_argument("-k", "--topk", default=10, type=int)
    parser.add_argument("-s", "--save_dir", default="datasets")
    return parser


class PgClient:
    def __init__(self, url, dataset, num):
        self.dataset = dataset
        self.num = num
        self.conn = psycopg.connect(url, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vchord CASCADE;")
        register_vector(self.conn)
        self.conn.execute("""
            CREATE OR REPLACE FUNCTION max_sim(document vector[], query vector[]) RETURNS double precision AS $$
            WITH queries AS (
                SELECT row_number() OVER () AS query_number, * FROM (SELECT unnest(query) AS query)
            ),
            documents AS (
                SELECT unnest(document) AS document
            ),
            similarities AS (
                SELECT query_number, document <=> query AS similarity FROM queries CROSS JOIN documents
            ),
            max_similarities AS (
                SELECT MAX(similarity) AS max_similarity FROM similarities GROUP BY query_number
            )
            SELECT SUM(max_similarity) FROM max_similarities
            $$ LANGUAGE SQL
        """)
        self.token_encoder = TokenEmbedding()
        self.sentence_encoder = SentenceEmbedding()

    def create(self):
        with self.conn.cursor() as cursor:
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {self.dataset}_corpus (id TEXT, text TEXT, emb vector(1536), embs vector(128)[]);"
            )
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {self.dataset}_query (id TEXT, text TEXT, emb vector(1536), embs vector(128)[]);"
            )

    def insert(self, doc_ids, docs, qids, queries):
        start_time = perf_counter()

        with self.conn.cursor() as cursor:
            # with cursor.copy(
            #     f"COPY {self.dataset}_corpus (id, text, emb, embs) FROM STDIN WITH (FORMAT BINARY)"
            # ) as copy:
            #     for did, doc in tqdm(zip(doc_ids, docs), desc="copy corpus"):
            #         emb = self.sentence_encoder.encode_doc(doc)
            #         embs = self.token_encoder.encode_doc(doc)
            #         copy.write_row(
            #             (
            #                 did,
            #                 doc,
            #                 emb,
            #                 embs,
            #             )
            #         )
            for did, doc in tqdm(zip(doc_ids, docs), desc="insert corpus"):
                emb = self.sentence_encoder.encode_doc(doc)
                embs = self.token_encoder.encode_doc(doc)
                cursor.execute(
                    f"INSERT INTO {self.dataset}_corpus (id, text, emb, embs) VALUES (%s, %s, %s, %s)",
                    (did, doc, emb, [e for e in embs]),
                )

            for qid, query in tqdm(zip(qids, queries), desc="insert query"):
                emb = self.sentence_encoder.encode_query(query)
                embs = self.token_encoder.encode_query(query)
                cursor.execute(
                    f"INSERT INTO {self.dataset}_query (id, text, emb, embs) VALUES (%s, %s, %s, %s)",
                    (qid, query, emb, [e for e in embs]),
                )

            # with cursor.copy(
            #     f"COPY {self.dataset}_query (id, text, emb, embs)"
            # ) as copy:
            #     for qid, query in tqdm(zip(qids, queries), desc="copy query"):
            #         copy.write_row(
            #             (
            #                 qid,
            #                 query,
            #                 self.sentence_encoder.encode_query(query).tolist(),
            #                 self.token_encoder.encode_query(query).tolist(),
            #             )
            #         )

        logger.info(
            "insert %s in %f seconds", self.dataset, perf_counter() - start_time
        )

    def index(self, workers: int):
        start_time = perf_counter()
        centroids = min(4 * int(self.num**0.5), self.num // 40)
        ivf_config = f"""
        residual_quantization = true
        [build.internal]
        lists = [{centroids}]
        build_threads = {workers}
        spherical_centroids = false
        """
        with self.conn.cursor() as cursor:
            cursor.execute(f"SET max_parallel_maintenance_workers TO {workers}")
            cursor.execute(f"SET max_parallel_workers TO {workers}")
            cursor.execute(
                f"CREATE INDEX {self.dataset}_rabitq ON {self.dataset}_corpus USING vchordrq (emb vector_l2_ops) WITH (options = $${ivf_config}$$)"
            )

        logger.info("build index takes %f seconds", perf_counter() - start_time)

    def query(self, topk: int):
        start_time = perf_counter()
        with self.conn.cursor() as cursor:
            cursor.execute(
                f"select q.id as qid, c.id, c.score from {self.dataset}_query q, lateral ("
                f"select id, {self.dataset}_corpus.emb <-> q.emb as score from "
                f"{self.dataset}_corpus order by score limit {topk}) c;"
            )
            res = cursor.fetchall()
        logger.info("query takes %f seconds", perf_counter() - start_time)
        return res

    def rerank(self, results: dict, topk: int):
        start_time = perf_counter()
        ans = {}
        with self.conn.cursor() as cursor:
            for qid, recalls in results.items():
                cursor.execute(
                    "SELECT c.id, max_sim(c.embs, "
                    f"(SELECT q.embs from {self.dataset}_query q WHERE q.id = %s)"
                    f") AS score FROM {self.dataset}_corpus c WHERE id = ANY(%s) "
                    f"ORDER BY score DESC LIMIT {topk}",
                    (qid, list(recalls.keys())),
                )
                ans[qid] = dict(cursor.fetchall())

        logger.info("rerank takes %f seconds", perf_counter() - start_time)
        return ans


def main(dataset, topk, save_dir):
    data_path = download_and_unzip(BASE_URL.format(dataset), save_dir)
    split = "dev" if dataset == "msmarco" else "test"
    corpus, query, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    num_doc = len(corpus)

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

    logger.info("Corpus: %d, query: %d", num_doc, len(qids))

    client = PgClient(
        "postgresql://postgres:postgres@172.17.0.1:5432/", dataset, num_doc
    )
    client.create()
    client.insert(corpus_ids, corpus_text, qids, query_text)
    client.index(int(len(os.sched_getaffinity(0)) * 0.8))
    results = client.query(topk)

    format_results = {}
    for qid, cid, score in results:
        key = str(qid)
        if key not in format_results:
            format_results[key] = {}
        format_results[key][str(cid)] = float(score)

    format_results = client.rerank(format_results, topk)

    os.makedirs("results", exist_ok=True)
    with open(f"results/vectorchord_{dataset}.json", "w") as f:
        json.dump(format_results, f, indent=2)

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, format_results, [1, 10, 100, 1000]
    )
    logger.info("NDCG: %s", ndcg)
    logger.info("Recall: %s", recall)
    logger.info("Precision: %s", precision)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    logger.info(args)
    main(**vars(args))
