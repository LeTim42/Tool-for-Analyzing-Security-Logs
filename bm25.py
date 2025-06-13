import numpy as np
from rank_bm25 import BM25Okapi


class BM25:
    def __init__(self, logs):
        self.logs = logs
        tokenized_logs = [log.split(" | ", 1)[-1].lower().split() for log in logs]
        self.bm25 = BM25Okapi(tokenized_logs)

    def search(self, queries, top_k, threshold=1.0):
        indices = []
        for query in queries:
            tokenized_query = query.split()
            log_scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.where(log_scores >= threshold)[0]
            if len(top_indices) > top_k:
                top_indices = np.argsort(log_scores)[-top_k:]
            indices.extend(top_indices)
        indices = np.unique(np.array(indices))
        return [self.logs[i] for i in indices]
