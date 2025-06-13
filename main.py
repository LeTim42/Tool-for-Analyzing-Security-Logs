import os
import json
import pandas as pd
from pandasgui import show

from bm25 import BM25
from llm import LLM

with open("config\\config.json", 'r') as file:
    config = json.load(file)

with open("config\\queries.json", 'r') as file:
    queries = json.load(file)

filtered_logs = []
for config_log in config["logs"]:
    logs = []
    log_dir = config_log["path"]
    for filename in os.listdir(log_dir):
        file_path = os.path.join(log_dir, filename)
        with open(file_path, 'r') as file:
            logs.extend([f"File '{file_path}' | {line}" for line in file.readlines()])
    bm25 = BM25(logs)
    cur_queries = [queries[threat] for threat in config_log["threats"]]
    filtered_logs.extend(bm25.search(cur_queries, config_log["max_lines_per_query"]))
filtered_logs = ''.join(filtered_logs)
print(filtered_logs)

llm = LLM(config["model"], config["context_size"])
report = llm.analyze_logs(filtered_logs)

df = pd.DataFrame(report)
df.drop_duplicates(inplace=True)
df["datetime"] = pd.to_datetime(df["datetime"], format="%b %d %H:%M:%S")
df.sort_values("datetime", inplace=True)

gui = show(df)
