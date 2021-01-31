import pandas as pd
from collections import defaultdict
import json

df = pd.read_csv("html_viz_data.csv")

topic2hashtag = {}
for topic_id, group in df.groupby("topic_id"):
    hashtags = group.apply(lambda row: {"name": row["hashtag"], "count": str(row["hashtag_count"])}, axis=1).tolist()
    topic2hashtag[topic_id] = {"name": str(topic_id), "hashtags": hashtags, "count": str(group["topic_coverage"].mean())}


data = []
df = df[["meta_id", "meta_coverage", "topic_id"]].drop_duplicates()

for (meta_id, meta_coverage), group in df.groupby(["meta_id", "meta_coverage"]):
    if int(meta_id) == -1:
        meta_id = "undefined"
    topics = group.apply(lambda row: row["topic_id"], axis=1).tolist()

    data.append({
        "name": str(meta_id),
        "topics": [topic2hashtag[topic] for topic in topics],
        "count": str(meta_coverage)
    })
for metatopics in data:
    metatopics["topics"] = sorted(metatopics["topics"], key=lambda topic: -float(topic["count"])) 
    for topic in metatopics["topics"]:
        topic["hashtags"] = sorted(topic["hashtags"], key=lambda hashtag: -float(hashtag["count"])) 

json_object = json.dumps(data, indent = 4)

with open("src/data.js", "w") as f:
    f.write("export default")
    f.write(json_object)