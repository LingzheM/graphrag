
import os
import json
import hashlib
import pandas as pd
import ollama
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import Entity, Relationship

CHAT_MODEL = "qwen2.5:7b"
OUTPUT_DIR = "output"
CACHE_DIR = "cache/llm"
SUMMARIZE_MAX_WORDS = 120

LLM_MISSES = 0

SUMMARIZE_PROMPT = """你要把关于同一个对象的多条描述，合并成一条全面、连贯的描述。
要求：综合所有描述里的信息，不要遗漏；若有矛盾就消解成一个一致的说法；
用第三人称，并带上对象名字以便有上下文；总长度不超过 {max_words} 个词。

对象：{name}
描述列表：
{description_list}

输出（只给最终那一条描述）："""


def stable_id(text: str) -> str:
  return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def cached_chat_text(prompt: str) -> str:
  """带缓存的纯文本 LLM 调用（注意：这里不加 format='json', 要的是自然语言摘要）。"""
  global LLM_MISSES
  os.makedirs(CACHE_DIR, exist_ok=True)
  path = os.path.join(CACHE_DIR, stable_id(CHAT_MODEL + prompt) + ".json")
  if os.path.exists(path):
    with open(path, encoding="utf-8") as f:
      return json.load(f)
  resp = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
  content = resp["message"]["content"].strip()
  with open(path, "w", encoding="utf-8") as f:
    json.dump(content, f)
  LLM_MISSES += 1
  return content


def merge_entities(raw: pd.DataFrame) -> dict:
  """把 raw 实体按id（=stable(name+type)）合并。
  返回{id: {"id", "title", "type", "description":[...],"text_unit_ids":[...]}}。"""
  merged: dict = {}
  for row in raw.itertuples(index=False):
    b = merged.setdefault(row.id, {"id": row.id, "title": row.title, "type": row.type, "descriptions": [], "text_unit_ids": []})

    if row.description:
      b["descriptions"].append(row.description)
    b["text_unit_ids"].extend(row.text_unit_ids)
  return merged


def summarize(name: str, descriptions: list[str]) -> str:
  descriptions = [d for d in descriptions if d]
  if len(descriptions) <= 1:
    return descriptions[0] if descriptions else ""
  return cached_chat_text(SUMMARIZE_PROMPT.format(
    name=name, max_words=SUMMARIZE_MAX_WORDS,
    description_list="\n- ".join(descriptions)))
  

def merge_relationships(raw: pd.DataFrame) -> list[Relationship]:
  """关系按 id（stable_id(source+target)）合并，再摘要描述。"""
  buckets: dict = {}
  for r in raw.itertuples(index=False):
    b = buckets.setdefault(r.id, {"id": r.id, "source": r.source, "target": r.target,
                                  "descriptions": [], "text_unit_ids": []})
    if r.description:
      b["descriptions"].append(r.description)
    b["text_unit_ids"].extend(r.text_unit_ids)
  out = []
  for b in buckets.values():
    out.append(Relationship(
      id=b["id"], source=b["source"], target=b["target"],
      description=summarize(f"{b['source']} - {b['target']}", b["descriptions"]),
      text_unit_ids=list(dict.fromkeys(b["text_unit_ids"])),
    ))
  return out


def run_merge():
  raw_e = pd.read_parquet(f"{OUTPUT_DIR}/entities_raw.parquet")
  raw_r = pd.read_parquet(f"{OUTPUT_DIR}/relationships_raw.parquet")

  merged = merge_entities(raw_e)
  entities: list[Entity] = []
  for b in merged.values():
    entities.append(Entity(
      id=b["id"], title=b["title"], type=b["type"],
      description=summarize(b["title"], b["descriptions"]),
      text_unit_ids=list(dict.fromkeys(b["text_unit_ids"]))
    ))
  relationships = merge_relationships(raw_r)

  pd.DataFrame([e.__dict__ for e in entities]).to_parquet(f"{OUTPUT_DIR}/entities.parquet")
  pd.DataFrame([r.__dict__ for r in relationships]).to_parquet(f"{OUTPUT_DIR}/relationships.parquet")

  G = nx.Graph()
  for e in entities:
    G.add_node(e.title, type=e.type)
  for r in relationships:
    if r.source in G and r.target in G:
      G.add_edge(r.source, r.target)
  print(f"合并后：entities={len(entities)}（raw {len(raw_e)}）"
        f"relationships={len(relationships)}（raw {len(raw_r)}）（LLM_MISSES={LLM_MISSES}）")
  print(f"图：{G.number_of_nodes()} 节点，{G.number_of_edges()} 边")

  plt.figure(figsize=(12, 9))
  pos = nx.spring_layout(G, seed=42)
  nx.draw(G, pos, with_labels=True, node_size=600, font_size=7, node_color="#9ec5e8", edge_color="#bbb")
  plt.savefig(f"{OUTPUT_DIR}/graph.png", dpi=150, bbox_inches="tight")
  print(f"图已保存 → {OUTPUT_DIR}/graph.png")

if __name__  == "__main__":
  run_merge()