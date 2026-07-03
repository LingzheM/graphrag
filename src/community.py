import os
import json
import hashlib
import ollama
import pandas as pd
import networkx as nx
from collections import Counter, defaultdict
from graspologic.partition import hierarchical_leiden
from model import Community, CommunityReport

CHAT_MODEL = "qwen2.5:7b"
OUTPUT_DIR = "output"
CACHE_DIR = "cache/llm"
MAX_CLUSTER_SIZE = 10
REPORT_MAX_WORDS = 200
LLM_MISSES = 0

COMMUNITY_REPORT_PROMPT = """你是信息分析助手。下面给你一个“社区”—— 一组相关实体及其关系。
写一份社区报告，让决策者快速理解这个社区讲的是什么。
严格只输出 JSON：
{{"title":"简短、尽量含代表性实体的标题","summary":"综述：核心实体、它们如何相互关联，以及关键信息"}}
总长度不超过 {max_words} 个词。

实体：
{entities}

关系：
{relationships}
"""


def stable_id(text: str) ->str:
  return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def cached_chat_json(prompt: str) -> dict:
  global LLM_MISSES
  os.makedirs(CACHE_DIR, exist_ok=True)
  path = os.path.join(CACHE_DIR, stable_id(CHAT_MODEL + prompt) + ".json")
  if os.path.exists(path):
    with open(path, encoding="utf-8") as f:
      raw = json.load(f)
  else:
    resp = ollama.chat(model=CHAT_MODEL,
                       messages=[{"role": "user", "content": prompt}],
                       format="json")
    raw = resp["message"]["content"]
    with open(path, "w", encoding="utf-8") as f:
      json.dump(raw, f)
    LLM_MISSES += 1
  try:
    return json.loads(raw)
  except Exception:
    return {}


def build_graph() -> tuple[nx.Graph, dict]:
  """从最终 entities/relationships 建图。取最大联通分量，
  也避免 Leiden 在非连通图上出错。"""
  ent = pd.read_parquet(f"{OUTPUT_DIR}/entities.parquet")
  rel = pd.read_parquet(f"{OUTPUT_DIR}/relationships.parquet")
  G = nx.Graph()
  for e in ent.itertuples(index=False):
    G.add_node(e.title, type=e.type, description=e.description)
  for r in rel.itertuples(index=False):
    if r.source in G and r.target in G:
      G.add_edge(r.source, r.target, description=r.description)
  if G.number_of_nodes() and nx.number_connected_components(G) > 1:
    lcc = max(nx.connected_components(G), key=len)
    dropped = G.number_of_nodes() - len(lcc)
    G = G.subgraph(lcc).copy()
    print(f"取最大连通分量：保留 {len(lcc)} 节点，丢弃 {dropped} 个孤立节点）")
  node_map = {e.title: e for e in ent.itertuples(index=False)}
  return G, node_map


def detect_communities(G: nx.Graph) -> list[Community]:
  """"""
  result = hierarchical_leiden(G, max_cluster_size=MAX_CLUSTER_SIZE)
  groups = defaultdict(list)
  for item in result:
    groups[(item.level, item.cluster)].append(item.node)

  communities: list[Community] = []
  for (level, cluster), node in groups.items():
    communities.append(Community(
      id=stable_id(f"{level}-{cluster}"),
      level=level,
      entity_ids=[]
    ))
  return communities


def generate_report(community: Community, G: nx.Graph) -> CommunityReport:
  entity_lines = []
  for title in community.entity_ids:
    desc = G.nodes[title].get("description", "") if title in G else ""
    entity_lines.append(f"- {title}: {desc}")
  entities_txt = "\n".join(entity_lines)

  sub = G.subgraph(community.entity_ids)
  rel_lines = [f"- {u} -{v}: {data.get('description', '')}"
               for u, v, data in sub.edges(data=True)]
  relationships_txt = "\n".join(rel_lines) if rel_lines else "（无内部关系）"

  prompt = COMMUNITY_REPORT_PROMPT.format(max_words=REPORT_MAX_WORDS,
                                          entities=entities_txt,
                                          relationships=relationships_txt)
  data = cached_chat_json(prompt)
  return CommunityReport(id=stable_id(community.id), community_id=community.id,
                         title=data.get("title", ""), summary=data.get("summary", ""),
                         full_content=data.get("summary", ""))


def run():
  G, _ = build_graph()
  if G.number_of_nodes() == 0:
    print("图是空的，先把 M2/M2b 跑出来")
    return
  
  communities = detect_communities(G)
  reports: list[CommunityReport] = []
  for c in communities:
    reports.append(generate_report(c, G))

  pd.DataFrame([c.__dict__ for c in communities]).to_parquet(f"{OUTPUT_DIR}/communities.parquet")
  pd.DataFrame([r.__dict__ for r in reports]).to_parquet(f"{OUTPUT_DIR}/community_reports.parquet")
  by_level = Counter(c.level for c in communities)
  print(f"社区：{len(communities)}（各层 level→数量：{dict(sorted(by_level.items()))}）")
  print(f"报告：{len(reports)}（LLM_MISSES={LLM_MISSES}）")


if __name__ == "__main__":
  run()