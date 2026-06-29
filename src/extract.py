
import os
import hashlib
import pandas as pd

CHAT_MODEL = "qwen2.5:7b"
OUTPUT_DIR = "output"
CACHE_DIR = "cache/llm"
ENTITY_TYPES = "PERSON, PLACE, ORGANIZATION, EVENT, OBJECT"
SAMPLE_N = 8  # 先只抽前 N 块文本

LLM_MISSES = 0  # 缓存未命中（真打了模型）的次数

# 抽取prompt：改写自官方 GRAPH_EXTRACTION_PROMPT, 输出格式从 <|> /## 分隔符改成 JSON
EXTRACT_PROMPT = """你是信息抽取器。从下面的文本中抽取实体(entities)和关系(relationships)。
只抽这些类型的实体：{entity_types}。
严格只输出 JSON，格式如下：
{{"entities":[{{"name":"实体名","type":"类型","description":"对该实体的简要描述"}}],
  "relationships":[{{"source":"实体A","target":"实体B","description":"二者为何相关"}}]}}
要求：实体名用文中称呼；relationships 的 source/target 必须是上面 entities 里出现过的 name。

文本：
{input_text}
"""


def stable_id(text: str) -> str:
  return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def cached_chat(prompt: str) -> str:
  """带磁盘缓存的LLM调用：同一个 prompt 只真正打一次模型。"""
  global LLM_MISSES
  os.makedirs(CACHE_DIR, exist_ok=True)
  key = stable_id(CHAT_MODEL + prompt)
  path = os.path.join(CACHE_DIR, key + ".json")
  # TODO 实现缓存逻辑
  

def extract_from_text(text: str, text_unit_id: str) -> tuple[list[Entity], list[RelationShip]]:
  """对一块文本抽取 entities + relationships。"""
  prompt = EXTRACT_PROMPT.format(entity_types=ENTITY_TYPES, input_text=text)
  raw = cached_chat(prompt)
  entities: list[Entity] = []
  relationships: list[Relationship] = []

  return entities, relationships


def run_extract():
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  tu = pd.read_parquet(f"{OUTPUT_DIR}/text_units.parquet").head(SAMPLE_N)
  entities: list[Entity] = []
  relationships: list[Relationship] = []

