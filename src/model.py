from dataclasses import dataclass, field


@dataclass
class Document:
  """一篇输入文档"""
  id: str
  title: str
  text: str
  text_unit_ids: list[str] = field(default_factory=list)


@dataclass
class TextUnit:
  """一块待分析的文本 chunk —— 图抽取的基本单位， 也是 provenance 的锚点"""
  id: str
  text: str
  document_ids: list[str] = field(default_factory=list) # provenance: 我来自哪篇文章
  n_tokens: int = 0
  human_readable_id: int = 0  # 给人看的递增序号（非主键）

# —— 以下为M2+ 占位， 展示完整 Knowledge Model 的轮廓 ——
@dataclass
class Entity:
  id: str
  title: str
  type: str = ""
  description: str = ""
  text_unit_ids: list[str] = field(default_factory=list)  # 出现在哪些 chunk


@dataclass
class Relationship:
  id: str
  source: str
  target: str
  description: str = ""
  text_unit_ids: list[str] = field(default_factory=list)

