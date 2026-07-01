"""
M1 · GraphRAG Knowledge Model（精简版）
------------------------------------------------
这是全管线读写的『契约』(contract)：每个 phase 都按这些类型读入、写出。
也是我们这门课的术语表 (glossary)——以后所有 milestone 都遵守这里的命名。

M1 只真正构造 Document / TextUnit；其余类型先留作占位，
让你看清完整轮廓，M2（Entity/Relationship）、M3（Community/CommunityReport）再填。
"""
from dataclasses import dataclass, field


@dataclass
class Document:
    """一篇输入文档（一个 .txt 文件 / CSV 一行）。"""
    id: str
    title: str
    text: str
    text_unit_ids: list[str] = field(default_factory=list)  # 我被切成了哪些 TextUnit


@dataclass
class TextUnit:
    """一块待分析的文本 chunk —— 图抽取的基本单位，也是 provenance 的锚点。"""
    id: str
    text: str
    document_ids: list[str] = field(default_factory=list)    # provenance: 我来自哪篇文档
    n_tokens: int = 0
    human_readable_id: int = 0                               # 给人看的递增序号（非主键）


# —— 以下为 M2+ 占位，展示完整 Knowledge Model 的轮廓 ——

@dataclass
class Entity:            # M2：LLM 从 TextUnit 抽出的实体（人/地/物/事件…）
    id: str
    title: str
    type: str = ""
    description: str = ""
    text_unit_ids: list[str] = field(default_factory=list)   # 我出现在哪些 chunk → 可溯源


@dataclass
class Relationship:      # M2：两个实体之间的关系
    id: str
    source: str
    target: str
    description: str = ""
    text_unit_ids: list[str] = field(default_factory=list)


@dataclass
class Community:         # M3：对实体图做 Leiden 层级聚类得到的簇
    id: str
    level: int = 0
    entity_ids: list[str] = field(default_factory=list)


@dataclass
class CommunityReport:   # M3：每个社区由 LLM 写的摘要报告（global search 的源材料）
    id: str
    community_id: str
    title: str = ""
    summary: str = ""
    full_content: str = ""
