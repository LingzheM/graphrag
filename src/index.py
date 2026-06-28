"""
M1 · 管线骨架：input/*.txt → documents.parquet + text_units.parquet（带 provenance）
------------------------------------------------
你要填 3 个 TODO。填完跑：
    python index.py
    python check_m1.py      # 全绿 = M1 通过

依赖：pip install pandas pyarrow
"""
import os
import glob
import hashlib
import pandas as pd
from model import Document, TextUnit

INPUT_DIR = "input"
OUTPUT_DIR = "output"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100


def stable_id(text: str) -> str:
    """内容哈希做 id：同样的文本永远得到同样的 id（重跑/换机器都 idempotent）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_documents() -> list[Document]:
    docs = []
    for path in sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt"))):
        text = open(path, encoding="utf-8").read()
        # TODO 1 ✅：id 由内容决定（哈希），title 用文件名，text_unit_ids 等切完块再回填
        docs.append(Document(id=stable_id(text), title=os.path.basename(path), text=text))
    return docs


def chunk_document(doc: Document) -> list[TextUnit]:
    """把一篇文档切成带 provenance 的 TextUnit 列表。"""
    units: list[TextUnit] = []
    # TODO 2 ✅：定长切块（带 overlap），每块建一个 TextUnit。
    text = doc.text
    i = 0
    while i < len(text):
        chunk = text[i:i + CHUNK_SIZE].strip()
        if chunk:
            units.append(TextUnit(
                id=stable_id(chunk),       # id 认文本本身 → idempotent
                text=chunk,
                document_ids=[doc.id],     # ← provenance：这一块来自哪篇文档
                n_tokens=len(chunk) // 4,  # 字符数粗估 token，M1 够用
            ))
        i += CHUNK_SIZE - CHUNK_OVERLAP    # 步长 < CHUNK_SIZE → 相邻块有 overlap
    return units


def run_index():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    docs = load_documents()
    all_units: list[TextUnit] = []
    for doc in docs:
        units = chunk_document(doc)
        for i, u in enumerate(units):
            u.human_readable_id = len(all_units) + i
        # TODO 3 ✅：反向连——doc 记住自己被切成了哪些 unit（双向 provenance）
        doc.text_unit_ids = [u.id for u in units]
        all_units.extend(units)

    # 落盘（这部分给好了）
    pd.DataFrame([d.__dict__ for d in docs]).to_parquet(f"{OUTPUT_DIR}/documents.parquet")
    pd.DataFrame([u.__dict__ for u in all_units]).to_parquet(f"{OUTPUT_DIR}/text_units.parquet")
    print(f"documents: {len(docs)}  text_units: {len(all_units)}  → 写入 {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_index()
