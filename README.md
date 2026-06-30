# M1阶段：文档切块

将 txt 原始文本文件处理成结构化数据：

1. 加载文档 —— 读取所有txt文件，每篇生成一个 Document（id由内容hash决定）
2. 定长切块 —— 按 1200 字符切块、100字符重叠，每块生成一个 TextUnit
3. 双向溯源 —— Document 记录它被切成了哪些 TextUnit，TextUnit 记录它来自哪篇 Document
4. 落盘 —— 输出 `documents.parquet` 和 `text_units.parquet`

# M2阶段：实体/关系抽取

读取 M1 阶段的产物，用 LLM 从每块文本中抽取知识图谱元素：
