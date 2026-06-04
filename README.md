# PaperPilot

**EN** | [中文](#中文)

A retrieval-augmented question-answering system for academic papers, with a systematic benchmarking framework to evaluate and compare retrieval strategies.

---

## Features

- **PDF / arXiv ingestion** — parse papers via GROBID into paragraph and table chunks
- **Table-aware retrieval** — dual-field indexing (`index_text` for BM25, Markdown for LLM); table-specific BM25 sub-index for numerical and comparison queries
- **Query-type routing** — five-class classifier (numerical / comparison / why_how / methodology / factual) drives differentiated retrieval behaviour
- **BM25 → Cross-encoder pipeline** — benchmark-validated: dense retrieval performs near random baseline within single papers; removed in favour of BM25 + CE reranking
- **Cross-table merging** — comparison queries trigger LLM-based routing; top-2 tables are merged into a joint context for comparative reasoning
- **Demo mode** — pre-parsed papers ship as JSON; no GROBID required at runtime for demo users
- **i18n** — EN / 中文 toggle, preference persisted in `localStorage`
- **Benchmark suite** — evaluates BM25 / Dense / RRF / CE on QASPER with NDCG@5, Recall@5, MRR; two-pass LLM-as-Judge for sufficiency and failure taxonomy

---

## Quick Start

### Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| Python | ≥ 3.10 | |
| GROBID | 0.8.1 | Only needed to parse new PDFs |
| OpenAI-compatible API | — | Set `OPENAI_API_KEY` in `.env` |

```bash
pip install -r requirements.txt
pip install fastapi uvicorn httpx hnswlib
```

### Start GROBID (for PDF parsing only)

```bash
docker run --rm -d -p 8070:8070 lfoppiano/grobid:0.8.1
```

### Configure

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
```

### Run

```bash
python start.py
# → http://localhost:8000
```

---

## Demo Mode (no GROBID required)

Pre-parse papers once and commit the output JSON. Demo users never need Docker.

```bash
# Parse from arXiv (requires GROBID running)
python scripts/save_demo_paper.py \
    --arxiv 1706.03762 \
    --key   attention \
    --name  "Attention Is All You Need"

# Parse from local PDF
python scripts/save_demo_paper.py \
    --pdf   papers/bert.pdf \
    --key   bert \
    --name  "BERT"
```

Output goes to `demo_papers/<key>.json` and is loaded at startup without GROBID.

---

## Library Mode

Index a collection of papers for cross-paper search:

```bash
# Place PDFs in papers/
python scripts/build_library_index.py \
    --papers_dir ./papers/ \
    --output_dir ./library_index/

python start.py
# → http://localhost:8000/library
```

---

## Benchmark

Evaluate retrieval strategies on QASPER:

```bash
python run_bench.py --max_papers 50
```

Strategies compared: BM25 · Dense · RRF · Cross-encoder  
Metrics: NDCG@5 · Recall@5 · MRR · Section Redundancy · LLM sufficiency judge

---

## Architecture

```
PDF / arXiv
    │
    ▼ GROBID (TEI XML)
parser.py ── paragraphs + table chunks (Markdown text + flat index_text)
    │
    ▼
PaperIndex
  ├─ BM25 (full corpus, index_text for tables)
  ├─ table_bm25 (table chunks only — for numerical / comparison fallback)
  └─ CrossEncoder reranker
    │
    ▼ query_type routing
  retrieve_ce()
  ├─ numerical   → force-inject top table via table_bm25
  ├─ comparison  → cross-table merge (LLM router + table_bm25 fallback)
  ├─ why_how     → expand top_k
  └─ methodology → section-name score boost
    │
    ▼
generator.py → LLM answer with cited sources
```

---

---

## 中文

**[English](#paperPilot)** | 中文

面向学术论文的检索增强问答系统，配套系统性 Benchmark 评测框架，用于对比和优化检索策略。

---

## 功能特性

- **PDF / arXiv 解析** — 通过 GROBID 将论文切分为段落和表格 chunk
- **表格感知检索** — 双字段索引（BM25 使用 `index_text`，LLM 使用 Markdown）；为数值和对比类查询构建独立的表格 BM25 子索引
- **查询类型路由** — 五类分类器（numerical / comparison / why_how / methodology / factual）驱动差异化检索行为
- **BM25 → Cross-encoder 管道** — 消融实验验证：Dense 检索在单篇论文场景下接近随机基线，管道裁剪为 BM25 + CE 重排
- **跨表格合并** — 对比类查询触发 LLM 路由，top-2 表格合并为联合上下文，支持跨表推理
- **Demo 模式** — 预解析论文以 JSON 形式随代码发布，Demo 用户无需运行 GROBID
- **中英切换** — 前端 i18n，偏好存储于 `localStorage`
- **Benchmark 评测** — 在 QASPER 上对比 BM25 / Dense / RRF / CE，指标包含 NDCG@5、Recall@5、MRR；两阶段 LLM-as-Judge 评测检索充分性与失败根因

---

## 快速开始

### 依赖

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | ≥ 3.10 | |
| GROBID | 0.8.1 | 仅解析新 PDF 时需要 |
| OpenAI 兼容 API | — | 在 `.env` 中配置 `OPENAI_API_KEY` |

```bash
pip install -r requirements.txt
pip install fastapi uvicorn httpx hnswlib
```

### 启动 GROBID（仅解析 PDF 时需要）

```bash
docker run --rm -d -p 8070:8070 lfoppiano/grobid:0.8.1
```

### 配置

```bash
cp .env.example .env
# 填写 OPENAI_API_KEY、OPENAI_BASE_URL、OPENAI_MODEL
```

### 运行

```bash
python start.py
# → http://localhost:8000
```

---

## Demo 模式（无需 GROBID）

预先解析好论文并提交 JSON 文件，Demo 用户克隆后即可直接体验：

```bash
# 从 arXiv 解析（需要 GROBID 运行）
python scripts/save_demo_paper.py \
    --arxiv 1706.03762 \
    --key   attention \
    --name  "Attention Is All You Need"

# 从本地 PDF 解析
python scripts/save_demo_paper.py \
    --pdf   papers/bert.pdf \
    --key   bert \
    --name  "BERT"
```

输出写入 `demo_papers/<key>.json`，服务启动时自动加载，无需 GROBID。

---

## 论文库模式

为一批论文建立跨篇检索索引：

```bash
# 将 PDF 放入 papers/ 目录
python scripts/build_library_index.py \
    --papers_dir ./papers/ \
    --output_dir ./library_index/

python start.py
# → http://localhost:8000/library
```

---

## Benchmark 评测

```bash
python run_bench.py --max_papers 50
```

对比策略：BM25 · Dense · RRF · Cross-encoder  
评测指标：NDCG@5 · Recall@5 · MRR · Section Redundancy · LLM 充分性判定

---

## 系统架构

```
PDF / arXiv
    │
    ▼ GROBID（TEI XML）
parser.py ── 段落 + 表格 chunk（Markdown text + 平铺 index_text）
    │
    ▼
PaperIndex
  ├─ BM25（全量，表格使用 index_text）
  ├─ table_bm25（仅表格 chunk — 数值/对比类查询保底）
  └─ CrossEncoder 重排
    │
    ▼ 查询类型路由
  retrieve_ce()
  ├─ numerical   → table_bm25 强制注入最相关表格
  ├─ comparison  → 跨表合并（LLM 路由 + table_bm25 兜底）
  ├─ why_how     → 扩展 top_k
  └─ methodology → Section 名称评分加权
    │
    ▼
generator.py → LLM 生成带来源引用的答案
```
