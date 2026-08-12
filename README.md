# RAGBenchMark / PaperPilot

这是一个用于论文问答和 RAG 策略评测的项目。当前版本重点保留两条主线：

- `Agentic-RAG`：真正的工具调用循环，由模型根据 observation 决定下一步工具、是否继续、何时停止。
- `Traditional BM25+CE`：用户上传单篇论文场景下的传统检索增强问答基线，使用 BM25 候选检索加 CrossEncoder 重排。

前端已重构为展示型问答工作台，只保留本地 PDF/demo paper 加载、方法选择、问题输入、回答查看、证据查看和 Agent 工具调用轨迹。arXiv 在线导入暂时不作为演示入口。

## 环境准备

建议使用 Python 3.10+。

```powershell
pip install -r requirements.txt
pip install fastapi uvicorn httpx
```

如果要解析新的 PDF，需要启动 GROBID：

```powershell
docker run --rm -d -p 8070:8070 lfoppiano/grobid:0.8.1
```

如果只使用 `demo_papers` 中已经解析好的 demo paper，则不需要 GROBID。

## 大模型配置

Agentic-RAG 需要 OpenAI-compatible API。任选一种配置方式：

```powershell
$env:OPENAI_API_KEY="your_key"
$env:OPENAI_MODEL="gpt-4o-mini"
```

或使用 DeepSeek：

```powershell
$env:DEEPSEEK_API_KEY="your_key"
$env:OPENAI_MODEL="deepseek-chat"
```

也可以写入 `.env`。后端会优先读取 `OPENAI_API_KEY`，没有时读取 `DEEPSEEK_API_KEY`。使用 DeepSeek 且没有显式设置 `OPENAI_BASE_URL` 时，默认使用 `https://api.deepseek.com`。

## 启动前端演示

```powershell
python -m uvicorn paperpilot.app:app --host 127.0.0.1 --port 8000
```

打开：

```text
http://127.0.0.1:8000/?ui=agentic-demo-9
```

页面流程：

1. 通过本地 PDF 或 demo paper 加载一篇论文。
2. 在右上方法选择中切换 `Agentic-RAG` 或 `Traditional BM25+CE`。
3. 输入问题并发送。
4. 查看回答、Evidence 列表。
5. 如果选择 `Agentic-RAG`，可以展开 `Tool calls`，人工检查每一轮工具调用、调用理由、observation 和新增 evidence。

注意：服务启动时会加载 `cross-encoder/ms-marco-MiniLM-L-6-v2`。如果本地没有缓存且当前网络无法访问 HuggingFace，启动会卡在模型下载/检查阶段。解决方式是提前缓存该模型，或配置可用的 HuggingFace 网络代理。

如果出现端口占用错误：

```text
[Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)
```

先检查端口：

```powershell
netstat -ano | Select-String ":8000|:8001|:8020"
```

本机曾出现 `Code.exe` 占用 8000 和 8001 的情况，这通常来自 VS Code 端口转发或预览功能。不要直接杀 VS Code，优先换一个空闲端口：

```powershell
python -m uvicorn paperpilot.app:app --host 127.0.0.1 --port 8020
```

然后打开：

```text
http://127.0.0.1:8020/?ui=agentic-demo-9
```

## 当前 Agentic-RAG 工具

Agentic-RAG 的核心实现位于：

```text
paperpilot/core/agentic_rag.py
```

当前工具包括：

- `rewrite_query`：生成查询改写，但不直接检索。
- `decompose_question`：拆解复杂、对比、数值类问题。
- `retrieve_evidence`：直接事实型检索。
- `retrieve_fusion`：多 query 融合检索，适合比较、why/how、methodology 问题。
- `retrieve_numeric_or_table`：偏向表格、指标、数值证据的检索。
- `list_neighbor_chunks`：获取命中 chunk 的相邻 chunk，用于补全局部上下文。
- `extract_table_data`：从已命中的表格 evidence 中抽取结构化行列。
- `inspect_context`：检查当前 evidence 是否足够。
- `verify_answer_support`：验证候选答案是否被当前 evidence 支持。
- `finish`：模型主动停止循环并给出最终答案。

关键约束：

- 每轮只能调用一个工具。
- 每次调用后必须读取 observation 再决定下一步。
- 不允许固定 `retrieve -> inspect -> finish` 的 workflow。
- 数值、对比、多事实答案需要优先验证支撑性。
- 达到 evidence 足够或继续检索收益很低时，由模型调用 `finish` 停止。

## 前端重构内容

本次前端重构主要修改：

```text
paperpilot/static/index.html
paperpilot/static/app.js
paperpilot/static/style.css
paperpilot/app.py
```

删除或淡出旧前端中不适合展示的内容：

- Library 导航入口。
- 旧的 `CE / RRF` 小切换。
- 多语言 i18n 包袱。
- 反馈弹窗和反馈链路。
- 过多产品化说明。

保留并强化：

- 论文加载。
- 两种方法选择：`Agentic-RAG`、`Traditional BM25+CE`。
- 问答输入。
- 回答展示。
- evidence 展示。
- Agentic 工具调用轨迹展示。

后端 `/api/query` 当前支持：

- `mode=agentic`
- `mode=rrf_ce`

为了兼容旧调用，也接受：

- `agentic_rag`
- `agentic-rag`
- `traditional`
- `ce`
- `rrf`

其中 `rrf_ce` 展示路径复用项目现有的单论文 BM25 候选检索 + CrossEncoder 重排实现。

## 运行 Agentic QA 实验

小规模 smoke test：

```powershell
python run_agentic_qa.py --max-papers 5 --limit 20 --output data\runs\agentic_qa_smoke\trace.jsonl
```

全量实验：

```powershell
python run_agentic_qa.py --max-papers 0 --limit 0 --output data\runs\agentic_qa_full_rewrite_verify\trace.jsonl
```

注意 PowerShell 中不要把两条命令粘在同一行，否则会出现：

```text
error: unrecognized arguments: run_agentic_qa.py
```

## 运行传统 RAG 基线

传统基线实验：

```powershell
python run_bench.py --max-papers 0 --limit 0 --output-dir data\runs\traditional_full_no_llm
```

已跑出的关键结果：

```text
Traditional RRF+CE full-library baseline
N = 845
Hit@1 = 0.1337
Hit@5 = 0.2911
EvidenceRecall@5 = 0.2111
AllEvidence@5 = 0.1574
MRR = 0.1927
Average latency = 112.8 ms
```

```text
Agentic-RAG full run
N = 845
Hit@1 = 0.3456
Hit@5 = 0.7385
EvidenceRecall@5 = 0.5747
AllEvidence@5 = 0.4521
MRR = 0.4901
NDCG@5 = 0.4645
Average latency = 6366.7 ms
```

相对传统 RRF+CE 全库基线：

```text
Hit@1: +0.2119
Hit@5: +0.4474
EvidenceRecall@5: +0.3636
AllEvidence@5: +0.2947
MRR: +0.2974
```

## 查看 Agent 完整日志

Agentic QA 运行会写 JSONL，每条 QA 一行，包含：

- question
- groundtruth
- final answer
- retrieved evidence
- tool trace
- rounds
- metrics

常用路径：

```text
data/runs/agentic_qa_full_rewrite_verify/trace.jsonl
data/runs/agentic_qa_smoke/trace.jsonl
data/runs/agentic_qa_simplified_p5/trace.jsonl
```

如果需要更适合人工检查的 HTML/Markdown 报告，可以使用：

```powershell
python scripts/render_agentic_trace_report.py --input data\runs\agentic_qa_smoke\trace.jsonl --output data\runs\agentic_qa_smoke\trace_report.html
```

## 当前报告文件

简历/项目复盘报告：

```text
AGENTIC_RAG_RESUME_REPORT.md
```

该报告基于当前全量实验结果撰写，可用于描述项目问题、方法、指标提升和工程实现。


## 路由 Debug

路由错误会直接导致 Agentic-RAG 调错工具，例如把 `key experimental results` 当成普通事实题。可以运行 500 条带标签的离线路由测试：

```powershell
python -B scripts\debug_router_500.py --output-dir data\runs\router_debug_500
```

输出文件：

```text
data/runs/router_debug_500/router_debug_500_summary.md
data/runs/router_debug_500/router_debug_500.jsonl
```

当前结果：

```text
Accuracy: 500/500 = 1.000
```

## 本地小模型路由

路由采用企业常见的混合策略：

1. 正则/规则先处理高精度明显问题。
2. 本地小模型只在高置信时覆盖规则结果。
3. 如果本地模型不存在，系统自动退回纯规则，不影响启动。
4. 复杂问题仍可进入大模型 planner，由大模型输出检索策略、rewrite query 和 evidence policy。

当前小模型使用：

```text
sentence-transformers/all-MiniLM-L6-v2
```

项目内下载位置：

```text
models/router/all-MiniLM-L6-v2
```

下载命令：

```powershell
python -B scripts\download_router_model.py --model sentence-transformers/all-MiniLM-L6-v2 --output-dir models\router\all-MiniLM-L6-v2
```

在 QASPER 上验证路由时，不要把 answer/evidence 暴露给 router。默认命令只输入 question：

```powershell
python -B scripts\debug_qasper_router.py --qasper data\qasper_raw\qasper-dev-v0.3.json --output-dir data\runs\qasper_router_debug_question_only_model_tuned --proxy-mode question_only
```

当前 QASPER question-only proxy 结果：

```text
Total questions: 1005
Proxy accuracy: 685/1005 = 0.682
```

注意：QASPER 有官方答案形态标签，通常可按五类统计为 `extractive/free_form/yes/no/unanswerable`。这和本项目路由使用的 `numerical/comparison/why_how/methodology/factual` 检索意图标签不是同一件事，所以这里仍是 question-only proxy accuracy，不是真实 router gold accuracy。debug JSONL 会保留官方 answer-form labels，便于后续做辅助分析。

阈值扫描命令：

```powershell
python -B scripts\tune_qasper_router_model.py --qasper data\qasper_raw\qasper-dev-v0.3.json --proxy-mode question_only --output data\runs\qasper_router_threshold_sweep.json
```

当前采用的覆盖阈值：

```text
semantic_score >= 0.55
semantic_margin >= 0.15
```

Docker 部署时把模型目录一起复制进镜像即可，例如：

```dockerfile
COPY models/router/all-MiniLM-L6-v2 /app/models/router/all-MiniLM-L6-v2
```

如需放到镜像外部挂载，也可以设置环境变量：

```powershell
$env:ROUTER_MODEL_DIR="D:\path\to\all-MiniLM-L6-v2"
```

## 测试与检查

运行测试：

```powershell
python -m pytest tests
```

当前已通过：

```text
24 passed
```

前端和后端语法检查：

```powershell
python -B -c "import ast, pathlib; ast.parse(pathlib.Path('paperpilot/app.py').read_text(encoding='utf-8')); print('app syntax ok')"
node --check paperpilot\static\app.js
```

FastAPI 首页检查：

```powershell
python -B -c "from fastapi.testclient import TestClient; from paperpilot.app import app; c=TestClient(app); r=c.get('/?ui=agentic-demo-9'); print(r.status_code, 'Agentic-RAG' in r.text, 'Traditional BM25+CE' in r.text)"
```

期望输出：

```text
200 True True
```

## 数据目录清理建议

可以删除临时测试目录：

```text
data/test_runs
```

`data/runs` 建议至少保留最终全量实验结果：

```text
data/runs/agentic_qa_full_rewrite_verify
data/runs/traditional_full_no_llm
```

如果只想保留最新一次 Agentic 全量实验，也可以删除旧 smoke run 和中间 run。但删除前确认其中没有需要人工检查的 trace/report。

`.npy` 文件通常是索引或向量缓存。若只关心最终全量实验结果和报告，可以删除中间测试产生的 `.npy`；若后续还要复现实验或快速启动已有索引，建议保留对应全量 run 或 library index 下的缓存。

## 重要文件索引

```text
paperpilot/app.py                         FastAPI 后端和 /api/query
paperpilot/core/agentic_rag.py            Agentic-RAG 工具调用循环
paperpilot/core/indexer.py                单论文 BM25 + CrossEncoder 检索
paperpilot/core/query_rewrite.py          查询改写
paperpilot/core/numeric_reasoner.py       数值/表格辅助逻辑
bench/agentic_eval.py                     Agentic QA 评测
run_agentic_qa.py                         Agentic QA CLI
run_bench.py                              传统 RAG benchmark CLI
scripts/render_agentic_trace_report.py    Trace 人工检查报告生成
paperpilot/static/index.html              重构后的前端入口
paperpilot/static/app.js                  前端交互逻辑
paperpilot/static/style.css               前端样式
```





