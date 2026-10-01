# K4 — Level 3B, Ngày 14: AI Evaluation & Benchmarking Pipeline (225 phút)

**AICB-P1 · Phase 1 · Ngày 14 trong 15 · K4**

Lab này là bài **AI Evaluation**. Bạn sẽ hoàn thiện evaluation core trong `template.py`, xây dựng một golden dataset 20 câu, chạy một hệ thống RAG thật trên corpus **OrbitTech Store Customer Support**, rồi phân tích kết quả benchmark.

> Hệ thống RAG trong `domain_assistant.py` là **system under evaluation**. Nó sinh câu trả lời; `template.py` là **evaluation engine** chấm các câu trả lời đó. Hai phần có vai trò hoàn toàn độc lập.

---

## ⚠️ Bài Làm Cá Nhân

**Đây là bài tập cá nhân. Mỗi học viên nộp một repository của riêng mình.**

Tài liệu chính thức của bài lab:

- [SUBMISSION.md](SUBMISSION.md) — cấu trúc bài nộp, tên repo và nơi nộp
- [RUBRIC.md](RUBRIC.md) — tiêu chí chấm, bằng chứng và điều kiện mất điểm
- [CHECKPOINTS.md](CHECKPOINTS.md) — sản phẩm, kiến thức và cách tự kiểm tra từng checkpoint
- [RULES.md](RULES.md) — quy định làm bài, dùng AI, hợp tác và bảo mật

### Quy chuẩn đặt tên Repository

| Vai trò | Tên chuẩn |
|---|---|
| Assignment / starter repo (repo này) | `K4-L3B-AI-Evaluation` |
| Student submission repo | `K4-L3B-<HoVaTen>-<MSSV>-AIEvaluation` |
| Ví dụ | `K4-L3B-NguyenVanAn-L3A202600280-AIEvaluation` |

> ⚠️ **Đặt sai tên repo = trừ 5 điểm** theo quy định trong [RUBRIC.md](RUBRIC.md).

Bài lab là **bài làm cá nhân**. **Mỗi cá nhân phải tự nộp link repo của mình lên Codelab** (không nộp hộ, không dùng chung repository).  
Hạn nộp mặc định: **23h59 ngày lab (GMT+7)**; coach có thể gia hạn tối đa ≤48h.

---

## Yêu cầu & Quick Start

**Yêu cầu:** Python 3.11 trở lên. Cần **OpenAI API key** để chạy `domain_assistant.py` (Part 3 — sinh 20 actual answers từ RAG thật); phần code core (`template.py`, Part 1–2) không cần API key.

```bash
python --version                                        # xác nhận Python 3.11+
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
pytest tests/ -v                                         # baseline: 42 tests collected, 42 failed
cp .env.example .env                                     # điền OPENAI_API_KEY (chỉ cần cho Part 3)
```

Chi tiết hướng dẫn theo hệ điều hành và xử lý lỗi: xem [`guide_lab.md`](guide_lab.md).

---

## Bài làm — Lệnh test, chạy pipeline và Dashboard

Học viên: **Đỗ Trung Tuyến** — MSSV `2A202602427`.

### 1. Kiểm tra (không cần API key)

```bash
python -m pip install -r requirements.txt
cp template.py solution/solution.py      # đồng bộ bản nộp (hai file phải giống hệt nhau)
pytest tests/ -v                         # kỳ vọng: 42 passed (đã làm bonus rerank_by_overlap)
python validate_golden_dataset.py        # kỳ vọng: PASS
```

### 2. Chạy lại benchmark

```bash
python evaluate_answers.py               # artifacts/actual_answers.json -> artifacts/benchmark_results.json (không gọi API)
python domain_assistant.py               # TỐN API: sinh lại 20 actual answers (cần .env có OPENAI_API_KEY)
```

`artifacts/` đã có sẵn kết quả của lần chạy thật, nên chỉ cần `evaluate_answers.py` để tính lại metrics.
Chỉ chạy `domain_assistant.py` khi muốn sinh câu trả lời mới, sau đó chạy lại `evaluate_answers.py`.

### 3. Bonus (Exercise 3.4 & 3.5)

```bash
python bonus/rerank_experiment.py        # 3.5: retrieval metrics trước/sau rerank_by_overlap -> artifacts/rerank_results.json

# 3.4: RAGAS vs DeepEval — dùng venv riêng, TỐN API
python -m venv .venv-bonus
.venv-bonus/Scripts/python -m pip install -r bonus/requirements-bonus.txt   # macOS/Linux: .venv-bonus/bin/python
.venv-bonus/Scripts/python bonus/compare_frameworks.py                      # -> artifacts/framework_comparison.json
```

### 4. Dashboard (`demo/index.html`)

Dashboard đọc trực tiếp `artifacts/benchmark_results.json`, `artifacts/actual_answers.json` và
`golden_dataset.json`. Trang không nhúng sẵn số liệu nên phải mở qua HTTP server: mở file bằng
`file://` thì trình duyệt sẽ chặn `fetch`.

- **Online (GitHub Pages):** https://trungtuyendo02.github.io/K4-L3B-DoTrungTuyen-2A202602427-AIEvaluation/demo/
- **Local:**

  ```bash
  python -m http.server 8000             # chạy tại thư mục gốc của repo
  # mở http://localhost:8000/demo/
  ```

- **Không có server:** mở `demo/index.html`, bấm **“Chọn file JSON”** rồi chọn cùng lúc 3 file JSON ở trên.

Sau khi chạy lại `evaluate_answers.py`, tải lại trang (F5) để dashboard hiển thị số liệu mới.

---

## Mục tiêu

Sau bài lab này, học viên có thể:

1. Xây dựng pipeline đánh giá tự động cho AI agent trên 20 test cases.
2. Triển khai các metrics lấy cảm hứng từ RAGAS (answer-side và retrieval-side).
3. Thiết kế LLM-as-a-Judge rubric theo thang điểm 1–5 và cơ chế kiểm soát bias.
4. Xây dựng golden dataset bằng phương pháp stratified sampling.
5. Thực hiện failure analysis bằng kỹ thuật failure clustering và 5 Whys.
6. Thiết lập evaluation pipeline như một quality gate trong CI / CD.

---

## Luồng end-to-end của bài lab

```text
data/technology_store/*.md
             │
             ├── học viên đọc và viết ──> golden_dataset.json
             │                               │
             └── DomainAssistant <── question
                       │
                       ├── retrieve chunks
                       └── generate actual answer
                                  │
                                  v
                     artifacts/actual_answers.json
                                  │
                    evaluate_answers.py
                                  │
                 template.py (evaluation core)
                                  │
                                  v
                  artifacts/benchmark_results.json
                                  │
                     exercises.md + reflection.md
```

`domain_assistant.py` chỉ đọc `id` và `question` khi sinh answer. Nó **không đọc `expected_answer` hoặc gold contexts**, nhằm tránh data leakage.

---

## Cấu trúc repo

```text
.
├── SUBMISSION.md                # quy định nộp bài, tên repo, deliverables, checklist
├── RUBRIC.md                    # bảng điểm 100, bằng chứng, deductions, bonus
├── CHECKPOINTS.md               # lộ trình CP0–CP5, sản phẩm, cách tự kiểm tra
├── RULES.md                     # quy định cá nhân, AI, hợp tác, bảo mật, deadline
├── README.md                    # tổng quan bài lab và quick start
├── guide_lab.md                 # hướng dẫn chi tiết từng bước end-to-end
├── exercises.md                 # worksheet bài tập Part 1–3
├── reflection.md                # báo cáo failure analysis, 5 Whys và regression
├── template.py                  # starter evaluation core chứa các TODO
├── solution/
│   └── solution.py              # bản sao hoàn thiện của template.py khi nộp bài
├── domain_assistant.py          # RAG system under evaluation (OrbitTech Support)
├── evaluate_answers.py          # adapter artifact → evaluation core
├── validate_golden_dataset.py   # script kiểm tra schema và provenance dataset
├── golden_dataset.json          # form 20 QA để học viên điền
├── data/technology_store/       # corpus tài liệu nguồn của OrbitTech Store
├── tests/                       # bộ unit tests kiểm tra evaluation core
├── requirements.txt
└── .env.example
```

Khi chạy benchmark, các script sẽ tạo thư mục `artifacts/` chứa `actual_answers.json` và `benchmark_results.json` để phục vụ phân tích.

---

## Tổng quan Tasks

- **Task 1 — Data Models:** Hoàn thiện `QAPair`, `EvalResult` và phương thức `overall_score()`.
- **Task 2 — RAGASEvaluator:** Triển khai 3 answer metrics (`faithfulness`, `relevance`, `completeness`) và 2 retrieval metrics (`context_recall`, `context_precision`).
- **Task 3 — LLMJudge:** Xây dựng `score_response()` chấm điểm theo rubric và `detect_bias()` phát hiện bias.
- **Task 4 — BenchmarkRunner:** Chạy pipeline benchmark, tổng hợp báo cáo và phát hiện regression (> 0.05).
- **Task 5 — FailureAnalyzer:** Phân loại lỗi (`categorize_failures`), chẩn đoán nguyên nhân gốc (`find_root_cause`) và tạo bảng `improvement_log`.
- **Task 6 — Golden Dataset & Real Benchmark:** Xây dựng 20 QA dataset, chạy RAG tạo actual answers, chạy benchmark và hoàn thiện `reflection.md`.

Chi tiết từng task và checkpoints xem tại [`CHECKPOINTS.md`](CHECKPOINTS.md) và [`guide_lab.md`](guide_lab.md).

---

## Thời gian làm bài

Buổi học diễn ra từ **9:15 đến 13:00**. Hoàn thành bài lab trước **12:00**; thời gian 12:00–13:00 dành cho demo và Q&A.

| Thời gian | Checkpoint | Hoạt động |
|---|---|---|
| 9:15–9:30 | **CP0** Setup | Tạo môi trường, baseline tests (42 failed), cấu hình `.env` |
| 9:30–9:45 | **CP1** Task 1 | Hoàn thành Data Models và `overall_score` (3 passed) |
| 9:45–10:20 | **CP2** Tasks 2–3 | Hoàn thành RAGAS metrics và LLMJudge (21 passed) |
| 10:20–10:40 | **CP3** Tasks 4–5 | BenchmarkRunner, FailureAnalyzer (full suite 41 passed, 1 skipped) |
| 10:40–11:35 | **CP4** Part 3 | Golden Dataset 20 QA, chạy RAG, benchmark thật và rubric |
| 11:35–12:00 | **CP5** Part 4 | Failure analysis, 5 Whys trong `reflection.md`, copy `solution/solution.py` |
| 12:00–13:00 | Wrap-up | Demo, review và Q&A |

---

## Đánh giá & Tiêu chí chấm điểm

| Tiêu chí | Điểm |
|---|---:|
| Core coding hoàn chỉnh, toàn bộ required tests pass | 50 |
| Golden dataset 20 QA đúng schema, stratification và evidence | 15 |
| LLM-as-a-Judge rubric design rõ ràng, domain-specific | 10 |
| Benchmark, 5 Whys, failure analysis và improvement log | 15 |
| Chất lượng code, type hints và regression strategy | 10 |
| **Tổng điểm bắt buộc** | **100** |

Điểm thưởng (Bonus):

| Tiêu chí Bonus | Điểm |
|---|---:|
| Exercise 3.4 — So sánh hai evaluation frameworks | +5 |
| Exercise 3.5 — Reranking và phân tích retrieval metrics | +5 |
| **Tổng bonus tối đa** | **+10** |

> Tổng bonus của bài lab tối đa **10 điểm** (Exercise 3.4 +5, Exercise 3.5 +5). Đây là điểm sản phẩm lab, không phải điểm giơ tay / pitching.

Chi tiết tiêu chí chấm điểm, bằng chứng và các trường hợp trừ điểm xem tại [RUBRIC.md](RUBRIC.md).  
Hướng dẫn nộp bài và checklist trước khi nộp xem tại [SUBMISSION.md](SUBMISSION.md).
