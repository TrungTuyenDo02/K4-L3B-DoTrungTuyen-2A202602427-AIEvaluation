# Day 14 — Reflection

## Evaluation Report & Failure Analysis

Dùng kết quả thật trong `artifacts/benchmark_results.json` và kiểm tra lại
answer/context trace trong `artifacts/actual_answers.json` trước khi kết luận.

> Run được phân tích: **RUN 2** — `domain-assistant`, model `gpt-4o-mini`, `top_k=5`,
> `prompt_version=1.0`, generated_at `2026-10-01T03:12:46Z`. RUN 1 (`02:47:40Z`, cùng
> cấu hình) đã bị ghi đè; so sánh hai run nằm ở Mục 5. Thứ hạng BM25 ngoài top-5
> được đo lại bằng `BM25Retriever.retrieve(question, top_k=<all chunks>)` trên cùng
> corpus.

---

## 1. Benchmark Results Summary

<!-- DRAFT: học viên cần đọc lại và viết theo ý mình -->

**Overall pass rate:** 50.0% (10/20). Theo độ khó: Easy 4/5, Medium 5/7, Hard 1/5,
Adversarial 0/3.

| Metric | Average | Min | Max | Nhận xét |
|---|---:|---:|---:|---|
| Context Recall | 0.874 | 0.160 (A01) | 1.000 (E01) | 18/20 case ≥ 0.8. Chỉ A01 (out-of-scope, lệch từ vựng) và H03 (0.619, câu nhiều điều kiện) bị thiếu evidence. |
| Context Precision | 0.914 | 0.333 (A01) | 1.000 (E01) | Chunk liên quan thường đứng hạng 1–2; ranking không phải vấn đề chính. |
| Faithfulness | 0.569 | 0.091 (A01) | 0.861 (M05) | Thấp chủ yếu vì metric so answer với **gold context ngắn**, không so với retrieved chunks; answer đúng nhưng thêm thông tin có thật vẫn bị phạt (E02, M06). |
| Relevance | 0.580 | 0.357 (H03) | 0.792 (M04) | 0/20 case ≥ 0.8: overlap theo token câu hỏi phạt answer diễn đạt lại và các lời từ chối. |
| Completeness | 0.624 | 0.080 (A01) | 0.939 (M05) | Phản ánh đúng H01 (0.294, sai version) và H03 (0.405, thiếu ngoại lệ), nhưng cũng phạt nặng các lời từ chối ngắn (A02, A03). |
| Overall Score | 0.591 | 0.257 (A01) | 0.808 (E04) | Chỉ 1 case ở mức Good. |

**Score interpretation**

- Metrics/cases ở mức Good (0.8–1.0): Context Recall (0.874), Context Precision (0.914). Theo Overall chỉ có E04 (0.808).
- Metrics/cases ở mức Needs Work (0.6–0.8): Completeness (0.624). Theo Overall có 9 case: E01, E02, E03, E05, M01, M04, M05, M07, H05.
- Metrics/cases ở mức Significant Issues (<0.6): Faithfulness (0.569), Relevance (0.580), Overall (0.591). Theo Overall có 10 case: M02, M03, M06, H01, H02, H03, H04, A01, A02, A03.

**Failure type distribution** (trên 10 failures)

| Failure Type | Count | Percentage |
|---|---:|---:|
| hallucination | 1 (A01) | 10% |
| irrelevant | 0 | 0% |
| incomplete | 1 (H01) | 10% |
| off_topic | 8 (E02, M03, M06, H02, H03, H04, A02, A03) | 80% |
| refusal | 0 (heuristic không có nhãn này) | 0% |

**Chẩn đoán tổng quan:** Vấn đề chính nằm ở retrieval, generation hay cả hai?
Dùng ít nhất hai metrics để bảo vệ kết luận.

> *Câu trả lời:* Vấn đề chính nằm ở **generation**. Ngoài ra có một vấn đề lớn thứ
> hai ở **chính evaluation metric**; retrieval chỉ là nguyên nhân phụ.
>
> - **Retrieval tốt:** Context Recall 0.874 và Precision 0.914, nghĩa là evidence cần
>   thiết thường có mặt và đứng đầu danh sách. Chỉ H03 và A01 thật sự thiếu chunk.
> - **Generation sai dù retrieval đúng:** H01 có Precision 1.000 và chunk quyết định
>   OT-09-P04 ở hạng 1, nhưng Completeness chỉ 0.294 và answer sai (45 ngày thay vì 21).
>   H04 (Recall 0.846) kết luận "không còn warranty" khi mới 11/12 tháng. Ở RUN 1, M02
>   còn kết luận USD 280 "above the USD 300 threshold". Cả ba là lỗi suy luận điều kiện/số.
> - **Metric chưa đáng tin:** đọc trace, nhãn `off_topic` (80%) gồm cả answer đúng
>   (E02, M03, M06, H02, A02, A03). M02 ở RUN 1 sai mà vẫn PASS; ở RUN 2 đúng nhưng điểm
>   thấp hơn. Faithfulness trung bình 0.569 bị kéo xuống vì so với gold context ngắn thay
>   vì retrieved context. Vì vậy **pass rate 50% vừa phạt oan answer đúng, vừa có thể
>   để lọt answer sai.**

---

## 2. Top 3 Worst Failures — 5 Whys

<!-- DRAFT: học viên cần đọc lại và viết theo ý mình -->

Phân loại failure trước khi đề xuất fix. Với mỗi case, kiểm tra cả gold evidence
và retrieved chunks; không suy luận chỉ từ một score.

### Failure 1

**ID và question:**

> A01 (adversarial, `out_of_scope`): "Which technology stocks should I invest in this
> year to make the most money?"

**Expected answer:**

> This is outside the assistant's scope: investment advice is unrelated to OrbitTech
> customer support. The assistant should briefly explain its role and offer examples
> of supported OrbitTech topics, such as products, orders, shipping, returns,
> warranty, and repairs.

**Actual answer:**

> The retrieved contexts do not provide any information regarding technology stocks
> or investment advice. Therefore, I cannot answer your question about which
> technology stocks to invest in this year.

**Scores:** Context Recall: 0.160 | Context Precision: 0.333 | Faithfulness: 0.091 |
Relevance: 0.600 | Completeness: 0.080 | Overall: 0.257

**Evidence inspection:** Retriever lấy đúng/thiếu/thừa chunks nào?

> Retriever chỉ trả về **3 chunk** (chỉ 3 chunk có BM25 > 0), cả ba đều là nhiễu:
> OT-05-P04 (bundle, 3.00), OT-02-P01 (order, 2.76), OT-04-P05 (lost package, 2.46).
> Hai gold chunk OT-00-P03 ("…investment advice… outside scope…") và OT-00-P01 có
> **BM25 = 0**: câu hỏi dùng "invest", "stocks", "money", còn `_normalize()` của retriever
> không đưa "investment" về "invest". Answer không bịa gì; nó chỉ thiếu phần giải
> thích vai trò và gợi ý chủ đề được hỗ trợ.

| Level | Question | Answer |
|---|---|---|
| Symptom | Vấn đề quan sát được là gì? | Assistant từ chối nhưng chỉ nói "contexts không có thông tin", không giải thích vai trò và không gợi ý chủ đề OrbitTech như policy yêu cầu. Metric gắn nhãn `hallucination` (faithfulness 0.091). |
| Why 1 | Tại sao symptom xảy ra? | Generator không thấy policy out-of-scope (OT-00-P03) nên chỉ áp dụng rule chung của prompt: "If evidence is insufficient, say so". |
| Why 2 | Tại sao nguyên nhân trên xảy ra? | BM25 là retrieval từ vựng: "invest" ≠ "investment" sau normalize, nên chunk scope có điểm 0. Câu out-of-scope vốn ít trùng từ với corpus nên luôn dễ bị trượt. |
| Why 3 | Tại sao vấn đề đó chưa được ngăn chặn? | Rule về scope và safety chỉ nằm trong corpus (`00_system_scope.md`) và phải *được retrieve* mới có hiệu lực. Prompt v1.0 không ghim rule out-of-scope (giải thích vai trò + gợi ý chủ đề). |
| Why 4 | Tại sao cơ chế hiện tại chưa phát hiện hoặc xử lý được? | Không có bước intent/scope classification trước retrieval. Metric hiện tại lại gắn nhãn sai thành `hallucination`, khiến lỗi trông như vấn đề grounding thay vì vấn đề scope. |
| Why 5 | Root cause có thể hành động được là gì? | **Hành vi bắt buộc (scope policy) phụ thuộc vào retrieval từ vựng thay vì được ghim vào system prompt.** Cách sửa: ghim tóm tắt `00_system_scope.md` vào system prompt kèm mẫu trả lời out-of-scope, và/hoặc thêm bước phân loại scope trước retrieval. |

**Root cause từ `find_root_cause()`:**

> Answer is missing key information — increase context window or improve generation

**Bạn đồng ý hay không? Dẫn evidence từ trace:**

> Đồng ý một phần. Đúng là answer thiếu thông tin (vai trò, chủ đề được hỗ trợ):
> completeness 0.080 thấp nhất. Nhưng "increase context window" không giúp gì: chunk
> cần thiết có BM25 = 0 nên tăng top-k vẫn không lấy được, và retriever chỉ có 3 chunk
> điểm dương. Nhãn `hallucination` của `run_full_eval()` cũng sai, vì answer không chứa
> claim bịa nào. Faithfulness thấp chỉ do answer ít trùng từ với gold context.

**Proposed fix cụ thể:**

> 1. Thêm vào system prompt một khối "Scope rules" (tóm tắt OT-00-P02, P03, P04) và mẫu:
>    *"I can only help with OrbitTech customer support, such as products, orders,
>    shipping, returns, warranty, and repairs."*
> 2. Cải thiện normalize/stemming (investment → invest) hoặc thêm dense retrieval.
> 3. Verify: chạy lại A01 cùng vài biến thể (crypto, y tế, pháp lý). Mục tiêu:
>    Completeness A01 ≥ 0.5 và judge Safety/scope (rubric 3.3) = 5. Nếu sửa cả
>    retrieval thì Context Recall A01 ≥ 0.8.

### Failure 2

**ID và question:**

> H03 (hard): "I ordered a NovaBook 14 in a promotional bundle with free AeroBuds Pro
> in September 2026. Ten days after delivery I opened the laptop and want to return it
> for preference but keep the earbuds. What deductions apply to my refund?"

**Expected answer:**

> The opened laptop is within the 14-day opened-device window, so it can be returned
> with a 10% restocking fee; the fee is waived only if a defect is verified during the
> return window. Because a promotional bundle must be returned as a bundle, keeping
> the free AeroBuds Pro means their stated promotional value is deducted from the
> refund. Original standard-shipping fees are not refunded for preference returns.

**Actual answer:**

> To return the NovaBook 14 while keeping the AeroBuds Pro, the stated promotional
> value of the earbuds will be deducted from your refund. Since you opened the laptop,
> the return falls under the conditions of Return Policy version 2.0, which allows for
> a 14-day return window and charges a 10% restocking fee. Therefore, your refund will
> be the purchase price of the NovaBook 14 minus the promotional value of the AeroBuds
> Pro and the 10% restocking fee.

**Scores:** Context Recall: 0.619 | Context Precision: 0.950 | Faithfulness: 0.316 |
Relevance: 0.357 | Completeness: 0.405 | Overall: 0.359

**Evidence inspection:**

> Top-5: OT-03-P04 (bundle rule, **gold ✓**, hạng 1), OT-09-P04 (tóm tắt version, có
> "v2.0… 14 days opened… 10%"), OT-01-P03 (mô tả sản phẩm AeroBuds, **nhiễu**),
> OT-06-P01 (warranty, **nhiễu**), OT-05-P04 (tham chiếu bundle). **Thiếu** hai gold chunk:
> OT-05-P01 (phí 10% + miễn phí khi lỗi được xác minh) ở **hạng 6**, và OT-05-P05
> ("standard-shipping fees are not refunded for preference returns") ở **hạng 10**.
> Answer đúng với những gì nó có (10% lấy từ OT-09-P04, khoản trừ bundle lấy từ
> OT-03-P04), nhưng bỏ sót hai khoản/ngoại lệ có trong hai chunk bị thiếu.

| Level | Question | Answer |
|---|---|---|
| Symptom | Vấn đề quan sát được là gì? | Answer thiếu 2/4 ý: không nói phí ship standard không được hoàn và không nói phí restocking được miễn nếu xác minh có lỗi. Khách sẽ nhận ít tiền hơn mức họ được báo. |
| Why 1 | Tại sao symptom xảy ra? | Hai chunk chứa các rule đó (OT-05-P01, OT-05-P05) không nằm trong top-5 context. |
| Why 2 | Tại sao nguyên nhân trên xảy ra? | Câu hỏi chứa nhiều tên sản phẩm ("NovaBook 14", "AeroBuds Pro"). Các term này khớp mạnh với chunk catalog/warranty (OT-01-P03, OT-06-P01) và đẩy chunk return-policy xuống. Cơ chế giảm điểm khi lặp nguồn (`SOURCE_REPEAT_DECAY`) cũng hạ thêm các chunk thứ 2–3 của `05_returns_and_exchanges.md`. |
| Why 3 | Tại sao vấn đề đó chưa được ngăn chặn? | Một câu hỏi nhiều phần (window, phí, bundle, phí ship) được retrieve bằng **một query duy nhất với top_k cố định = 5**, không tách thành các câu hỏi con. |
| Why 4 | Tại sao cơ chế hiện tại chưa phát hiện hoặc xử lý được? | Generator không biết mình thiếu evidence nên không nói "không đủ thông tin về phí ship". Pipeline chưa có ngưỡng Context Recall để báo động riêng cho câu Hard. |
| Why 5 | Root cause có thể hành động được là gì? | **Cấu hình retrieval (BM25 một query, top-5, trọng số lệch về tên sản phẩm) không phù hợp với câu hỏi policy nhiều điều kiện.** Cách sửa: tách query (multi-query) hoặc tăng top-k cho câu dài, kết hợp reranker theo nội dung policy. |

**Root cause và proposed fix:**

> `find_root_cause()` → **"Multiple issues detected — review full pipeline"** (vì cả 3
> score < 0.5). Đồng ý một phần: score thấp ở cả ba chiều, nhưng trace cho thấy nguyên
> nhân **chính là retrieval** (Recall 0.619). Phần generation đúng với context nó có;
> Relevance 0.357 chủ yếu do heuristic overlap.
>
> **Fix:** (1) multi-query retrieval: tách câu hỏi thành các ý "opened device return
> fee", "bundle free gift kept", "shipping fee refund", retrieve từng ý rồi gộp và khử
> trùng; hoặc (2) top_k = 8 rồi rerank. **Verify:** chạy lại H03 và toàn bộ Hard.
> Context Recall H03 ≥ 0.9, Completeness ≥ 0.6, đồng thời Context Precision trung bình
> không giảm > 0.05.

### Failure 3

**ID và question:**

> H01 (hard): "I am an OrbitPlus member. I ordered an unopened NovaBook 14 on August
> 28, 2026, and it was delivered on September 3, 2026. How long do I have to return it
> unopened?"

**Expected answer:**

> Return Policy version 1.0 applies because the order was placed before September 1,
> 2026, so the unopened-device window is 21 calendar days, counted from confirmed
> delivery on September 3. The 45-day OrbitPlus benefit does not apply: orders placed
> before September 1 keep the 21-day version 1.0 window regardless of membership.

**Actual answer:**

> As an OrbitPlus member, you have 45 calendar days to return the unopened NovaBook 14,
> starting from the delivery date of September 3, 2026. This means you can return it
> until October 18, 2026.

**Scores:** Context Recall: 0.853 | Context Precision: 1.000 | Faithfulness: 0.455 |
Relevance: 0.526 | Completeness: 0.294 | Overall: 0.425

**Evidence inspection:**

> Retrieval gần như đủ: **OT-09-P04 ở hạng 1** chứa nguyên câu quyết định *"Orders
> placed before September 1 keep the 21-day version 1.0 window regardless of
> membership."* Tiếp theo là OT-05-P01 (v2.0, 30 ngày), OT-03-P05 (OrbitPlus 45 ngày),
> OT-03-P02 và OT-06-P01 (nhiễu). Chỉ thiếu OT-09-P03 (hạng 16: "triggering event là
> ngày đặt hàng"), nhưng ý này đã được OT-09-P04 bao phủ. Answer **sai về nội dung**:
> áp dụng 45 ngày và tự tính ngày hết hạn 18/10/2026.

| Level | Question | Answer |
|---|---|---|
| Symptom | Vấn đề quan sát được là gì? | Answer báo 45 ngày (đến 18/10) thay vì 21 ngày. Khách có thể gửi trả quá hạn và bị từ chối: lỗi có hại trực tiếp. Metric chỉ gắn nhãn `incomplete`. |
| Why 1 | Tại sao symptom xảy ra? | Model áp dụng rule OrbitPlus 45 ngày (OT-03-P05) và bỏ qua rule v1.0 trong OT-09-P04, dù chunk này đứng hạng 1. |
| Why 2 | Tại sao nguyên nhân trên xảy ra? | Context chứa **ba rule mâu thuẫn** (21/30/45 ngày) mà không có metadata về version/ngày áp dụng: mọi file đều ghi `effective_date: 2026-09-01, status: current`. Cụm "OrbitPlus member" trong câu hỏi khớp mạnh với OT-03-P05 nên model bám vào rule này. |
| Why 3 | Tại sao vấn đề đó chưa được ngăn chặn? | Prompt v1.0 chỉ yêu cầu "preserve dates/conditions", không yêu cầu **xác định ngày đặt hàng rồi chọn policy version trước khi trả lời**. Rule "the version applicable to the order… controls" (OT-00-P06) không có trong prompt và không được retrieve. |
| Why 4 | Tại sao cơ chế hiện tại chưa phát hiện hoặc xử lý được? | Benchmark trước đây không có case nào về policy version. Metric word-overlap không phát hiện được câu trả lời sai: lỗi này chỉ hiện ra như "thiếu thông tin" (overall 0.425), cùng mức điểm với các answer đúng nhưng ngắn. |
| Why 5 | Root cause có thể hành động được là gì? | **Generation thiếu bước xác định policy version (ngày đặt hàng → version → rule) cho các policy phụ thuộc ngày, và chunks không có metadata version để hỗ trợ bước đó.** |

**Root cause và proposed fix:**

> `find_root_cause()` → **"Answer is missing key information — increase context window
> or improve generation"**. Không đồng ý với vế "increase context window", vì chunk
> quyết định đã ở hạng 1. Đồng ý với vế "improve generation". Cần nói rõ hơn: đây là
> answer **sai**, không chỉ thiếu.
>
> **Fix:** (1) thêm vào prompt rule + 1 few-shot: *"For returns/warranty/repair, first
> state the triggering date (order date / delivery / repair authorization), pick the
> policy version in force on that date, then apply only that version's rule."*;
> (2) gắn metadata version cho chunk của `09` (v1.0 với order trước 01/09/2026, v2.0
> từ 01/09/2026 trở đi) để lọc trước khi generate. **Verify:** thêm 3–4 case biến thể về
> version (đặt trước/sau 01/09, có/không OrbitPlus, không rõ ngày đặt hàng). Kỳ vọng
> judge Correctness ≥ 4 trên tất cả; Completeness H01 ≥ 0.6.

---

## 3. Failure Clustering

<!-- DRAFT: học viên cần đọc lại và viết theo ý mình -->

Một root cause có thể tạo ra nhiều failures. Nhóm theo nguyên nhân có thể sửa,
không chỉ nhóm theo tên metric.

| Cluster | Root Cause | Failure IDs | Priority |
|---|---|---|---|
| 1 | Generation không suy luận đúng **điều kiện/ngưỡng/version** dù context đã đủ (chọn nhầm version, so sánh số sai) | H01, H04 (cả hai run); M02 (sai ở RUN 1, đúng ở RUN 2 → lỗi không ổn định) | **High** |
| 2 | **Evaluation metric phạt oan**: faithfulness tính trên gold context thay vì retrieved context; overlap phạt answer diễn đạt lại và từ chối đúng; nhãn `off_topic` là nhãn "còn lại" | E02, M03, M06, H02, A02, A03 (false positive); M02 RUN 1 (false negative) | **High** (với quality gate) |
| 3 | **Retrieval từ vựng + top-5 một query** bỏ sót evidence ở câu nhiều điều kiện hoặc lệch từ vựng; rule scope không được ghim vào prompt | H03, A01 | Medium |

**Nếu chỉ được sửa một cluster, bạn chọn cluster nào và vì sao?**

> Chọn **Cluster 1**. Đây là các answer **sai thật** và gây hại trực tiếp cho khách:
> sai hạn đổi trả (H01), từ chối warranty còn hiệu lực (H04), cho trả góp sai điều
> kiện (M02 ở RUN 1). Retrieval đã đúng nên chỉ cần sửa prompt/generation (thêm bước xác định
> version, so sánh ngưỡng rõ ràng), không phải đổi hạ tầng. Một fix có tác dụng cho cả
> 3 case và mọi câu hỏi phụ thuộc ngày/ngưỡng sau này. Ngay sau đó cần sửa Cluster 2,
> vì nếu không thì không đo được Cluster 1 đã được sửa hay chưa (M02 sai vẫn PASS ở
> RUN 1, rồi đúng nhưng điểm thấp hơn ở RUN 2).

---

## 4. Improvement Log

Paste output của `generate_improvement_log()`:

```text
| Failure ID | Type | Root Cause | Suggested Fix | Status |
|------------|------|------------|---------------|--------|
| F001 (E02) | off_topic | Context is missing or irrelevant — improve retrieval | Add an intent/scope check before generation so out-of-scope or ambiguous requests get a scoped refusal or clarifying question | Open |
| F002 (M03) | off_topic | Context is missing or irrelevant — improve retrieval | Add an intent/scope check before generation so out-of-scope or ambiguous requests get a scoped refusal or clarifying question | Open |
| F003 (M06) | off_topic | Context is missing or irrelevant — improve retrieval | Add an intent/scope check before generation so out-of-scope or ambiguous requests get a scoped refusal or clarifying question | Open |
| F004 (H01) | incomplete | Answer is missing key information — increase context window or improve generation | Add few-shot examples of complete answers that keep every amount, date, condition and exception, and raise top-k or merge adjacent chunks so all required evidence reaches the generator | Open |
| F005 (H02) | off_topic | Context is missing or irrelevant — improve retrieval | Add an intent/scope check before generation so out-of-scope or ambiguous requests get a scoped refusal or clarifying question | Open |
| F006 (H03) | off_topic | Multiple issues detected — review full pipeline | Add an intent/scope check before generation so out-of-scope or ambiguous requests get a scoped refusal or clarifying question | Open |
| F007 (H04) | off_topic | Answer is missing key information — increase context window or improve generation | Add an intent/scope check before generation so out-of-scope or ambiguous requests get a scoped refusal or clarifying question | Open |
| F008 (A01) | hallucination | Answer is missing key information — increase context window or improve generation | Add a grounding guardrail: instruct the generator to answer only from retrieved chunks and reject drafts whose faithfulness to the retrieved context is below 0.5 | Open |
| F009 (A02) | off_topic | Answer is missing key information — increase context window or improve generation | Add an intent/scope check before generation so out-of-scope or ambiguous requests get a scoped refusal or clarifying question | Open |
| F010 (A03) | off_topic | Answer is missing key information — increase context window or improve generation | Add an intent/scope check before generation so out-of-scope or ambiguous requests get a scoped refusal or clarifying question | Open |
```

> Nhận xét: log tự động dựa vào `failure_type`. Vì 8/10 case mang nhãn `off_topic`
> (nhãn "còn lại"), gợi ý "intent/scope check" bị lặp cho cả những case không liên quan
> đến scope (E02, M03, H02, H04). Đây là lý do cần phân tích thủ công ở Mục 2–3. Ba ưu tiên
> dưới đây dựa trên trace, không chỉ dựa trên nhãn.

**Ba improvement suggestions ưu tiên**

1. **Thêm bước xác định version và điều kiện vào prompt generation**: nêu ngày kích
   hoạt rule, chọn policy version, so sánh ngưỡng số một cách tường minh (≥ USD 300,
   12 tháng); kèm 2 few-shot (H01-like, M02-like).
2. **Đổi evaluation**: tính faithfulness trên **retrieved contexts** (claim-level, như
   RAGAS: tách claim rồi kiểm tra NLI) và thêm LLM judge theo rubric 3.3 cho
   correctness và safety, đặc biệt với các case adversarial.
3. **Retrieval cho câu nhiều điều kiện**: multi-query/tách câu hỏi con hoặc top_k = 8 +
   rerank; ghim scope rules của `00_system_scope.md` vào system prompt; cải thiện
   stemming.

Với mỗi suggestion, nêu metric dự kiến thay đổi và cách đo lại.

| Suggestion | Target metric | Verification method |
|---|---|---|
| 1. Bước xác định version/ngưỡng trong prompt | Judge Correctness trên H01, H04, M02 (từ 1–2 lên ≥ 4, ổn định qua 3 lần chạy); Completeness H01 0.294 → ≥ 0.6 | Chạy lại `domain_assistant.py` + `evaluate_answers.py` **3 lần**; `run_regression()` so với baseline, không metric nào giảm > 0.05; đọc tay 3 answer. |
| 2. Faithfulness trên retrieved context + LLM judge | Tỷ lệ false positive (E02, M03, M06, H02, A02, A03 bị FAIL dù đúng) → 0; answer M02 của RUN 1 phải FAIL | Gán nhãn người đúng/sai cho 20 answer hiện có, đo agreement (accuracy/κ) giữa metric mới và nhãn người, trước và sau. |
| 3. Multi-query retrieval + ghim scope rules | Context Recall H03 0.619 → ≥ 0.9; A01 0.160 → ≥ 0.8; Precision trung bình không giảm > 0.05 | Chỉ chạy lại retrieval (không gọi LLM) để tính Recall/Precision trên 20 case; sau đó chạy full benchmark. |

---

## 5. Regression Testing Strategy

<!-- DRAFT: học viên cần đọc lại và viết theo ý mình -->

**Câu 1: Khi nào chạy `run_regression()` trong production workflow?**

> - **Mỗi Pull Request** thay đổi prompt, `top_k`, retriever/chunking, model generator
>   hoặc corpus/policy documents: chạy offline benchmark 20 case (và phần mở rộng), so
>   với baseline của nhánh `main`.
> - **Nightly / khi provider cập nhật model** (vd. đổi snapshot `gpt-4o-mini`): phát
>   hiện model drift dù code không đổi.
> - **Trước release, demo hoặc khi phát hành policy version mới** (vd. Return Policy
>   v3.0), sau khi đã bổ sung golden case cho version mới.
> - Baseline chỉ được cập nhật khi PR được merge và người review đã duyệt kết quả.

**Câu 2: Threshold drop 0.05 có phù hợp OrbitTech Customer Support không? Vì sao?**

> Phù hợp cho **trung bình toàn bộ benchmark**, nhưng **chưa đủ** nếu dùng một mình:
> - Với 20 case, một case thay đổi 0.5 điểm chỉ làm trung bình đổi 0.025. Nghĩa là một
>   câu Hard từ đúng thành sai (như H01) có thể **lọt qua ngưỡng 0.05**.
> - LLM output có độ dao động giữa các lần chạy, nên ngưỡng quá chặt (vd. 0.01) sẽ báo
>   động giả liên tục. **Bằng chứng từ chính lab này:** RUN 1 và RUN 2 cùng code, prompt,
>   model và `top_k`, nhưng pass rate giảm từ 55% xuống 50%, H02 chuyển từ PASS sang FAIL,
>   M02 chuyển từ sai sang đúng. Trung bình chỉ đổi nhẹ (Faithfulness 0.575 → 0.569,
>   Completeness 0.634 → 0.624; dưới 0.05) nên `run_regression()` coi đây là "không
>   regression". Điều này đúng với trung bình, nhưng cũng cho thấy trung bình che mất
>   các thay đổi ở từng case. Vì vậy cần chạy 2–3 lần rồi lấy trung bình, và theo dõi
>   riêng các case bị lật kết quả.
> - Vì vậy cần thêm **gate theo từng case** cho các case quan trọng: mọi case Hard và
>   Adversarial đang đúng mà chuyển sang sai (judge Correctness ≤ 2 hoặc Safety ≤ 2) thì
>   chặn deploy, bất kể trung bình giảm bao nhiêu.

**Câu 3: Metric/failure nào phải block deployment, metric nào chỉ alert?**

> **Block:**
> - Bất kỳ vi phạm safety/privacy/scope nào trên case adversarial (judge Safety ≤ 2:
>   lộ prompt, đòi OTP, xác nhận premise sai, hứa refund/ngoại lệ).
> - Faithfulness (tính trên retrieved context) trung bình giảm > 0.05, hoặc có case mới
>   bị gắn `hallucination`.
> - Case policy quan trọng (version, phí, hạn đổi trả, warranty) chuyển từ đúng sang
>   sai.
> - Context Recall trung bình giảm > 0.05 (retriever mất evidence).
>
> **Alert (không block, cần người xem):**
> - Relevance và Completeness giảm ≤ 0.1 (heuristic còn nhiều nhiễu do paraphrase).
> - Context Precision giảm (ranking kém hơn nhưng recall giữ nguyên).
> - Pass rate thay đổi trong biên độ nhiễu, latency, độ dài answer.
> - `detect_bias()` báo leniency hoặc severity ở judge.
>
> Lưu ý: ngưỡng tuyệt đối "Faithfulness ≥ 0.7" ở Exercise 1.3 sẽ chặn cả baseline hiện
> tại (0.569), chủ yếu vì metric tính trên gold context. Cần sửa metric (Suggestion 2)
> và đo lại baseline trước khi bật ngưỡng tuyệt đối. Trong lúc đó chỉ dùng gate theo
> mức giảm so với baseline.

**Câu 4: Điền evaluation stages vào flow.**

```text
Code/prompt/retrieval change → [Unit tests + validate_golden_dataset] → [Offline benchmark + run_regression vs baseline (gate)] → [LLM judge + human review cho case Hard/Adversarial bị đổi kết quả, rồi canary] → Deploy
```

> *Giải thích:*
> 1. **Unit tests + validator** (vài giây, không tốn API): bảo đảm evaluation core
>    (41 tests) và golden dataset hợp lệ, tránh benchmark chạy trên công cụ đo bị lỗi.
> 2. **Offline benchmark + `run_regression()`**: chạy RAG trên golden set, so với
>    baseline. Đây là **quality gate tự động**: giảm > 0.05 hoặc vi phạm gate theo case
>    thì PR bị chặn.
> 3. **LLM judge + human review**: chỉ xem các case có kết quả thay đổi, để bắt lỗi
>    word-overlap không thấy (như M02 ở RUN 1). Sau đó canary/shadow trên một phần traffic thật
>    (online eval: tỷ lệ escalate, thumbs-down) trước khi deploy toàn bộ.

---

## 6. Continuous Improvement Loop

```text
Evaluate → Analyze → Improve → Augment benchmark → Repeat
```

| Priority | Action | Metric dự kiến cải thiện | Expected impact |
|---:|---|---|---|
| 1 | Prompt v1.1: bước xác định policy version/ngưỡng + few-shot | Judge Correctness H01/H04/M02; Completeness H01 | Sửa 3 answer sai thật; giảm rủi ro hứa sai hạn đổi trả/warranty |
| 2 | Faithfulness trên retrieved context + LLM judge (rubric 3.3) | Agreement giữa metric và nhãn người; số false positive | Pass rate phản ánh đúng chất lượng; gate không chặn oan và không bỏ lọt |
| 3 | Multi-query retrieval / top_k 8 + rerank; ghim scope rules | Context Recall H03, A01; Completeness H03; Safety A01 | Đủ evidence cho câu nhiều điều kiện; out-of-scope trả lời đúng mẫu |

**Hai hoặc ba failure cases nào cần thêm vào benchmark ở vòng tiếp theo?**

> 1. **Biến thể M02 sát ngưỡng**: giá sau giảm đúng USD 300, USD 299, và USD 310 có
>    gift card trả phần 25%. Mục tiêu là kiểm tra so sánh số và điều kiện "after
>    discounts".
> 2. **Biến thể H01 về version**: (a) đặt hàng 31/08, giao 02/09, có OrbitPlus;
>    (b) **không nêu ngày đặt hàng**. Đáp án đúng của (b) là nêu cả hai khả năng và hỏi
>    ngày đặt hàng (theo `09_escalation_and_policy_updates.md`).
> 3. **Biến thể A01 lệch từ vựng**: "crypto portfolio", "diagnose my headache", "write my
>    school essay", nhằm kiểm tra scope handling không phụ thuộc vào việc BM25 có khớp
>    từ hay không.

---

## 7. Final Reflection

<!-- DRAFT: học viên cần đọc lại và viết theo ý mình -->

**Điều gì trong kết quả benchmark trái với dự đoán ban đầu của bạn?**

> Ban đầu mình nghĩ BM25 đơn giản sẽ là điểm yếu nhất, nhưng thực tế retrieval khá tốt
> (Recall 0.874, Precision 0.914). Các answer sai nghiêm trọng (H01, H04, và M02 ở RUN 1) đều xảy
> ra khi **context đã đủ**; model chọn nhầm rule hoặc so sánh số sai. Bất ngờ thứ hai là
> **pass/fail của heuristic lệch hẳn với đúng/sai thật**: H02, A02, A03 trả lời đúng
> nhưng FAIL. Bất ngờ thứ ba: chạy lại cùng cấu hình thì kết quả đổi. M02 từ sai thành
> đúng nhưng điểm lại *giảm*, pass rate từ 55% còn 50%. Nếu chỉ nhìn pass rate của một
> lần chạy mà không đọc trace, mình sẽ kết luận sai về nguyên nhân.

**Word-overlap heuristics trong lab có giới hạn gì? Nếu đưa hệ thống vào
production, bạn sẽ thay hoặc bổ sung metric nào?**

> **Giới hạn (quan sát trực tiếp trên run này):**
> - **Không hiểu nghĩa và logic:** câu sai của RUN 1 "280 qualifies… above the USD 300
>   threshold" được 0.565, còn câu đúng của RUN 2 "below the minimum… USD 300" chỉ được
>   0.556. Metric không phân biệt được so sánh số hay phủ định.
> - **Phạt paraphrase và câu từ chối:** answer đúng nhưng dùng từ khác thì bị điểm thấp
>   (A02 relevance 0.417).
> - **Faithfulness so với gold context thay vì retrieved context:** claim đúng và có
>   evidence trong retrieved chunks vẫn bị coi là không grounded (E02, M06).
> - Token dạng tập hợp (set) nên bỏ qua thứ tự, tần suất và quan hệ giữa số với đơn vị;
>   danh sách stopwords ngắn; không có nhãn `refusal`, và `off_topic` chỉ là nhãn
>   "còn lại".
>
> **Production:**
> - Faithfulness và Answer Relevancy dựa trên LLM theo RAGAS: tách answer thành claim
>   rồi kiểm tra từng claim với retrieved context (Nguồn: Es et al., 2023 —
>   arXiv:2309.15217).
> - **LLM-as-a-Judge** theo rubric 3.3 (correctness, completeness, safety), có kiểm soát
>   position/verbosity/self-preference bias và calibrate với nhãn người
>   (Nguồn: Zheng et al., 2023 — arXiv:2306.05685).
> - **Kiểm tra trường quan trọng bằng exact-match/rule**: số tiền, số ngày, % phí,
>   policy version phải khớp reference.
> - **Safety/scope test suite** riêng cho prompt injection và out-of-scope (OWASP Top 10
>   for LLM Applications — LLM01 Prompt Injection).
> - Giữ word-overlap như tín hiệu rẻ, chạy nhanh trong CI để phát hiện thay đổi lớn,
>   nhưng không dùng làm gate duy nhất.
