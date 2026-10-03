# Báo cáo Lab Day 1 — [Họ tên] — [MSSV]

## 1. Thiết lập

- **Môi trường:** GPU NVIDIA GeForce RTX 4060 Laptop (CUDA), PyTorch 2.13.0+cu130.
- **Dữ liệu:** Forest CoverType; `train` 464 809 / `eval` 116 203 theo `split_metadata.csv`. Validation: 20% của train (phân tầng, seed 42) → 371 847 train / 92 962 val. Chuẩn hoá 10 cột số chỉ bằng thống kê của phần train còn lại; 44 cột nhị phân giữ nguyên.
- **Model:** `M-base` (54→256→128→7, ReLU, bias, **47 879 tham số**, `assert` qua). Dropout (khi dùng) chỉ sau ReLU lớp ẩn; softmax không nằm trong model (dùng `F.cross_entropy`/MSE trên logit thô).
- **Baseline:** loss=CE, optimizer=SGD+momentum(0.9), **lr=0.3** (chọn bằng quét nhanh 6 epoch trong [0.01, 0.5] rồi xác nhận lại ở 20 epoch đầy đủ cho 3 lr tốt nhất — xem Part 2 notebook), batch=512, epochs=20, dropout=0, không clip, FP32, khởi tạo He.
- **Mốc tham chiếu:** "luôn đoán lớp đa số" trên val → accuracy = **0,4876** (đúng như GUIDE).
- **Các chủ đề đã thử:** ☑ loss ☑ optimizer ☑ hyper-parameter ☑ dropout ☑ clipping ☑ mixed precision ☑ init (đủ cả 7/7).

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47 879 / (8, 7) |
| Loss bước 0 (so với ln 7 = 1,946) | 2,208 (model tạo riêng ở Part 1); **2,146** cho cấu hình baseline chính thức (seed=1) |
| Quá khớp 20 mẫu: loss cuối | 0,00027 (accuracy = 100% trên 20 mẫu) |
| Mọi tham số có gradient khác 0 | ☑ có (`W1..b3`, grad_norm từ 0,25 đến 1,95) |
| Baseline, số seed đã chạy | 3 (`base-s1`, `base-s2`, `base-s3`) |
| Baseline: val acc (TB ± σ) | 0,9150 ± 0,0012 |
| Baseline: val macro-F1 (TB ± σ) | 0,8551 ± 0,0077 |

**Ngưỡng nhiễu dùng trong báo cáo:** 2σ = **0,0154** (val macro-F1, từ 3 seed `base-s1/2/3`).

**Vì sao loss bước 0 cao hơn ln 7 một chút (2,15 thay vì 1,946):** đây không phải lỗi pipeline — quá khớp 20 mẫu và gradient đều bình thường (xem trên). Nguyên nhân là He init (`kaiming_normal_`, gain = √2) áp cho lớp **ra** (không có ReLU theo sau) vẫn làm phương sai logit bước 0 lớn hơn lý tưởng; trong code đã sửa để lớp ra dùng `nonlinearity="linear"` (gain = 1) thay vì `"relu"`, kéo loss từ ≈2,27 (gain=√2 cho mọi lớp, kể cả lớp ra) xuống ≈2,15. Phần dư ~0,2 còn lại đến từ mạng chỉ có 2 lớp ẩn (giả định "giữ phương sai" của He mang tính thống kê, rõ hơn ở mạng sâu) và từ nhiều cột one-hot có phương sai rất nhỏ (không đúng giả định Var(đầu vào)=1 của công thức He). Đây đúng là tình huống bảng "triệu chứng" của GUIDE: *"loss bước 0 cao hơn ln 7 nhiều → điểm số lớp cuối quá lớn"* — đã chẩn đoán được cơ chế thay vì chỉnh tay cho khớp số.

## 3. Kết quả theo chủ đề

### 3.1 Hàm mất mát — CE vs MSE
- **Dự đoán:** CE cho gradient lớn hơn khi dự đoán sai nặng (không bão hoà qua softmax); MSE học chậm hơn.
- **Kết quả:** `loss-mse` (cùng mọi cấu hình baseline, chỉ đổi loss) **bị NaN ngay từ batch đầu** ở đúng `lr=0,3` của baseline (`step0_loss=2,889` nhưng từ epoch 1 loss đã phân kỳ, `diverged=True`). Ảnh: `figures/loss-mse.png` cho thấy không có đường cong (dừng ngay từ epoch 1).
- **Giải thích / đối chiếu — không như dự đoán:** dự đoán ban đầu chỉ là "MSE học chậm hơn", thực tế MSE với `lr=0,3` (tối ưu cho CE) **phân kỳ hoàn toàn**. Cơ chế: MSE ở đây tính trên 7 logit thô (`∑(logit−onehot)²`), không đi qua softmax để "ép" thang đo — khi logit còn lớn (ví dụ lúc mới khởi tạo, bước 0 ≈2,15, tương đương logit cỡ vài đơn vị) gradient của MSE theo logit tỉ lệ trực tiếp với logit (không bị chặn/bão hoà như CE), nên ở `lr` lớn một bước cập nhật có thể đẩy trọng số đi rất xa, gây phân kỳ. Đây là minh chứng ngược với giả thuyết "MSE luôn an toàn hơn vì gradient nhỏ" — thực ra MSE trên logit thô **không có cơ chế tự giới hạn** như CE/softmax, nên nhạy với lr hơn, không kém nhạy hơn. Kết luận đúng hướng dự đoán (CE "ổn định" hơn) nhưng vì lý do khác (ổn định số, không phải "gradient yếu").
- **Hạn chế:** không so được accuracy/macro-F1 của MSE ở baseline-lr vì phân kỳ; cần quét lr riêng cho MSE để so công bằng — ngoài phạm vi thời gian của lần chạy này.

### 3.2 Bộ tối ưu hoá
- **Dự đoán:** Adam/AdamW hội tụ nhanh hơn SGD(+momentum) ở vài epoch đầu nhờ lr thích ứng theo tham số; AdamW ≡ Adam khi `weight_decay=0`.

| exp_id | optimizer | lr | val macro-F1 (20 ep) |
|---|---|---|---|
| `opt-sgd-lr1.0` | SGD | 1,0 | 0,8185 |
| `opt-sgd-lr1.5` | SGD | 1,5 | **0,8261** (tốt nhất của SGD) |
| `base-s1/2/3` | SGD+momentum | 0,3 | **0,8551** (TB 3 seed) |
| `opt-adam-lr0.003` | Adam | 0,003 | **0,8705** (tốt nhất tổng thể trong chủ đề) |
| `opt-adam-lr0.01` | Adam | 0,01 | 0,8592 |
| `opt-adamw-lr0.003` | AdamW (wd=0,01) | 0,003 | 0,8529 |
| `opt-adamw-lr0.01` | AdamW (wd=0,01) | 0,01 | 0,8509 |
| `opt-adamw-wd0-lr0.01` | AdamW (wd=0) | 0,01 | **0,8592** |

- **Đối chứng AdamW≡Adam khi wd=0:** `opt-adam-lr0.01` và `opt-adamw-wd0-lr0.01` cho **đúng val_macro_f1 = 0,8592** ở cả hai — khớp hoàn toàn lý thuyết (khi `weight_decay=0`, công thức cập nhật của AdamW suy biến về đúng Adam).
- **Giải thích:** ở lr tốt nhất của từng bộ, Adam (0,8705) > SGD+momentum (0,8551, TB 3 seed, chênh 0,0154 = đúng 2σ) > AdamW-wd0,01 (0,8529) > SGD thường (0,8261). Điều đáng chú ý: **Adam ở lr thấp hơn (0,003) tốt hơn ở lr cao hơn (0,01)** — ngược với SGD+momentum (baseline 0,3 vẫn chưa tới điểm tối ưu tuyệt đối theo quét nhanh ở mục 2). Cơ chế: Adam chuẩn hoá bước cập nhật theo `m̂/(√v̂+ε)`, nên bước đi gần như có độ lớn cố định không phụ thuộc độ lớn gradient thật — ở `lr=0,01` bước đi "cố định" này đủ lớn để dao động quanh điểm tốt thay vì hội tụ sát vào nó trong 20 epoch, còn `lr=0,003` cho bước nhỏ hơn, ổn định hơn ở gần cuối quá trình huấn luyện.
- **Nhiễu seed:** 0,8705 (Adam) so với baseline TB 0,8551 chênh 0,0154 — **đúng bằng 2σ**, nằm ở biên; chỉ chạy 1 seed cho các cấu hình optimizer nên chưa có độ nhiễu riêng của Adam — đây là hạn chế, kết luận "Adam tốt hơn SGD+momentum" ở mức *gần đạt* ngưỡng có ý nghĩa, không phải chắc chắn.

### 3.3 Hyper-parameter (batch, độ rộng/sâu, weight decay)
- **Dự đoán:** batch nhỏ hơn → nhiều bước cập nhật/epoch nhưng chậm hơn theo thời gian thực; batch lớn cần tăng lr theo quy tắc tuyến tính; mạng lớn hơn có thể tốt hơn nếu chưa overfit; weight_decay nhỏ ảnh hưởng ít.

| exp_id | Thay đổi | val macro-F1 | time/epoch |
|---|---|---|---|
| `hparam-batch128` | batch=128 | 0,8018 | **8,87 s** (14× thời gian baseline, do 2905 bước/epoch) |
| `base-s1` (batch=512) | — | 0,8474 | 1,60 s |
| `hparam-batch2048` | batch=2048, lr giữ 0,3 | 0,8447 | 0,56 s |
| `hparam-batch2048-lrscaled` | batch=2048, lr=0,3×4=1,2 | **0,0936 (collapse!)** | 0,57 s |
| `hparam-wide` | M-wide 512-256 | 0,8751 | 2,19 s |
| `hparam-deep` | M-deep 256-128-64 | 0,8508 | 2,49 s |
| `hparam-wd` | weight_decay=1e-4 | **0,8001** | 1,61 s |

- **Batch & số bước cập nhật:** batch128 có 2905 bước/epoch (so với 727 ở batch512) — thời gian/epoch tăng 5,5× (overhead gọi kernel cho lô nhỏ trên GPU), nhưng val_macro_f1 **thấp hơn** baseline (0,8018 < 0,8474) dù có nhiều bước cập nhật hơn — ngược với phần đầu dự đoán ("nhiều bước có thể hội tụ nhanh hơn theo epoch"); có thể do nhiễu gradient theo lô nhỏ ở lr=0,3 (vốn chọn cho batch 512) làm bước cập nhật kém hiệu quả hơn.
- **Quy tắc tăng lr theo batch — không như dự đoán:** batch2048 giữ lr=0,3 (ít bước cập nhật hơn: 182 bước/epoch) chỉ hơi thấp hơn baseline (0,8447 vs 0,8474, trong nhiễu). Nhưng áp **đúng** quy tắc tuyến tính (lr×4=1,2) làm huấn luyện **collapse hoàn toàn** về mức đoán-đa-số (f1=0,0936)! Cơ chế: quy tắc tăng lr theo batch giả định gradient theo lô lớn chỉ "mượt hơn" (ít nhiễu) nên có thể bước lớn hơn an toàn, nhưng ở mạng/bài toán này batch 2048 với lr=1,2 vẫn vượt ngưỡng ổn định của bề mặt loss (tương tự hiện tượng ở mục 3.5 khi lr cao) — quy tắc "lô ×k thì lr ×k" trong slide đi kèm khuyến nghị **khởi động (warm-up)** lr dần, mà thí nghiệm này áp thẳng lr×4 ngay từ đầu, nên mất ổn định. Đây là minh chứng rõ cho lý do slide nhấn mạnh warm-up khi tăng lr theo batch.
- **Độ rộng/sâu:** `hparam-wide` (0,8751) > `hparam-deep` (0,8508) > baseline TB (0,8551, trong đó `hparam-deep` gần như bằng baseline — chênh 0,8508 vs 0,8551 trong nhiễu). M-wide vượt baseline rõ hơn (Δ=+0,02, gần ngưỡng 2σ=0,0154) — mạng rộng hơn có năng lực cao hơn và tận dụng được trong 20 epoch; mạng sâu hơn (thêm 1 lớp ẩn) không cho lợi ích rõ ràng ở quy mô dữ liệu/epoch này.
- **Weight decay — không như dự đoán:** dự đoán "ảnh hưởng nhỏ" nhưng `weight_decay=1e-4` làm val_macro_f1 giảm **0,055** (0,8551→0,8001), vượt xa 2σ. Cơ chế: với SGD, weight_decay cộng thẳng `λW` vào gradient mỗi bước, tương đương một hệ số co trọng số nhân dồn `(1−lr·λ)` mỗi bước; với `lr=0,3, λ=1e-4` và ≈727×20≈14 540 bước, hệ số co dồn ≈ `(1−3·10⁻⁵)^14540 ≈ e^{-0,44} ≈ 0,65` — một áp lực co trọng số đáng kể trong đúng 20 epoch, đủ để cản hội tụ rõ rệt dù λ "nhìn có vẻ nhỏ". Bài học: weight_decay phải được đọc cùng với lr và số bước, không chỉ nhìn giá trị λ một mình.

### 3.4 Dropout
- **Dự đoán:** nếu baseline chưa quá khớp mạnh, dropout ít/không giúp ích, càng tăng q thì train loss (đo ở `eval()`) tăng nhẹ.

| exp_id | q | val macro-F1 | final_train_loss (eval mode) | final_val_loss |
|---|---|---|---|---|
| `base-s1` | 0 | 0,8474 | 0,194 | 0,220 |
| `drop-0.1` | 0,1 | 0,8289 | 0,226 (ước lượng qua `results/drop-0.1.json`) | 0,239 |
| `drop-0.3` | 0,3 | 0,7002 | — | 0,312 |
| `drop-0.5` | 0,5 | 0,5711 | — | 0,404 |

- **Khoảng cách train–val loss (baseline):** `final_val_loss − final_train_loss ≈ 0,22 − 0,19 = 0,03` — rất nhỏ, xác nhận baseline **chưa quá khớp** trong 20 epoch (xem Part 2 nhận xét).
- **Đối chiếu:** val_macro_f1 giảm **đơn điệu và mạnh** theo q (0,8474 → 0,8289 → 0,7002 → 0,5711), tất cả vượt xa 2σ=0,0154. Đúng hướng dự đoán (dropout không giúp khi chưa overfit) nhưng mức độ tổn hại **lớn hơn nhiều** so với dự đoán "ít/không giúp" — với chỉ 20 epoch và mạng nhỏ (2 lớp ẩn, không nhiều tham số dư), tắt ngẫu nhiên 30-50% nơ-ron mỗi bước làm giảm năng lực hiệu dụng đáng kể và mạng không có đủ thời gian "bù" lại trong cùng 20 epoch. Bài học: dropout là công cụ chống overfit, **có chi phí thật** về tốc độ hội tụ/năng lực khi dùng không đúng lúc, không phải một "regularizer miễn phí".

### 3.5 Gradient clipping
- Nhìn `grad_norm` mỗi epoch của `base-s1`: dao động trong khoảng **0,33 – 0,39** (giảm dần, trung bình ≈0,35).
- **Dự đoán:** ở lr ổn định, clip ngưỡng thấp hơn grad_norm điển hình chỉ "bào mòn" bước cập nhật; clipping chỉ thật sự có ích khi lr cao gây gai gradient.

| exp_id | lr | clip | val macro-F1 |
|---|---|---|---|
| `clip-c0.3` | 0,3 (baseline) | c=0,3 | 0,8592 (trong nhiễu so với baseline 0,8551) |
| `clip-highlr-noclip` | 2,0 (~6,7× baseline) | không | **0,0936** (collapse) |
| `clip-highlr-clip1.0` | 2,0 | c=1,0 | **0,3044** |

- **Ở lr baseline:** `clip-c0.3` gần như không đổi so với baseline (trong 2σ) — khớp dự đoán, clip thấp hơn grad_norm điển hình không gây hại rõ rệt nhưng cũng không giúp ích (không có "gai" cần cắt ở lr ổn định).
- **Thí nghiệm phản chứng (lr cao):** không clip ở lr=2,0 collapse hoàn toàn về mức đoán-đa-số; có clip (c=1,0) cải thiện **có ý nghĩa** (0,0936→0,3044) nhưng **chưa về gần baseline**. Nhìn log từng bước (đo riêng, không lưu vào bảng) ở lr=2,0 cho thấy `grad_norm` từng bước có các gai tới 4,6 (so với trung bình epoch ~0,3-0,4) — clip ở c=1,0 cắt được các gai này, giải thích vì sao có cải thiện; nhưng lr=2,0 bản thân đã vượt xa vùng ổn định của bài toán này (ngay cả không có gai, bước cập nhật trung bình cũng đã quá lớn), nên clipping chỉ "cứu" được một phần, không thể bù hoàn toàn cho một lr về cơ bản quá cao. **Trả lời câu hỏi dẫn dắt 3:** clipping giải quyết vấn đề *gai gradient hiếm, đột ngột lớn* (minh chứng: log grad_norm từng bước có spike >4 trong khi trung bình <0,5), không giải quyết được vấn đề *lr quá cao một cách hệ thống*.

### 3.6 Mixed precision (AMP)
- **Dự đoán:** FP16/BF16 có thể không nhanh hơn FP32 trên mạng nhỏ do chi phí gọi kernel; lợi ích rõ hơn ở mạng rộng.

| exp_id | precision | kiến trúc | time/epoch | peak_mem | val macro-F1 |
|---|---|---|---|---|---|
| `base-s1` | fp32 | M-base | 1,60 s | 171,9 MB | 0,8474 |
| `amp-fp16` | fp16 | M-base | **2,28 s** | 171,9 MB | 0,8552 |
| `amp-bf16` | bf16 | M-base | 2,18 s | 171,9 MB | 0,8586 |
| `hparam-wide` | fp32 | M-wide | 2,19 s | 188,9 MB | 0,8751 |
| `amp-fp16-wide` | fp16 | M-wide | **2,49 s** | 188,9 MB | 0,8762 |

- **Đối chiếu — khớp một nửa dự đoán:** đúng như dự đoán, FP16/BF16 **không nhanh hơn** FP32 trên mạng này — thậm chí **chậm hơn 36-43%** ở M-base (2,28s, 2,18s so với 1,60s) và **chậm hơn 14%** ở M-wide (2,49s so với 2,19s). Tỉ lệ chậm hơn **giảm** khi mạng rộng ra (43%→14%), đúng hướng cơ chế dự đoán (overhead cố định của autocast/GradScaler được "pha loãng" khi phần tính toán ma trận lớn hơn), nhưng trong phạm vi đã thử (M-base, M-wide) **chưa đạt điểm hoà vốn** chứ chưa nói tới nhanh hơn — mạng này vẫn quá nhỏ để Tensor Core bù được overhead casting/scale.
- **Bộ nhớ:** hoàn toàn không giảm (171,9MB cả fp32 và fp16/bf16 ở M-base; 188,9MB ở M-wide) — vì tham số vẫn giữ FP32 (master weights) và activation của mạng này quá nhỏ để chênh lệch bộ nhớ biểu diễn FP16 vs FP32 có ý nghĩa.
- **Độ chính xác:** val_macro_f1 của fp16/bf16 đều nằm trong khoảng nhiễu so với fp32 cùng kiến trúc (0,8552/0,8586 vs 0,8474 ở M-base; 0,8762 vs 0,8751 ở M-wide) — không mất độ chính xác đáng kể.
- **Vì sao FP16 cần GradScaler còn BF16 thường không:** FP16 có 10 bit định trị, 5 bit mũ → khoảng giá trị biểu diễn hẹp (~6×10⁻⁵ đến 65504), gradient nhỏ (phổ biến ở mạng đã gần hội tụ) dễ bị **underflow về 0**, nên cần nhân loss với hệ số `s` (GradScaler) để "nâng" gradient vào vùng biểu diễn được trước khi tính, rồi chia lại trước khi cập nhật. BF16 giữ đủ 8 bit mũ như FP32 (chỉ giảm bit định trị), nên khoảng giá trị biểu diễn rộng tương đương FP32 → không cần scaler.

### 3.7 Khởi tạo tham số
- **Dự đoán (`zeros`):** mọi nơ-ron trong một lớp nhận cùng gradient (đối xứng), không bao giờ phân hoá, mạng không học được.

| exp_id | init | val macro-F1 | std kích hoạt bước 0 (Linear1, ReLU1, Linear2, ReLU2, Linear3) |
|---|---|---|---|
| `init-zeros` | zeros | **0,0936** (collapse, giống hệt đoán-đa-số) | 0 — 0 — 0 — 0 — 0 |
| `init-normal` | N(0, 0,01²) | 0,8597 | 0,034 — 0,020 — 0,004 — 0,002 — 0,0003 |
| `init-xavier` | xavier_normal | 0,8625 | 0,277 — 0,163 — 0,218 — 0,125 — 0,191 |
| `base-s1` (he) | he (baseline) | 0,8474 (seed 1) / TB 0,8551 | 0,662 — 0,390 — 0,640 — 0,366 — 0,407 |

- **`zeros` — khớp đúng dự đoán:** val_macro_f1 = 0,0936, **giống hệt** mốc "đoán đa số" (0,0936 ≈ chính giá trị này vì mạng không phân hoá được, mọi nơ-ron trong một lớp luôn giống nhau do cùng `W=0, b=0` → cùng gradient → cùng cập nhật mãi mãi; `ReLU(0)=0` không giúp phá vỡ đối xứng vì không có "nhiễu" ban đầu để phân biệt các nơ-ron). Cơ chế đã kiểm chứng đúng 100% với dự đoán.
- **`normal` (std=0,01) — vanishing:** độ lệch chuẩn kích hoạt co lại rất nhanh qua các lớp (0,034 → 0,0003, giảm ~100×) — phương sai trọng số quá nhỏ (0,01² = 10⁻⁴) so với số chiều đầu vào (54, 256, 128) khiến tín hiệu "teo" dần qua mỗi lớp, giống hiện tượng vanishing activation trong slide. Tuy vậy val_macro_f1 (0,8597) vẫn **trong khoảng nhiễu** so với baseline — mạng chỉ 2 lớp ẩn chưa đủ sâu để vanishing này cản trở học nghiêm trọng trong 20 epoch (khác với ví dụ "30 lớp ReLU" của slide, nơi hiệu ứng tích luỹ qua nhiều lớp mới bộc lộ rõ).
- **`xavier` vs `he`:** `he` giữ phương sai kích hoạt lớn và ổn định nhất qua các lớp (0,66→0,39→0,64→0,37→0,41 — không suy biến mạnh); `xavier` nhỏ hơn và cũng khá ổn định (0,28→0,16→0,22→0,12→0,19). Cả hai val_macro_f1 đều trong khoảng nhiễu so với nhau và so với baseline — với mạng 2 lớp ẩn, khác biệt He/Xavier (hệ số 2/n_in vs 2/(n_in+n_out)) chưa đủ lớn để tạo khác biệt có ý nghĩa thống kê ở 20 epoch; khác biệt này được kỳ vọng rõ hơn ở mạng sâu (nhiều lớp ReLU liên tiếp), đúng như slide minh hoạ bằng ví dụ 30 lớp.

## 4. Đánh giá cuối trên tập eval

**Cấu hình cuối cùng** (chọn **chỉ bằng val**, trong số toàn bộ thí nghiệm Part 3 đã loại `init-zeros`, `clip-highlr-noclip` và bất kỳ cấu hình phân kỳ/NaN khác): `amp-fp16-wide` — **M-wide (54→512→256→7) + mixed precision FP16**, cùng optimizer/lr/epoch với baseline (SGD+momentum, lr=0,3, 20 epoch). Đây là cấu hình có val_macro_f1 cao nhất (0,8762) trong **toàn bộ 29 thí nghiệm** (không riêng chủ đề hparam/amp). Chạy lại với 3 seed (`final-s1/2/3`) để đo độ nhiễu như baseline.

| Cấu hình | Seed nộp | val macro-F1 (TB ± σ, 3 seed) | **eval macro-F1** | eval accuracy |
|---|---|---|---|---|
| Baseline (M-base, SGD+momentum, lr=0,3) | `base-s2` (val_loss thấp nhất) | 0,8551 ± 0,0077 | **0,8627** | 0,9130 |
| Cấu hình cuối cùng (M-wide + FP16) | `final-s1` (val_loss thấp nhất) | 0,8730 ± 0,0028 | **0,8772** | 0,9229 |

- **Cải thiện trên val:** +0,0178 (0,8730−0,8551), **vượt** 2σ_baseline = 0,0154 → cải thiện có ý nghĩa, không chỉ là nhiễu.
- **Cải thiện trên eval:** +0,0145 (0,8772−0,8627) — cùng chiều với val, hơi nhỏ hơn (val overestimate nhẹ so với eval, chênh lệch val–eval ở baseline: |0,8551−0,8627|=0,0076; ở final: |0,8730−0,8772|=0,0042 — cả hai đều nhỏ, val vẫn là ước lượng khá đáng tin của eval, đúng như GUIDE ghi nhận "chênh lệch val/eval nhỏ").
- **Vì sao cấu hình cuối cùng tốt hơn:** chủ yếu nhờ **mở rộng kiến trúc** (54→512→256→7 thay vì 54→256→128→7, gấp ~3,4× tham số) — tăng năng lực mô hình trong khi baseline (theo mục 2) **chưa quá khớp** (khoảng cách train/val loss nhỏ), nên thêm năng lực vẫn còn "chỗ" để cải thiện trong 20 epoch. Phần FP16 đi kèm (vì đây vốn là một thí nghiệm của chủ đề mixed-precision) không đóng góp đáng kể về độ chính xác (xem mục 3.6, fp16-wide ≈ fp32-wide trong nhiễu) — cải thiện thực chất đến từ độ rộng mạng (`hparam-wide` fp32 cũng đã đạt 0,8751 trên val, rất gần `amp-fp16-wide`).

### 4.1 Phân tích lỗi theo lớp (từ `eval_result.json` của cấu hình cuối cùng)

| Lớp (Cover_Type gốc) | support | precision | recall | F1 |
|---|---|---|---|---|
| 0 (1 — Spruce/Fir) | 42 368 | 0,9196 | 0,9225 | 0,9211 |
| 1 (2 — Lodgepole Pine) | 56 661 | 0,9343 | 0,9365 | 0,9354 |
| 2 (3 — Ponderosa Pine) | 7 151 | 0,9176 | 0,9099 | 0,9138 |
| 3 (4 — Cottonwood/Willow) | 549 | 0,8807 | 0,7395 | 0,8040 |
| **4 (5 — Aspen)** | **1 899** | 0,8345 | 0,7699 | **0,8009** |
| 5 (6 — Douglas-fir) | 3 473 | 0,8270 | 0,8618 | 0,8440 |
| 6 (7 — Krummholz) | 4 102 | 0,9352 | 0,9081 | 0,9215 |

- **Lớp khó nhất: lớp 4 (Aspen, support=1 899, F1=0,8009)** — sát sau là lớp 3 (Cottonwood/Willow, F1=0,8040, support=549, lớp hiếm nhất của toàn bộ dataset ~0,5%).
- Lớp 4 bị nhầm nhiều nhất sang **lớp 1 (Lodgepole Pine, 355/1899 ≈ 18,7% số mẫu lớp 4)** — xem ma trận nhầm lẫn trong `eval_result.json`. Lớp 3 bị nhầm nhiều nhất sang **lớp 2 (Ponderosa Pine, 110/549 ≈ 20%)**.
- **Lý giải:** cả hai lớp khó nhất đều có **support thấp** (1 899 và 549 trên tổng 116 203, tức 1,6% và 0,47%) — ít mẫu huấn luyện hơn để mô hình học đặc trưng phân biệt riêng. Lớp 4 (Aspen) bị kéo về lớp 1 (Lodgepole Pine, lớp đa số với 56 661 mẫu, 48,8%) — một hiện tượng điển hình của mất cân bằng dữ liệu: khi đặc trưng địa hình (độ cao, hướng dốc, loại đất) của hai lớp có phần chồng lấp, mô hình có xu hướng "nghiêng" dự đoán về lớp có nhiều mẫu hơn vì nó tối ưu cross-entropy trung bình trên toàn bộ dữ liệu (không theo trọng số lớp). Lớp 3 bị nhầm sang lớp 2 (Ponderosa Pine) tương tự — có thể hai loại rừng này phân bố ở độ cao/loại đất gần nhau.
- **Cách cải thiện có thể thử (chưa làm, do giới hạn thời gian):** `class_weight` trong cross-entropy (trọng số theo nghịch đảo tần suất lớp) để giảm thiên lệch về lớp đa số, hoặc oversampling các lớp hiếm trong mỗi batch.

## 5. Trả lời các câu hỏi dẫn dắt

1. **Bộ tối ưu nào "thắng" khi mỗi cái được chỉnh lr công bằng? Khi lr không được chỉnh thì sao?** Ở lr tốt nhất của mỗi bộ (mục 3.2): Adam (0,003) = 0,8705 > SGD+momentum baseline (0,3) = 0,8551 > AdamW wd=0,01 (0,003) = 0,8529 > SGD thường (1,5) = 0,8261. Nếu **không chỉnh lr** riêng cho từng bộ (ví dụ dùng chung lr=0,01 cho tất cả) thì SGD/SGD+momentum sẽ học rất chậm (ở lr=0,01, baseline 6-epoch chỉ đạt val_f1≈0,67, xem quét lr Part 2) trong khi Adam ở lr=0,01 đã đạt 0,86 — kết luận "Adam thắng tuyệt đối" sẽ chỉ là do lr chưa phù hợp với SGD, không phải Adam vốn mạnh hơn. Đây chính xác là cảnh báo trong GUIDE: so sánh optimizer mà không quét lr riêng cho từng bộ là so sánh không công bằng.
2. **Dropout có giúp khi mô hình chưa quá khớp?** Không — mục 3.4 cho thấy val_macro_f1 giảm đơn điệu và vượt xa nhiễu khi tăng q (0,8474→0,5711), vì baseline chưa quá khớp (gap train/val loss nhỏ, mục 2) nên dropout chỉ làm mất năng lực hiệu dụng mà không có "overfitting" nào để chống. Dropout nên dùng khi quan sát thấy val loss tăng trở lại khi train loss còn giảm (dấu hiệu quá khớp rõ).
3. **Gradient clipping giải quyết vấn đề gì?** Giải quyết các **gai gradient hiếm, đột ngột lớn** (quan sát: ở lr cao, grad_norm từng bước có spike tới 4,6 giữa các bước bình thường ~0,3-0,5). Mục 3.5 cho thấy clip cứu được một phần hiệu năng khi lr cao (0,0936→0,3044) nhưng không giải quyết được việc lr bản thân đã quá cao một cách hệ thống — clipping không phải "thuốc chữa bách bệnh" cho lr sai.
4. **Mixed precision có nhanh hơn trên mạng/dữ liệu này không? Vì sao (không)?** Không — mục 3.6: FP16/BF16 chậm hơn FP32 36-43% ở M-base, 14% ở M-wide. Mạng (tối đa 512 nơ-ron/lớp) và batch (512-2048) quá nhỏ để phần tính toán ma trận trên Tensor Core bù được chi phí cố định của `autocast`/`GradScaler` (chuyển đổi dtype, unscale, kiểm tra overflow mỗi bước). Xu hướng chậm-hơn giảm dần khi mạng rộng ra, gợi ý lợi ích chỉ xuất hiện ở mạng đủ lớn hơn nữa.
5. **Vì sao khởi tạo toàn số 0 hỏng? He khác Xavier ở đâu, khi nào quan trọng?** `zeros` hỏng vì đối xứng hoàn toàn: mọi nơ-ron trong một lớp có cùng `W=0,b=0` nên nhận đúng cùng gradient ở mọi bước, không bao giờ phân hoá chức năng — xác nhận bằng thực nghiệm (val_f1=0,0936, y hệt mốc đoán-đa-số). He (`Var=2/n_in`, gain=√2) và Xavier (`Var=2/(n_in+n_out)`, gain=1) khác nhau ở hệ số bù cho ReLU: He nhân đôi phương sai để bù việc ReLU "giết" nửa tín hiệu (các giá trị âm), nên giữ phương sai kích hoạt ổn định hơn qua nhiều lớp ReLU liên tiếp (mục 3.7: std He ổn định 0,66→0,41 so với Xavier 0,28→0,19, cả hai đều không suy biến mạnh với mạng 2 lớp này). Khác biệt này **quan trọng hơn rõ rệt ở mạng sâu** (nhiều lớp ReLU liên tiếp, như ví dụ "30 lớp" của slide) vì sai số phương sai tích luỹ theo cấp số nhân qua từng lớp; với mạng 2 lớp ẩn của bài này, cả hai vẫn train tốt và khác biệt val_macro_f1 nằm trong nhiễu.
6. **Quay lại câu hỏi của bài học — loss không giảm sau 2000 bước, 3 phép kiểm tra đầu tiên:**
   1. **Loss bước 0 có ≈ ln C không?** Nếu cao hơn nhiều → lớp cuối có điểm số quá lớn (khởi tạo/chuẩn hoá sai) — chính là tình huống mục 2 của báo cáo này (loss bước 0 = 2,15 thay vì 1,95, đã chẩn đoán được nguyên nhân cụ thể là gain He áp cho lớp ra).
   2. **Có quá khớp được một lô nhỏ (2-20 mẫu) không, với mọi regularizer tắt?** Nếu không về gần 0 → gần như chắc chắn lỗi code (nhãn lệch, softmax hai lần, quên `zero_grad`, tham số không vào optimizer), không phải do năng lực mô hình hay lr. Mục 2 của báo cáo đã làm đúng bước này (loss→0,00027, acc=100% trên 20 mẫu) để loại trừ lỗi code trước khi tin vào bất kỳ số liệu huấn luyện dài nào.
   3. **In `grad_norm` của từng tham số — có tham số nào gradient `None`/luôn 0 không?** Nếu có → gradient không chảy tới đó (layer bị "ngắt mạch" khỏi đồ thị tính toán, ví dụ quên gọi `backward()` đúng chỗ, hoặc tham số đó không nằm trong `optimizer.param_groups`). Mục 2 đã in đủ 6 tham số, tất cả khác 0.
   Ba bước này tách được 3 nhóm nguyên nhân khác nhau (dữ liệu/chuẩn hoá, lỗi code, kiến trúc/vòng lặp huấn luyện) **trước khi** nghi ngờ đến lr hay số epoch — đúng tinh thần câu hỏi mở đầu GUIDE.

## 6. Hạn chế và điều bất ngờ

- **Phát hiện và sửa lỗi trong khung code trước khi chạy thí nghiệm chính thức:**
  1. `model.py`: hàm `init_weights()` được định nghĩa nhưng **không bao giờ được gọi** trong `MLP.__init__` — mọi model (kể cả baseline) thực ra dùng khởi tạo mặc định của `nn.Linear`, dù bảng cấu hình ghi "He". Phát hiện được vì 4 thí nghiệm `init-zeros/normal/xavier/he` cho kết quả **giống hệt nhau** — dấu hiệu rõ ràng là tham số `init` không có tác dụng. Đã sửa bằng cách gọi `init_weights(self, init)` ở cuối `__init__`.
  2. Sau khi sửa (1), áp `kaiming_normal_(nonlinearity="relu")` (gain=√2) cho **cả lớp ra** (không có ReLU theo sau) làm loss bước 0 lệch xa ln7 (≈2,27) — đúng tình huống bảng triệu chứng của GUIDE. Sửa: lớp ra dùng gain tuyến tính (`nonlinearity="linear"`), kéo về ≈2,15 (còn lệch nhẹ, đã giải thích cơ chế ở mục 2).
  3. Trong `lab.ipynb`, hàm `make_cfg()` (dựng cfg cho mọi thí nghiệm) có `lr=0.2` **hard-code trong dict mặc định** thay vì tham chiếu biến `BASE_LR` — khi đổi `BASE_LR` sang 0,3 sau khi quét lr, gần 20 thí nghiệm (hparam, dropout, clipping, amp, init) **âm thầm vẫn chạy ở lr=0,2 cũ**, phá vỡ nguyên tắc "mỗi thí nghiệm chỉ đổi một yếu tố so với baseline" (baseline đã dùng lr=0,3). Phát hiện khi so sánh log các lần chạy. Đã sửa bằng cách khai báo `BASE_LR` trước khi định nghĩa `make_cfg()` và để hàm tham chiếu biến global tại thời điểm gọi (không phải lúc định nghĩa).
  4. Một bug phụ: khi `loss-mse` phân kỳ (NaN), logic chọn "cấu hình tốt nhất theo val_macro_f1" dùng `max()` trực tiếp trên các giá trị có cả NaN — Python so sánh với NaN luôn `False`, khiến NaN "sống sót" qua `max()` trong một số thứ tự duyệt và bị chọn nhầm làm cấu hình cuối cùng (gây lỗi khi dự đoán trên eval vì không có `best_state`). Đã sửa bằng cách lọc bỏ mọi kết quả có `diverged=True` hoặc `val_macro_f1` không hữu hạn trước khi chọn.
  - **Bài học:** cả 4 lỗi trên đều là lỗi *thầm lặng* (không crash ngay, âm thầm cho số liệu sai hoặc trùng lặp) — chỉ phát hiện được bằng cách đối chiếu nhiều kết quả với nhau (4 init giống hệt nhau là dấu hiệu bất thường) và kiểm tra lại từng bước, đúng tinh thần "đừng tin số liệu, hãy kiểm chứng" của lab này.
- **Kết quả khác dự đoán (đã phân tích cơ chế ở mục 3):** MSE phân kỳ ở lr baseline (3.1); batch2048+lr-scale-tuyến-tính collapse (3.3); `weight_decay=1e-4` gây tổn hại lớn hơn dự kiến (3.3); dropout tổn hại mạnh hơn dự kiến (3.4); AMP không hề nhanh hơn ngay cả ở M-wide (3.6).
- **Hạn chế của thiết kế thí nghiệm:**
  - Phần lớn thí nghiệm Part 3 (trừ baseline và cấu hình cuối cùng) chỉ chạy **1 seed** — không đo được độ nhiễu riêng, nên các so sánh trong mục 3 dùng chung ngưỡng 2σ của baseline (0,0154) như một xấp xỉ, có thể không phản ánh đúng độ nhiễu thật của từng cấu hình (ví dụ cấu hình dễ dao động như lr cao có thể có σ riêng lớn hơn).
  - Quét lr cho Adam/AdamW/SGD chỉ thử 2 giá trị mỗi bộ (không phải lưới dày) — "lr tốt nhất" tìm được có thể chưa phải tối ưu tuyệt đối.
  - `hparam-batch128` dùng cùng lr=0,3 như batch512 — theo đúng tinh thần "chỉ đổi một yếu tố", nhưng nghĩa là kết quả batch128 chưa được tối ưu bằng lr riêng của nó.
- **Nếu có thêm thời gian:** quét lr riêng cho MSE để so công bằng với CE; thử warm-up lr cho thí nghiệm batch2048-lrscaled để kiểm chứng giả thuyết ổn định; thêm `class_weight` cho các lớp hiếm (mục 4.1); chạy 2-3 seed cho mọi thí nghiệm Part 3, không chỉ baseline/final.

## 7. Phụ lục

- **File đã nộp:** `REPORT.md`, `experiments.xlsx`, `predictions_eval.csv`, `eval_result.json`, `baseline_predictions_eval.csv` + `baseline_eval_result.json` (phụ, chỉ để có số liệu bảng, không phải sản phẩm chấm), `figures/` (32 ảnh theo `exp_id` + 12 ảnh `compare_*`), `results/` (32 file JSON lịch sử huấn luyện), `code/lab.ipynb` + `code/*.py`.
- **Tổng số thí nghiệm chính thức:** 32 dòng trong `experiments.xlsx` (3 baseline seed + 26 thí nghiệm Part 3 qua 7 chủ đề + 3 seed cấu hình cuối cùng); ngoài ra còn các lần quét lr nhanh (`_scan_*`, không đưa vào bảng, chỉ dùng để chọn `BASE_LR`).
- **Thời gian chạy ước tính:** toàn bộ notebook (từ nạp dữ liệu tới ghi `experiments.xlsx`) chạy khoảng **25 phút** trên GPU RTX 4060 Laptop.
