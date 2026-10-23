# 02 — Giao thức thực nghiệm (khai báo TRƯỚC khi chạy)

## Câu hỏi nghiên cứu giai đoạn tái lập (MVP)
- **RQ1 (tái lập):** Trên môi trường `tpch_small`, Decima huấn luyện lại có cho avg JCT thấp hơn baseline mạnh nhất
  (đã tune trên validation) với khác biệt có ý nghĩa thống kê không?
- **RQ2 (độ ổn định):** Kết quả của Decima dao động thế nào giữa ≥ 3 seed huấn luyện?
- **RQ3 (chi phí):** Thời gian huấn luyện và độ trễ ra quyết định là bao nhiêu so với heuristic?

## Tập dữ liệu và chia tập (configs/splits.yaml)
- Một **seed** ⇔ một chuỗi job (thời điểm đến, query, kích thước). Train: seed của trainer (dải [0, 10⁶));
  validation: 20 seed bắt đầu 1 000 000; test: 50 seed bắt đầu 2 000 000.
- Validation dùng cho: tune α của Weighted-Fair, chọn checkpoint, chọn siêu tham số. Test chỉ chạy **một lần**
  (`scripts/06_evaluate.py` từ chối ghi đè nếu không có `--force`).
- Giới hạn cần nêu khi viết bài: test là chuỗi job **mới** nhưng dùng **cùng** 154 job template với train.
  Đánh giá tổng quát hóa thật sự (khác tải, khác số executor, giữ lại một số query) là thí nghiệm riêng ở giai đoạn sau.

## Baseline và ngân sách tune
| Tầng | Phương pháp | Tune |
|---|---|---|
| 1 | Random, FIFO, Fair | không có tham số |
| 1–2 | SJF-CP (ưu tiên tổng work, chọn stage trên đường găng), SRPT-CP (ưu tiên work còn lại) | không có tham số |
| 2 | Weighted-Fair(α) | quét α ∈ [−2, 2] bước 0.25 (MVP) / 0.1 (bản bài báo) trên 20 seed validation |
| 3 | Decima (spark-sched-sim, PPO) | giữ siêu tham số upstream; nếu tune thêm thì ghi rõ số cấu hình đã thử |
| 3 | SOTA khác (chọn theo tài liệu trong báo cáo review) | giai đoạn sau |
| 5 | Ablation phương pháp đề xuất | giai đoạn sau |
Nguyên tắc: cùng môi trường, cùng chuỗi job test, cùng loại thông tin; ghi rõ số liệu nào tự chạy, số liệu nào lấy từ bài báo.

## Metric và thống kê
- Chính: avg JCT/episode. Phụ: p95 JCT, makespan, utilization, decision latency, thời gian huấn luyện.
- Báo cáo: mean ± std và CI 95% (bootstrap) qua 50 episode test; với Decima lấy trung bình theo seed huấn luyện cho
  từng episode rồi mới kiểm định; báo cáo riêng mean ± std giữa các seed huấn luyện.
- Kiểm định: Wilcoxon signed-rank ghép cặp theo seed test (chính), paired t-test (tham khảo); hiệu chỉnh Holm khi so
  nhiều baseline; kèm cải thiện tương đối (%) và CI.
- Vẽ: CDF của JCT, cột mean với CI, learning curve của từng seed huấn luyện.

## Tiêu chí thành công (khai báo trước)
- Tái lập **đạt** nếu Decima tốt hơn Fair có ý nghĩa (p_Holm < 0.05) trên `tpch_small` và cùng xu hướng định tính với
  bài báo (Decima > Fair). So với SJF-CP/SRPT-CP: ghi nhận đúng thực tế, kể cả khi Decima **không** thắng — đó là phát
  hiện quan trọng để định hướng phương pháp đề xuất.
- Phương pháp đề xuất (giai đoạn sau) chỉ được coi là cải tiến khi thắng baseline mạnh nhất có ý nghĩa thống kê
  **và** mức cải thiện lớn hơn độ dao động giữa các seed huấn luyện.

## Ngân sách tính toán (ước lượng)
- Heuristic: ~2–3 s/episode `tpch_small` trên 1 nhân CPU ⇒ 6 phương pháp × 70 seed ≈ 15–20 phút.
- Decima `decima_small` (500 vòng, 8 rollout/vòng): đo trên máy 2 vCPU được ~12 s/vòng với 2 rollout; ước lượng
  ~20–30 s/vòng với 8 rollout trên máy 8 nhân ⇒ **~3–4 giờ/seed**, 3 seed ≈ 10–12 giờ (chạy qua đêm). Giảm
  `num_rollouts` nếu máy ít nhân.

## Quy ước đặt tên và lưu vết
- `results/train/<config>/seed<k>/`, `results/baselines/<env>_<split>/`, `results/final/<env>_test/`.
- Mỗi lần chạy lưu `meta.json` (commit git, phiên bản thư viện, cấu hình). Ghi mọi lần chạy vào `docs/03_nhat_ky_thuc_nghiem.md`.
