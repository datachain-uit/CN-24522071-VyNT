# rl-cluster-scheduling — Khung thực nghiệm RL cho lập lịch DAG job trên cluster

Repo khởi đầu cho đề tài **Ứng dụng Reinforcement Learning để tối ưu lập lịch cho hệ tính toán phân bố**.
Phạm vi giai đoạn 1: **lập lịch trực tuyến các job DAG (kiểu Spark) với job đến liên tục, tối thiểu average JCT**,
lấy Decima (Mao et al., SIGCOMM 2019) làm bài báo gốc.

- Lõi mô phỏng: [spark-sched-sim](https://github.com/ArchieGertsman/spark-sched-sim) (MIT) — bản PyTorch Geometric +
  Gymnasium của Decima, vendored tại `third_party/spark-sched-sim` (đã sửa lỗi trainer, xem `THIRD_PARTY_NOTICES.md`).
- Repo này bổ sung: baseline heuristic (Random, FIFO, Fair, Weighted-Fair, SJF-CP, SRPT-CP), chia seed
  train/val/test, đo đạc ở mức job/episode, thống kê ghép cặp (Wilcoxon + Holm, bootstrap CI), script chạy từng bước,
  kiểm thử tự động.

## 1. Cấu trúc
```
configs/            env/*.yaml (môi trường), train/*.yaml (huấn luyện), splits.yaml (dải seed)
src/rlsched/        envs.py, data.py, schedulers/ (heuristic + Decima), evaluation/ (runner, metrics, stats, plots)
scripts/            00_setup_data → 01_eda → 02_baselines → 03_tune → 04_train → 05_select → 06_evaluate → 07_analyze
tests/              pytest: tách tập, tất định theo seed, bất biến reward = −∑JCT, khớp Fair upstream, shim PyG
docs/               01 bài toán & MDP · 02 giao thức thực nghiệm · 03 nhật ký thực nghiệm
third_party/        spark-sched-sim (vendored, MIT) · pyg_ext_fallback (thay thế torch_scatter/torch_sparse khi cần)
data/, results/     tạo ra khi chạy (không commit)
```

## 2. Cài đặt (Python 3.10 hoặc 3.11)
```bash
python3.11 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                               # đủ cho heuristic + đánh giá
pip install -r requirements-rl.txt                            # thêm torch/PyG để huấn luyện Decima (CPU)
python scripts/00_setup_data.py                               # tải 154 job profile TPC-H vào data/tpch
python -m pytest -q tests                                     # 35 PASS (chưa cài torch-scatter/sparse: 34 PASS + 1 SKIP)
```
Không cài được `torch-scatter`/`torch-sparse`? Bỏ hai dòng đó trong `requirements-rl.txt` và thêm `--pyg-fallback`
vào các lệnh 04–06 (hoặc `export RLSCHED_PYG_FALLBACK=1`). Bản thay thế chậm hơn nhưng cho cùng kết quả toán học.
Mọi script phải chạy **từ thư mục gốc repo** (simulator đọc dữ liệu theo đường dẫn tương đối `data/tpch`).

## 3. Quy trình chạy (MVP trên `configs/env/tpch_small.yaml`: 10 executor, 50 job, tải ≈ 0.85)
```bash
python scripts/01_eda_tpch.py                                                   # thống kê dữ liệu, hiệu chỉnh tải
python scripts/02_run_baselines.py --env configs/env/tpch_small.yaml --split val
python scripts/03_tune_weighted_fair.py --env configs/env/tpch_small.yaml       # tune α trên validation
python scripts/04_train_decima.py --config configs/train/decima_smoke.yaml --seed 0   # chạy thử 1–3 phút
for s in 0 1 2; do python scripts/04_train_decima.py --config configs/train/decima_small.yaml --seed $s; done
for s in 0 1 2; do python scripts/05_select_checkpoint.py --run results/train/decima_small/seed$s --env configs/env/tpch_small.yaml; done
python scripts/06_evaluate.py --env configs/env/tpch_small.yaml \
    --methods random fifo fair sjf_cp srpt_cp "weighted_fair:alpha=@results/tuning/tpch_small/best.json" \
    --decima-runs results/train/decima_small/seed0 results/train/decima_small/seed1 results/train/decima_small/seed2
python scripts/07_analyze.py --in results/final/tpch_small_test --reference Decima
```
Theo dõi huấn luyện: `tensorboard --logdir results/train`, nhật ký `results/train/<cfg>/seed<k>/stdout/main.out`.

## 4. Quy tắc bắt buộc
1. Tập test (seed 2 000 000+) chỉ dùng **một lần** ở bước 06; mọi lựa chọn (α, checkpoint, siêu tham số) dùng validation.
2. Mọi phương pháp chạy trên **cùng** chuỗi job test và cùng cấu hình môi trường; ghi rõ thông tin mỗi phương pháp được dùng.
3. Phương pháp học: **≥ 3 seed huấn luyện**; báo cáo mean ± std, CI 95%, Wilcoxon ghép cặp + Holm.
4. Ghi mọi lần chạy vào `docs/03_nhat_ky_thuc_nghiem.md`; không xóa kết quả xấu.
5. Không sửa `third_party/` khi chưa ghi lý do + diff vào `THIRD_PARTY_NOTICES.md`.

## 5. Ghi chú tái lập
- Hành động của Decima được lấy mẫu bằng module `random` toàn cục; `run_episode` đặt `random.seed(seed)` trước mỗi
  episode nên kết quả tất định. (`examples.py` của upstream không làm vậy: cùng seed môi trường 1234, 4 lần chạy cho
  avg JCT 577–635 s.)
- Bản thay thế `third_party/pyg_ext_fallback` cho kết quả trùng khớp với torch-scatter 2.1.1/torch-sparse 0.6.17
  (đến 4 chữ số thập phân trên 4 episode Decima).
- Tài liệu tham khảo đã kiểm chứng (BibTeX): `docs/references.bib`.

## 6. Trích dẫn
Mao, H., Schwarzkopf, M., Venkatakrishnan, S. B., Meng, Z., & Alizadeh, M. (2019). Learning Scheduling Algorithms for
Data Processing Clusters. *ACM SIGCOMM 2019*, 270–288. https://doi.org/10.1145/3341302.3342080
Simulator: Gertsman, A. spark-sched-sim, https://github.com/ArchieGertsman/spark-sched-sim (MIT License).
