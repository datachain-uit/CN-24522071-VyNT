# 03 — Nhật ký thực nghiệm

Ghi MỌI lần chạy (kể cả lần lỗi). Một dòng cho mỗi lần chạy; chi tiết để trong thư mục kết quả.

| Ngày | Commit | Lệnh / cấu hình | Seed | Thời gian chạy | Kết quả chính | Khác biệt so với kế hoạch / ghi chú |
|---|---|---|---|---|---|---|
| 2026-__-__ | | `python scripts/00_setup_data.py` | – | | 154 template OK | |
| | | `python -m pytest -q tests` | – | | | |
| | | `python scripts/01_eda_tpch.py` | – | | | |
| | | `python scripts/02_run_baselines.py --env configs/env/tpch_small.yaml --split val` | val | | | |
| | | `python scripts/03_tune_weighted_fair.py --env configs/env/tpch_small.yaml` | val | | α* = | |
| | | `python scripts/04_train_decima.py --config configs/train/decima_small.yaml --seed 0` | train | | | |

## Sai khác so với bài báo Decima cần ghi lại khi viết bài
- Simulator: spark-sched-sim (PyTorch Geometric, PPO, discounted return, không curriculum) thay vì bản TensorFlow gốc
  (REINFORCE, average reward, curriculum + memoryless termination).
- Dữ liệu: 7 kích thước TPC-H (có 80g) theo mã nguồn, trong khi bài báo mô tả 6 kích thước.
- Quy mô: 10 executor / 50 job (MVP) thay vì cụm 25 VM × 2 executor và 1000 job.
- Baseline cài lại theo mô tả bài báo (Weighted-Fair dùng tổng work từ profile; SJF-CP); Graphene*/Tetris chưa có.
