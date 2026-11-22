# 03 — Nhật ký thực nghiệm

Ghi MỌI lần chạy (kể cả lần lỗi). Một dòng cho mỗi lần chạy; chi tiết để trong thư mục kết quả.

| Ngày | Commit | Lệnh / cấu hình | Seed | Thời gian chạy | Kết quả chính | Khác biệt so với kế hoạch / ghi chú |
|---|---|---|---|---|---|---|
| 2026-11-23 | c4d54485 | K1: `python scripts/00_setup_data.py` | – | Không ghi nhận | Lỗi `UnicodeEncodeError`: cp1252 không mã hóa được tiếng Việt khi in thông báo | Ghi nhận từ ảnh terminal; log lỗi riêng không còn được lưu. Bật `PYTHONUTF8=1` và UTF-8 cho PowerShell rồi chạy lại. |
| 2026-11-23 | c4d54485 | K1: `python scripts/00_setup_data.py` | – | Không ghi nhận | `already-present`; số file thiếu: 0; đủ 154 template | Thành công sau khi bật UTF-8. Log: [data_setup.log](results/2026-11-23/data_setup.log). |
| 2026-11-23 | c4d54485 | K1: `python -m pytest -q tests` | – | 69.35 s | **35 passed** | Log: [pytest.log](results/2026-11-23/pytest.log). Đạt tiêu chí kiểm thử K1; tiếp theo thực hiện K2 — EDA. |
| | | `python scripts/01_eda_tpch.py` | – | | | |
| | | `python scripts/02_run_baselines.py --env configs/env/tpch_small.yaml --split val` | val | | | |
| | | `python scripts/03_tune_weighted_fair.py --env configs/env/tpch_small.yaml` | val | | α* = | |
| | | `python scripts/04_train_decima.py --config configs/train/decima_small.yaml --seed 0` | train | | | |

### Môi trường và bằng chứng K1 — 23/11/2026

- Hệ điều hành / shell: Windows / PowerShell, theo ảnh terminal.
- Python: 3.11, theo đường dẫn trong traceback; chưa xác nhận phiên bản patch.
- Thư mục chạy: `D:\Paper\CN-24522071-VyNT`.
- Commit mã nguồn được đối chiếu: `c4d54485eb86eabe70a3e778e85692434bc0065a` (HEAD hiện tại; không có thay đổi mã nguồn khi ghi nhật ký).
- Lệnh thực tế dùng `& $py`; log được lưu bằng `2>&1` và `Tee-Object`. Chưa xác nhận đường dẫn đầy đủ của `$py` trong phiên PowerShell đã chạy.
- Khắc phục lỗi xuất tiếng Việt: đặt `$env:PYTHONUTF8 = "1"`, `[Console]::OutputEncoding` và `$OutputEncoding` dùng UTF-8.
- Log gốc: `results/setup/data_setup.log`, `results/setup/pytest.log`; bản sao bằng chứng được lưu trong `docs/results/2026-11-23/` để commit.
- Chưa chạy lại thí nghiệm khi ghi nhật ký; kết quả trên lấy từ log đã có. Các dòng trống còn lại là mẫu cho những bước tiếp theo, không phải kết quả đã chạy.

## Sai khác so với bài báo Decima cần ghi lại khi viết bài
- Simulator: spark-sched-sim (PyTorch Geometric, PPO, discounted return, không curriculum) thay vì bản TensorFlow gốc
  (REINFORCE, average reward, curriculum + memoryless termination).
- Dữ liệu: 7 kích thước TPC-H (có 80g) theo mã nguồn, trong khi bài báo mô tả 6 kích thước.
- Quy mô: 10 executor / 50 job (MVP) thay vì cụm 25 VM × 2 executor và 1000 job.
- Baseline cài lại theo mô tả bài báo (Weighted-Fair dùng tổng work từ profile; SJF-CP); Graphene*/Tetris chưa có.
