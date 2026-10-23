# 01 — Phát biểu bài toán và mô hình MDP (phạm vi đã chốt: lập lịch DAG job theo hướng Decima)

## 1. Bài toán dưới dạng input → output → metric

| Thành phần | Nội dung |
|---|---|
| Input (tại mỗi thời điểm ra quyết định) | Tập job đang có trong hệ thống, mỗi job là một DAG các stage (số task còn lại, thời lượng task ước lượng, quan hệ cha–con); trạng thái executor (bao nhiêu executor cần gán, job nào đang nhả executor, mỗi job đang giữ bao nhiêu executor) |
| Output | Một hành động: chọn **stage** kế tiếp được chạy và **số executor** cấp cho stage đó (giới hạn song song) |
| Metric chính | **Average JCT** (job completion time trung bình, giây) trên toàn bộ job của một chuỗi test |
| Metric phụ | p95/p99 JCT, makespan, executor utilization, số job trung bình trong hệ thống, độ trễ ra quyết định (ms), thời gian huấn luyện |

Giả định của môi trường (spark-sched-sim, giống Decima §6.2): N executor **đồng nhất** (chỉ một loại tài nguyên là
executor slot — KHÔNG mô phỏng CPU/RAM riêng); job đến theo quá trình Poisson; dữ liệu job là 154 profile TPC-H
(22 query × 7 kích thước); thời lượng task được lấy mẫu từ profile (có hiệu ứng "wave", warm-up 1 s, chuyển executor
giữa các job mất 2 s); không preemption; không lỗi máy.

## 2. MDP

| Thành phần | Định nghĩa trong môi trường | Vị trí trong mã |
|---|---|---|
| Điểm ra quyết định | Khi có job đến, stage hết task cần gán, task xong và executor được nhả. Tại một sự kiện, agent được gọi **lặp lại** cho tới khi gán hết executor hoặc hết stage schedulable | `spark_sched_sim.py::step`, `_resume_simulation` |
| State s_k (thô) | Mỗi stage đang hoạt động: (số task còn lại, thời lượng task gần nhất, có schedulable); cạnh DAG; `dag_ptr`; `num_committable_execs`; `source_job_idx`; `exec_supplies` | `_observe` |
| State của Decima | 5 đặc trưng/stage: trần executor có thể cấp/N, thuộc job nguồn (±1), số executor của job/N, task còn lại/200, work còn lại/1e5 + DAG cho GNN | `schedulers/decima/env_wrapper.py` |
| Action a_k | (stage_idx ∈ tập schedulable — đóng vai trò action mask, num_exec ∈ {1..num_committable}) | `_take_action` |
| Chuyển trạng thái | Mô phỏng sự kiện rời rạc (discrete-event) tới sự kiện cần quyết định kế tiếp | `_resume_simulation` |
| Reward r_k | −∑_jobs (thời gian job nằm trong hệ thống trong [t_{k−1}, t_k]) (ms). Tổng reward của episode = −∑ JCT ⇒ tối đa hóa reward ⇔ tối thiểu avg JCT (định luật Little). Trainer dùng chiết khấu liên tục β (`beta_discount`) hoặc reward vi phân (average reward) | `_compute_jobtime`, `trainers/utils/returns_calculator.py` |
| Episode | `reset(seed)` sinh chuỗi `job_arrival_cap` job. Huấn luyện: thời điểm dừng ngẫu nhiên ~ Exp(`mean_time_limit`) (memoryless termination như Decima). Đánh giá: chạy tới khi mọi job xong | `wrappers/stochastic_time_limit.py`, `src/rlsched/evaluation/runner.py` |
| Policy | π_θ(a|s) = π(stage | s) · π(số executor | s, job của stage); GNN truyền thông điệp theo DAG + 2 mạng tính điểm + softmax | `schedulers/decima/scheduler.py` |
| Giảm phương sai | Baseline tính riêng cho từng chuỗi job (input-driven baseline, Mao et al. ICLR 2019): nhiều rollout trên cùng một chuỗi job | `trainers/utils/baselines.py` |

## 3. Đối chiếu với danh sách biến/hàm trong báo cáo "Yêu cầu về Môi trường Thực nghiệm"

| Biến trong báo cáo | Trong phạm vi đã chốt | Ghi chú |
|---|---|---|
| arrival_time | `job.t_arrival` | sinh từ Poisson(λ = `job_arrival_rate`, đơn vị 1/ms) |
| duration | thời lượng **từng task** lấy mẫu từ profile; ước lượng work của stage = số task × thời lượng TB | không có "duration của job" cố định — JCT phụ thuộc lịch |
| cpu_required, memory_required | **không có** trong môi trường cơ bản (một loại tài nguyên: executor) | đa tài nguyên (Decima §7.3) là hướng mở rộng, cần sửa simulator |
| cpu_capacity, cpu_available | N = `num_executors`; `num_committable_execs`, `exec_supplies` | |
| DAG (biến tùy chọn) | **bắt buộc** trong phạm vi này | cạnh `edge_links`, cấu trúc `job.dag` |
| state / action / reward | xem bảng mục 2 | |
| Biến đánh giá | JCT (chính), makespan, utilization, p95 JCT… | `src/rlsched/evaluation/metrics.py` |

Các hàm cần có và vị trí: `env.reset(seed)`, `env.step(action)`, `scheduler.schedule(obs)`, `run_episode(...)`
(đo đạc), `episode_summary(...)` (metric), `paired_compare(...)` (kiểm định), `make_trainer(cfg).train()` (huấn luyện).
