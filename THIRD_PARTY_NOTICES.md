# Mã nguồn và dữ liệu bên thứ ba

## spark-sched-sim (third_party/spark-sched-sim)
- Nguồn: https://github.com/ArchieGertsman/spark-sched-sim — commit `2ba8d80` (09/01/2024).
- Giấy phép: MIT (xem `third_party/spark-sched-sim/LICENSE`). Giữ nguyên thông báo bản quyền khi phân phối lại.
- Đã sửa (xem `third_party/spark-sched-sim/PATCHES.diff`), chỉ trong `trainers/trainer.py`:
  1. Vòng lặp gom kết quả dùng lại biến `i` của vòng lặp huấn luyện -> số vòng lặp in ra sai, checkpoint
     không bao giờ được lưu (khi số worker không chia hết cho checkpointing_freq) hoặc gây FileExistsError
     rồi treo tiến trình. Đổi tên biến thành `rank`.
  2. Bọc vòng huấn luyện bằng try/finally để luôn dừng các tiến trình rollout; dừng worker an toàn.
  3. Lưu checkpoint cuối cùng; dùng `os.makedirs(..., exist_ok=True)`.
- Lỗi cấu hình đã biết nhưng KHÔNG sửa trong upstream (xử lý ở `src/rlsched/envs.py` và `configs/train/`):
  `config/decima_tpch.yaml` dùng khóa `dataset` thay cho `data_sampler_cls` (KeyError) và `device: 'cuda'`.

## Decima (Mao et al., SIGCOMM 2019) và dữ liệu TPC-H DAG
- Bài báo: https://doi.org/10.1145/3341302.3342080 ; mã gốc: https://github.com/hongzimao/decima-sim
- Dữ liệu job profile TPC-H (thư mục `spark_env/tpch` của decima-sim) KHÔNG được đóng gói trong repo này;
  `scripts/00_setup_data.py` tải trực tiếp từ nguồn gốc. Repo decima-sim không kèm tệp giấy phép, vì vậy
  không phân phối lại dữ liệu và luôn trích dẫn nguồn khi sử dụng.

## Bản thay thế torch_scatter / torch_sparse (third_party/pyg_ext_fallback)
- Viết mới cho repo này, chỉ cài đặt các hàm Decima cần; có kiểm thử đối chiếu trong `tests/test_pyg_fallback.py`.
