# 03 - Experiment Log

Record EVERY run, including failed ones. Append new entries at the END of the Run log, in chronological order. Keep details in the results directory.

Entry format (add `seed` / `split` only when relevant):

- DATE | STEP | COMMIT | `command` | runtime | RESULT
  - Note: deviations from plan, log link, next step.


## Environment

- Windows / PowerShell; Python 3.11 (patch version not confirmed); working directory `D:\Paper\CN-24522071-VyNT`.
- Logging: `python ... 2>&1 | Out-File -Encoding utf8 <log>` with `$env:PYTHONUTF8 = "1"` (Windows PowerShell 5.1 `Tee-Object` writes UTF-16).
- `.gitignore` uses `/results/` (root only) so that `docs/results/` can be committed.


## Deviations from the Decima paper (record when writing the report)

- Simulator: spark-sched-sim (PyTorch Geometric, PPO, discounted return, no curriculum) instead of the original TensorFlow version (REINFORCE, average reward, curriculum + memoryless termination).
- Data: 7 TPC-H sizes (including 80g) according to the source code, while the paper describes 6 sizes.
- Scale: 10 executors / 50 jobs (MVP) instead of 25 VMs x 2 executors and 1000 jobs.
- Baselines reimplemented from the paper description (Weighted-Fair uses total work from the profile; SJF-CP); Graphene*/Tetris not implemented yet.


## Run log

- 2026-11-23 | K1 | c4d5448 | `python scripts/00_setup_data.py` | n/a | FAILED: `UnicodeEncodeError` (cp1252 cannot encode Vietnamese console messages)
  - Note: from a terminal screenshot; error log not kept. Fixed with `PYTHONUTF8=1` and UTF-8 in PowerShell.

- 2026-11-23 | K1 | c4d5448 | `python scripts/00_setup_data.py` | n/a | OK: `already-present`, missing files 0, 154 templates
  - Note: UTF-16 log with Vietnamese messages, not merged into main; superseded by the 2026-11-24 rerun.

- 2026-11-23 | K1 | c4d5448 | `python -m pytest -q tests` | 69.35 s | 35 passed
  - Note: superseded by the 2026-11-24 rerun.

- 2026-11-24 | Code change | 317f9c0 | translate console messages in `scripts/`, `src/rlsched/`, `tests/` to English; add UTF-8 `reconfigure(errors="replace")` in `scripts/_common.py` | - | `UnicodeEncodeError` no longer occurs with redirected output
  - Note: display strings only, no behavior change. All logs are in English from now on; docs and code comments stay Vietnamese.

- 2026-11-24 | K1 rerun | 317f9c0 | `python scripts/00_setup_data.py` | not measured | OK: `already-present`, missing files 0, 154 templates (7 sizes x 22 queries)
  - Log: [data_setup.log](results/2026-11-24/data_setup.log)

- 2026-11-24 | K1 rerun | 317f9c0 | `python -m pytest -q tests` | n/a | FAILED (environment): `No module named pytest`; `python` resolved to `C:\msys64\ucrt64\bin\python.exe`, not the project environment
  - Note: no test was executed. Rerun with `.\.venv\Scripts\python.exe`. The error log was overwritten by the next run.

- 2026-11-24 | K1 rerun | 317f9c0 | `.\.venv\Scripts\python.exe -m pytest -q tests` | 61.14 s | 35 passed
  - Log: [pytest.log](results/2026-11-24/pytest.log). Meets the K1 test criterion; next step is K2 (EDA).