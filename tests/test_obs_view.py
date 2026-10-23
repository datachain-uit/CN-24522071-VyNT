import numpy as np

from rlsched.data import critical_path
from rlsched.schedulers.obs_view import ObsView


def test_critical_path_simple_dag():
    # 0 -> 1 -> 3 ; 0 -> 2 -> 3 ; trọng số 1, 5, 2, 1  => cp(0) = 1 + 5 + 1 = 7
    children = [[1, 2], [3], [3], []]
    cp = critical_path(children, np.array([1.0, 5.0, 2.0, 1.0]))
    assert cp.tolist() == [7.0, 6.0, 3.0, 1.0]


def _view():
    # 2 job: job0 có stage 0 -> 1 ; job1 có stage 2. Stage 0 và 2 schedulable.
    nodes = np.array([[4, 1000.0, 1], [2, 500.0, 0], [10, 2000.0, 1]])
    return ObsView(nodes=nodes, edge_links=np.array([[0, 1]]), dag_ptr=np.array([0, 2, 3]), num_committable=5,
                   source_job_idx=2, exec_supplies=np.array([0, 0]))


def test_view_basic_fields():
    v = _view()
    assert v.num_jobs == 2 and not v.has_source_job
    assert v.action_of_node == {0: 0, 2: 1}
    assert v.frontier.tolist() == [True, False, True]
    assert np.allclose(v.job_work, [4 * 1000 + 2 * 500, 10 * 2000])
    assert v.pick_stage(0, "frontier") == 0 and v.pick_stage(1, "critical_path") == 1
