import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from run_cpt_leg import save_optimizer_state, restore_optimizer_state


def test_optimizer_state_roundtrip(tmp_path):
    params = {"w": mx.zeros((4,)), "b": mx.zeros((2,))}
    grads = {"w": mx.ones((4,)), "b": mx.ones((2,))}

    opt_a = optim.Adam(learning_rate=1e-3)
    for _ in range(5):
        opt_a.apply_gradients(grads, params)
    step_before = int(opt_a.step.item())
    assert step_before == 5

    path = tmp_path / "opt_state.safetensors"
    save_optimizer_state(opt_a, path)

    opt_b = optim.Adam(learning_rate=1e-3)
    opt_b.init(params)
    restore_optimizer_state(opt_b, path)

    assert int(opt_b.step.item()) == step_before
    # continuing training on opt_b should pick up from step 5, not 0
    opt_b.apply_gradients(grads, params)
    assert int(opt_b.step.item()) == step_before + 1


def test_restore_preserves_adam_moments(tmp_path):
    params = {"w": mx.zeros((4,))}
    grads = {"w": mx.array([1.0, 2.0, 3.0, 4.0])}
    opt_a = optim.Adam(learning_rate=1e-3)
    opt_a.apply_gradients(grads, params)
    m_before = opt_a.state["w"]["m"]

    path = tmp_path / "opt_state.safetensors"
    save_optimizer_state(opt_a, path)

    opt_b = optim.Adam(learning_rate=1e-3)
    opt_b.init(params)
    restore_optimizer_state(opt_b, path)
    assert mx.allclose(opt_b.state["w"]["m"], m_before)
