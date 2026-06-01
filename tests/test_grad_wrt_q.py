"""Tests for RNEA / ABA differentiability w.r.t. configuration q and velocity qd.

These complement ``test_autograd.py`` (which covers CRBA d/dq, ABA d/dtau, and
RNEA d/dqd... d/dqdd). Here we verify the *configuration* gradients that the
in-place scratch-buffer implementations previously did not support:

    - RNEA: d(tau)/d(q),  d(tau)/d(qd)
    - ABA:  d(qdd)/d(q),  d(qdd)/d(qd)

Gradient flows through the cached spatial quantities (Xup, v) produced by the
functional ``update_kinematics`` path. The dynamics algorithms detect this via
``Xup.requires_grad / v.requires_grad`` and switch their per-node force/inertia
loops from in-place buffer writes to functional list accumulation.
"""

from pathlib import Path

import pytest
import torch

import bard


@pytest.fixture(scope="session")
def urdf_path():
    path = Path(__file__).parent / "go2_description/urdf/go2.urdf"
    if not path.exists():
        pytest.skip(f"Required test asset not found: {path}")
    return str(path)


def _model_data(urdf_path, dtype, device, batch_size, floating_base):
    model = bard.build_model_from_urdf(urdf_path, floating_base=floating_base).to(
        dtype=dtype, device=device
    )
    return model, bard.create_data(model, max_batch_size=batch_size)


def _random_state(model, batch_size, dtype, device, seed=0):
    torch.manual_seed(seed)
    nv = model.nv
    if model.has_floating_base:
        t = torch.randn(batch_size, 3, dtype=dtype, device=device)
        quat = torch.randn(batch_size, 4, dtype=dtype, device=device)
        quat = quat / quat.norm(dim=-1, keepdim=True)
        q_joints = 0.2 * torch.randn(batch_size, model.n_joints, dtype=dtype, device=device)
        q = torch.cat([t, quat, q_joints], dim=1)
    else:
        q = 0.2 * torch.randn(batch_size, model.nq, dtype=dtype, device=device)
    qd = 0.1 * torch.randn(batch_size, nv, dtype=dtype, device=device)
    qdd = 0.1 * torch.randn(batch_size, nv, dtype=dtype, device=device)
    return q, qd, qdd


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# ---------------------------------------------------------------------------
# RNEA  d(tau)/d(q),  d(tau)/d(qd)
# ---------------------------------------------------------------------------
class TestRNEAGradWrtConfig:
    @pytest.mark.parametrize("floating_base", [False, True])
    @pytest.mark.parametrize("device", DEVICES)
    def test_rnea_grad_wrt_q_nonzero(self, urdf_path, device, floating_base):
        model, data = _model_data(urdf_path, torch.float64, device, 4, floating_base)
        q, qd, qdd = _random_state(model, 4, torch.float64, device)
        q = q.clone().requires_grad_(True)

        bard.update_kinematics(model, data, q, qd)
        tau = bard.rnea(model, data, qdd)
        tau.pow(2).sum().backward()

        assert q.grad is not None and q.grad.abs().sum() > 0
        assert torch.isfinite(q.grad).all()

    @pytest.mark.parametrize("floating_base", [False, True])
    @pytest.mark.parametrize("device", DEVICES)
    def test_rnea_grad_wrt_qd_nonzero(self, urdf_path, device, floating_base):
        model, data = _model_data(urdf_path, torch.float64, device, 4, floating_base)
        q, qd, qdd = _random_state(model, 4, torch.float64, device)
        qd = qd.clone().requires_grad_(True)

        bard.update_kinematics(model, data, q, qd)
        tau = bard.rnea(model, data, qdd)
        tau.pow(2).sum().backward()

        assert qd.grad is not None and qd.grad.abs().sum() > 0
        assert torch.isfinite(qd.grad).all()

    @pytest.mark.parametrize("floating_base", [False, True])
    def test_rnea_gradcheck_wrt_q(self, urdf_path, floating_base):
        model, data = _model_data(urdf_path, torch.float64, "cpu", 1, floating_base)
        q, qd, qdd = _random_state(model, 1, torch.float64, "cpu", seed=3)
        q = q.clone().requires_grad_(True)

        def fn(q_in):
            bard.update_kinematics(model, data, q_in, qd)
            return bard.rnea(model, data, qdd)

        assert torch.autograd.gradcheck(fn, (q,), eps=1e-6, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize("floating_base", [False, True])
    def test_rnea_gradcheck_wrt_qd(self, urdf_path, floating_base):
        model, data = _model_data(urdf_path, torch.float64, "cpu", 1, floating_base)
        q, qd, qdd = _random_state(model, 1, torch.float64, "cpu", seed=4)
        qd = qd.clone().requires_grad_(True)

        def fn(qd_in):
            bard.update_kinematics(model, data, q, qd_in)
            return bard.rnea(model, data, qdd)

        assert torch.autograd.gradcheck(fn, (qd,), eps=1e-6, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize("device", DEVICES)
    def test_rnea_grad_wrt_qdd_still_works(self, urdf_path, device):
        """Pre-existing d(tau)/d(qdd) path must remain functional."""
        model, data = _model_data(urdf_path, torch.float64, device, 4, True)
        q, qd, qdd = _random_state(model, 4, torch.float64, device)
        qdd = qdd.clone().requires_grad_(True)

        bard.update_kinematics(model, data, q, qd)
        bard.rnea(model, data, qdd).pow(2).sum().backward()
        assert qdd.grad is not None and qdd.grad.abs().sum() > 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_rnea_training_loop_pattern(self, urdf_path, device):
        """Repeated update->rnea->backward on the same grad data (no retain_graph)."""
        model, data = _model_data(urdf_path, torch.float64, device, 4, True)
        q0, qd, qdd = _random_state(model, 4, torch.float64, device)
        q = q0.clone().requires_grad_(True)
        for _ in range(3):
            q.grad = None
            bard.update_kinematics(model, data, q, qd)
            bard.rnea(model, data, qdd).pow(2).sum().backward()
            assert q.grad is not None and q.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# ABA  d(qdd)/d(q),  d(qdd)/d(qd)
# ---------------------------------------------------------------------------
class TestABAGradWrtConfig:
    @pytest.mark.parametrize("floating_base", [False, True])
    @pytest.mark.parametrize("device", DEVICES)
    def test_aba_grad_wrt_q_nonzero(self, urdf_path, device, floating_base):
        model, data = _model_data(urdf_path, torch.float64, device, 4, floating_base)
        q, qd, _ = _random_state(model, 4, torch.float64, device)
        tau = 0.1 * torch.randn(4, model.nv, dtype=torch.float64, device=device)
        q = q.clone().requires_grad_(True)

        bard.update_kinematics(model, data, q, qd)
        bard.aba(model, data, tau).pow(2).sum().backward()

        assert q.grad is not None and q.grad.abs().sum() > 0
        assert torch.isfinite(q.grad).all()

    @pytest.mark.parametrize("floating_base", [False, True])
    def test_aba_gradcheck_wrt_q(self, urdf_path, floating_base):
        model, data = _model_data(urdf_path, torch.float64, "cpu", 1, floating_base)
        q, qd, _ = _random_state(model, 1, torch.float64, "cpu", seed=5)
        tau = 0.1 * torch.randn(1, model.nv, dtype=torch.float64)
        q = q.clone().requires_grad_(True)

        def fn(q_in):
            bard.update_kinematics(model, data, q_in, qd)
            return bard.aba(model, data, tau)

        assert torch.autograd.gradcheck(fn, (q,), eps=1e-6, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize("device", DEVICES)
    def test_aba_grad_wrt_tau_still_works(self, urdf_path, device):
        """Pre-existing d(qdd)/d(tau) path must remain functional."""
        model, data = _model_data(urdf_path, torch.float64, device, 4, True)
        q, qd, _ = _random_state(model, 4, torch.float64, device)
        tau = (0.1 * torch.randn(4, model.nv, dtype=torch.float64, device=device)).requires_grad_(
            True
        )
        bard.update_kinematics(model, data, q, qd)
        bard.aba(model, data, tau).pow(2).sum().backward()
        assert tau.grad is not None and tau.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Cross-check: ABA and RNEA stay mutually consistent on the autograd path.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("floating_base", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_aba_rnea_roundtrip_grad_path(urdf_path, device, floating_base):
    """rnea(aba(tau)) == tau, evaluated entirely on the grad-enabled code path."""
    model, data = _model_data(urdf_path, torch.float64, device, 2, floating_base)
    q, qd, _ = _random_state(model, 2, torch.float64, device, seed=7)
    tau = (0.1 * torch.randn(2, model.nv, dtype=torch.float64, device=device)).requires_grad_(True)
    q = q.requires_grad_(True)  # force the functional (grad) path in both algos

    bard.update_kinematics(model, data, q, qd)
    qdd = bard.aba(model, data, tau)
    bard.update_kinematics(model, data, q, qd)
    tau_rt = bard.rnea(model, data, qdd)

    assert torch.allclose(tau_rt, tau, atol=1e-9), (tau_rt - tau).abs().max().item()
