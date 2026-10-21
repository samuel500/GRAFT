"""Unit tests for DualLoReFT, composite adapter plans, and the vLLM-side
VLLMDualLoReFT math (exercised without vLLM via a stub decoder layer)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from gradient_routing import (
    DualLoReFT,
    DualMLPAdapter,
    _orthonormal_rows,
    adapter_modules_for_sync,
    adapter_plan_kappas,
    adapter_plan_role_params,
    apply_adapter_plan,
    build_adapter_plan,
    collect_routing_params,
    fused_routing,
    parse_adapter_spec,
    parse_layer_spec,
    set_scales,
)
from vllm_mlp_adapter import VLLMDualLoReFT

H = 32


def _rand_loreft(retain_rank=3, forget_rank=2, seed=0):
    torch.manual_seed(seed)
    m = DualLoReFT(H, retain_rank, forget_rank, init="pyreft")
    with torch.no_grad():   # non-trivial biases (pyreft init is small but nonzero)
        for p in m.parameters():
            if p.dim() == 1:
                p.normal_()
    return m


def _tiny_qwen3(num_layers=4, seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=97, hidden_size=H, intermediate_size=64,
                      num_hidden_layers=num_layers, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    return Qwen3ForCausalLM(cfg).eval()


# --- DualLoReFT math ---


def test_orthonormal_rows_fixed_point_and_orthonormal():
    torch.manual_seed(0)
    q = torch.empty(4, H)
    nn.init.orthogonal_(q)
    assert torch.allclose(_orthonormal_rows(q), q, atol=1e-6)
    R = _orthonormal_rows(torch.randn(4, H))
    assert torch.allclose(R @ R.t(), torch.eye(4), atol=1e-5)


@pytest.mark.parametrize("ranks", [(3, 2), (1, 1), (8, 0)])
def test_identity_init_is_exact_identity(ranks):
    m = DualLoReFT(H, *ranks, init="identity")
    h = torch.randn(2, 5, H)
    assert torch.equal(m(h), h)


def test_matches_loreft_formula():
    m = _rand_loreft()
    h = torch.randn(2, 5, H)
    want = h.clone()
    for role in ("retain", "forget"):
        R = _orthonormal_rows(getattr(m, f"rot_{role}"))
        proj = getattr(m, f"proj_{role}")
        want = want + (h @ proj.weight.t() + proj.bias - h @ R.t()) @ R
    assert torch.allclose(m(h), want, atol=1e-5)


def test_forget_scale_zero_matches_retain_only():
    m = _rand_loreft()
    retain_only = DualLoReFT(H, 3, 0, init="pyreft")
    retain_only.rot_retain.data.copy_(m.rot_retain.data)
    retain_only.proj_retain.load_state_dict(m.proj_retain.state_dict())
    h = torch.randn(2, 5, H)
    m.forget_scale = 0.0
    assert torch.equal(m(h), retain_only(h))


def test_role_params_partition():
    m = _rand_loreft()
    r, f = m.get_retain_params(), m.get_forget_params()
    assert len(r) == len(f) == 3
    assert {id(p) for p in r}.isdisjoint({id(p) for p in f})
    assert {id(p) for p in r + f} == {id(p) for p in m.parameters()}


def test_dtype_cast_keeps_fp32_params():
    m = _rand_loreft()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    m.to(torch.bfloat16)
    for k, v in m.state_dict().items():
        assert v.dtype == torch.float32
        assert torch.equal(v, before[k])
    out = m(torch.randn(2, 5, H, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16


# --- fused routing + split-moment natural gradient ---


def _grads(m, h, loss_fn):
    m.zero_grad()
    h = h.clone().requires_grad_(True)
    loss_fn(m(h)).backward()
    return {n: p.grad.clone() for n, p in m.named_parameters()}, h.grad.clone()


def test_fused_routing_gates_param_grads_only():
    m = _rand_loreft()
    h = torch.randn(1, 6, H)
    w = torch.randn(1, 6, H)
    loss = lambda out: (out * w).sum()
    g_ref, hg_ref = _grads(m, h, loss)

    ones = torch.ones(1, 6, 1)
    zeros = torch.zeros(1, 6, 1)
    with fused_routing(ones, zeros, ones):          # retain gated off everywhere
        g, hg = _grads(m, h, loss)
    assert torch.allclose(hg, hg_ref, atol=1e-5)    # input gradient untouched
    for n in g:
        if "retain" in n:
            assert torch.count_nonzero(g[n]) == 0
        else:
            assert torch.allclose(g[n], g_ref[n], atol=1e-5)

    # Per-token gate: only token 2 trains forget. Equals the grad of a loss
    # restricted to token 2, for forget params.
    fmask = torch.zeros(1, 6, 1)
    fmask[0, 2] = 1
    with fused_routing(ones, ones, fmask):
        g, _ = _grads(m, h, loss)
    g_tok, _ = _grads(m, h, lambda out: (out[:, 2] * w[:, 2]).sum())
    for n in g:
        if "forget" in n:
            assert torch.allclose(g[n], g_tok[n], atol=1e-5)


def test_split_moment_capture_on_loreft_plan():
    """PreRoutingGradAccumulator on a LoReFT plan, under fused routing with a
    bad (forget-only) span and a coherence (retain-only, forget-off) span:
    .grad holds the routed gradient while _pre_routing_grad holds the natural
    one, i.e. the gradient of an unrouted forward at the same forget scales."""
    from gradient_routing import PreRoutingGradAccumulator

    model = _tiny_qwen3()
    apply_adapter_plan(model, build_adapter_plan(4, "loreft:2", "loreft:1"), loreft_init="pyreft")
    params = [p for p in model.parameters() if p.requires_grad]
    ids = torch.randint(0, 97, (1, 12))
    w = torch.randn(1, 12, 97)
    loss_fn = lambda: (model(ids).logits * w).sum()
    # tokens 0-3 good, 4-7 bad (retain gated off), 8-11 coherence (forget off)
    ffs = torch.tensor([1.0] * 8 + [0.0] * 4).view(1, 12, 1)
    rgm = torch.tensor([1.0] * 4 + [0.0] * 4 + [1.0] * 4).view(1, 12, 1)
    fgm = torch.tensor([1.0] * 8 + [0.0] * 4).view(1, 12, 1)

    cap = PreRoutingGradAccumulator(model)
    cap.reset()
    with fused_routing(ffs, rgm, fgm):
        loss_fn().backward()
        cap.flush(ffs)
    routed = [p.grad.clone() for p in params]
    natural = [p._pre_routing_grad.clone() for p in params]
    cap.remove()

    with fused_routing(ffs, torch.ones_like(rgm), torch.ones_like(fgm)):
        model.zero_grad()
        loss_fn().backward()
    for p, n, r in zip(params, natural, routed):
        assert torch.allclose(n, p.grad, atol=1e-4, rtol=1e-4)
    assert any(not torch.allclose(n, r, atol=1e-6) for n, r in zip(natural, routed))


def test_natural_adapter_output_reproduces_param_grads():
    m = _rand_loreft()
    h = torch.randn(1, 6, H)
    g_out = torch.randn(1, 6, H)
    g_ref, _ = _grads(m, h, lambda out: (out * g_out).sum())
    params = m.get_retain_params() + m.get_forget_params()
    nat = torch.autograd.grad(m.natural_adapter_output(h, 1.0), params, grad_outputs=g_out)
    names = {id(p): n for n, p in m.named_parameters()}
    for p, gp in zip(params, nat):
        assert torch.allclose(gp, g_ref[names[id(p)]], atol=1e-5)


# --- vLLM engine module (stub decoder layer, no vLLM import) ---


class _StubLayer(nn.Module):
    """Mimics a vLLM decoder layer: returns (mlp_out, residual) with the block
    output h = mlp_out + residual."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, positions, hidden_states, residual):
        return hidden_states, residual


class _StubPunica:
    def __init__(self, idx):
        self._token_lora_indices = idx


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_engine_module_matches_trainer_per_slot(dtype):
    S = 4
    trainers = {1: _rand_loreft(seed=1), 3: _rand_loreft(seed=3)}
    eng = VLLMDualLoReFT(_StubLayer().to(dtype), H, S, retain_rank=3, forget_rank=2)
    for slot, m in trainers.items():
        eng.set_weights(slot, {k: v.to(dtype) for k, v in m.export_weights().items()})
    eng.set_scales(3, 1.0, 0.0)                 # slot 3: retain-only
    trainers[3].forget_scale = 0.0

    T = 7
    idx = torch.tensor([1, 3, -1, 1, 3, -1, 1])  # -1 = no adapter
    eng.set_mapping(_StubPunica(idx))
    mlp_out = torch.randn(T, H).to(dtype)
    resid = torch.randn(T, H).to(dtype)
    out, new_resid = eng(None, mlp_out, resid)
    assert torch.count_nonzero(new_resid) == 0

    h = mlp_out + resid
    for t in range(T):
        want = h[t] if idx[t] < 0 else trainers[int(idx[t])](h[t:t + 1])[0]
        tol = 1e-5 if dtype == torch.float32 else 3e-2
        assert torch.allclose(out[t].float(), want.float(), atol=tol, rtol=tol), t


def test_engine_module_reset_zeroes_slot():
    eng = VLLMDualLoReFT(_StubLayer(), H, 2, retain_rank=1, forget_rank=1)
    eng.set_weights(0, _rand_loreft(1, 1).export_weights())
    eng.reset_lora(0)
    eng.set_mapping(_StubPunica(torch.zeros(3, dtype=torch.long)))
    x, r = torch.randn(3, H), torch.randn(3, H)
    out, _ = eng(None, x, r)
    assert torch.equal(out, x + r)


# --- plan parsing ---


def test_parse_specs():
    assert parse_adapter_spec("loreft:8") == ("loreft", 8)
    assert parse_adapter_spec("mlp:64") == ("mlp", 64)
    assert parse_adapter_spec("none") == ("none", 0)
    with pytest.raises(AssertionError):
        parse_adapter_spec("lora:8")
    assert parse_layer_spec(None, 6) == list(range(6))
    assert parse_layer_spec("1-3", 6) == [1, 2, 3]
    assert parse_layer_spec("0-5:2,5", 6) == [0, 2, 4, 5]
    with pytest.raises(AssertionError):
        parse_layer_spec("4-6", 6)


def test_build_plan_mixed_kinds_and_layers():
    plan = build_adapter_plan(4, "mlp:8", "loreft:2", retain_layers="0-2", forget_layers="2-3")
    assert plan == [
        {"kind": "mlp", "layer": 0, "retain": 8, "forget": 0},
        {"kind": "mlp", "layer": 1, "retain": 8, "forget": 0},
        {"kind": "mlp", "layer": 2, "retain": 8, "forget": 0},
        {"kind": "loreft", "layer": 2, "retain": 0, "forget": 2},
        {"kind": "loreft", "layer": 3, "retain": 0, "forget": 2},
    ]
    same = build_adapter_plan(2, "loreft:1", "loreft:1")
    assert same == [{"kind": "loreft", "layer": i, "retain": 1, "forget": 1} for i in range(2)]


# --- plans on a real (tiny, random) Qwen3 ---


def test_apply_plan_identity_at_init_and_routing_partition():
    base = _tiny_qwen3()
    ids = torch.randint(0, 97, (2, 9))
    with torch.no_grad():
        ref = base(ids).logits
    plan = build_adapter_plan(4, "mlp:8", "loreft:2", retain_layers="0-2", forget_layers="1-3")
    apply_adapter_plan(base, plan)
    with torch.no_grad():
        assert torch.allclose(base(ids).logits, ref, atol=1e-6)

    retain, forget = collect_routing_params(base)
    n = adapter_plan_role_params(base)
    assert sum(p.numel() for p in retain) == n["retain"] == 3 * (3 * H * 8)
    assert sum(p.numel() for p in forget) == n["forget"] == 3 * (2 * H * 2 + 2)
    assert [(k, l) for k, l, _ in adapter_modules_for_sync(base)] == \
        [(e["kind"], e["layer"]) for e in plan]


def test_plan_scales_ablate_forget_and_checkpoint_roundtrip():
    model = _tiny_qwen3()
    plan = build_adapter_plan(4, "loreft:2", "loreft:2")
    apply_adapter_plan(model, plan, loreft_init="pyreft")
    ids = torch.randint(0, 97, (2, 9))

    # Ablating forget == a retain-only plan with the same retain weights.
    saved = {k: v.clone() for k, v in model.named_parameters() if v.requires_grad}
    retain_only = _tiny_qwen3()
    apply_adapter_plan(retain_only, build_adapter_plan(4, "loreft:2", "none"))
    retain_only.load_state_dict({k: v for k, v in saved.items() if "retain" in k}, strict=False)
    set_scales(model, 1.0, 0.0)
    with torch.no_grad():
        assert torch.allclose(model(ids).logits, retain_only(ids).logits, atol=1e-5)
    set_scales(model, 1.0, 1.0)

    # Trainable-params-only checkpoint (train.py _save_adapter_checkpoint)
    # restores the exact model, and survives a whole-model bf16 cast.
    restored = _tiny_qwen3()
    apply_adapter_plan(restored, plan)
    restored.load_state_dict(saved, strict=False)
    with torch.no_grad():
        assert torch.equal(model(ids).logits, restored(ids).logits)
    restored.to(torch.bfloat16)
    assert all(p.dtype == torch.float32 for n, p in restored.named_parameters() if "loreft" in n)


def test_kappas():
    model = _tiny_qwen3()
    apply_adapter_plan(model, build_adapter_plan(4, "loreft:8", "loreft:8"))
    cfg = {"retain_adapter": "loreft:8", "forget_adapter": "loreft:8",
           "role_params": adapter_plan_role_params(model)}
    assert adapter_plan_kappas(cfg) == (2.0, 2.0)
    mixed = {"retain_adapter": "mlp:64", "forget_adapter": "loreft:1", "role_params": cfg["role_params"]}
    with pytest.raises(NotImplementedError):
        adapter_plan_kappas(mixed)
    assert adapter_plan_kappas({**mixed, "adapter_kappa": "1.5,3"}) == (1.5, 3.0)


def test_wire_roundtrip_composite():
    """Client flat pack -> server plan unflatten recovers each module's
    exported tensors (bf16 on the wire), in plan order."""
    from vllm_client import _pack_weights_flat
    from vllm_mlp_adapter import VLLMCompositeAdapterManager

    model = _tiny_qwen3()
    plan = build_adapter_plan(4, "mlp:8", "loreft:2", forget_layers="1-3")
    apply_adapter_plan(model, plan, loreft_init="pyreft")
    payload, shapes, modules = _pack_weights_flat(model)
    assert modules == [[e["kind"], e["layer"]] for e in plan]

    mgr = object.__new__(VLLMCompositeAdapterManager)
    mgr.plan = plan
    flat = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16)
    views = mgr._unflatten_plan_views(flat, shapes)
    for (_, _, module), w in zip(adapter_modules_for_sync(model), views):
        exported = module.export_weights()
        assert list(w) == list(exported)
        for k in w:
            assert torch.equal(w[k], exported[k].to(torch.bfloat16)), k


def test_mlp_only_plan_matches_legacy_dual_mlp():
    from gradient_routing import apply_dual_mlp
    legacy = _tiny_qwen3()
    apply_dual_mlp(legacy, retain_neurons=8, forget_neurons=8)
    composite = _tiny_qwen3()
    apply_adapter_plan(composite, build_adapter_plan(4, "mlp:8", "mlp:8"))
    assert [n for n, _ in legacy.named_parameters()] == [n for n, _ in composite.named_parameters()]
    assert all(isinstance(m, DualMLPAdapter) for _, _, m in adapter_modules_for_sync(composite))
