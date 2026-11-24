import glob
import importlib.util
import math
import os
import tokenize
import unittest
import torch

from residual_benchmark.metrics import (
    check_doubly_stochastic,
    measure_activation_memory_mb,
    measure_signal_and_gradient_flow,
    normalized_erank,
)
from residual_benchmark.residual_architectures import (
    ALL_ARCHITECTURES,
    GeometricOrthogonalResidual,
    build_residual_network,
    clone_shared_block_weights,
    sinkhorn_knopp,
)
from residual_benchmark.run_benchmark import build_adamw_param_groups
from residual_benchmark.synthetic_data import (
    generate_controlled_alignment_step,
    generate_markov_deep_matrix_dataset,
    generate_outlier_stress_matrix_dataset,
    generate_structured_deep_matrix_dataset,
)


class TestResidualBenchmark(unittest.TestCase):
    def test_gor_matches_repo_qwen_moefication_exactly(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        qwen_path = os.path.join(repo_root, "qwen_upcycling", "qwen_moefication.py")
        spec = importlib.util.spec_from_file_location("qwen_moefication_ref", qwen_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        ref_gor = mod.GeometricOrthogonalResidual(eps=1e-7, k=3.0)
        bench_gor = GeometricOrthogonalResidual(eps=1e-7, k=3.0)

        torch.manual_seed(7)
        x = torch.randn(8, 12, 64)
        u = torch.randn(8, 12, 64)
        diff = (ref_gor(x, u) - bench_gor(x, u)).abs().max().item()
        self.assertLess(diff, 1e-6)

    def test_gor_geometric_identities(self):
        gor = GeometricOrthogonalResidual(eps=1e-7, k=3.0)
        x = torch.tensor([[3.0, 4.0, 0.0, 0.0]])
        u_orth = torch.tensor([[0.0, 0.0, 2.0, -1.0]])
        out_orth = gor(x, u_orth)
        self.assertLess((out_orth - (x + u_orth)).abs().max().item(), 1e-5)

        u_pos = 0.5 * x
        expected_pos = x + (2.0 * torch.sigmoid(torch.tensor(3.0))) * u_pos
        self.assertLess((gor(x, u_pos) - expected_pos).abs().max().item(), 1e-5)

        u_neg = -0.5 * x
        expected_neg = x + (2.0 * torch.sigmoid(torch.tensor(-3.0))) * u_neg
        self.assertLess((gor(x, u_neg) - expected_neg).abs().max().item(), 1e-5)

        net = build_residual_network("gor", dim=32, num_layers=4)
        self.assertEqual(net.count_residual_overhead_params(), 0)

    def test_sinkhorn_birkhoff_polytope_invariants(self):
        torch.manual_seed(11)
        for scale in [0.1, 1.0]:
            logits = torch.randn(4, 8, 4, 4) * scale
            ds = sinkhorn_knopp(logits, n_iters=20)
            stats = check_doubly_stochastic(ds)
            self.assertGreaterEqual(stats["min_entry"], 0.0)
            self.assertLess(stats["max_row_err"], 1e-4)
            self.assertLess(stats["max_col_err"], 1e-4)
            self.assertLessEqual(stats["max_spectral_norm"], 1.0 + 1e-4)

        mhc_net = build_residual_network("mhc", dim=32, num_layers=4)
        streams = torch.randn(4, 6, 4, 32)
        _, _, h_res = mhc_net.couplings[0].compute_matrices(streams)
        mhc_stats = check_doubly_stochastic(h_res)
        self.assertGreaterEqual(mhc_stats["min_entry"], 0.0)
        self.assertLess(mhc_stats["max_row_err"], 1e-4)
        self.assertLess(mhc_stats["max_col_err"], 1e-4)
        self.assertLessEqual(mhc_stats["max_spectral_norm"], 1.0 + 1e-4)

        extreme_logits = torch.randn(4, 8, 4, 4) * 15.0
        ds_20 = sinkhorn_knopp(extreme_logits, n_iters=20)
        ds_60 = sinkhorn_knopp(extreme_logits, n_iters=60)
        self.assertTrue(torch.isfinite(ds_20).all().item())
        self.assertTrue(torch.isfinite(ds_60).all().item())
        s20 = check_doubly_stochastic(ds_20)
        s60 = check_doubly_stochastic(ds_60)
        self.assertLess(s60["max_row_err"], s20["max_row_err"])
        self.assertLess(s60["max_col_err"], 1e-4)

    def test_all_architectures_shapes_stats_and_gradients(self):
        torch.manual_seed(19)
        dim = 32
        for depth in [1, 5, 10, 16]:
            base = build_residual_network("pre_norm", dim=dim, num_layers=depth)
            x3d = torch.randn(4, 6, dim)
            target3d = torch.randn(4, 6, dim)

            for arch in ALL_ARCHITECTURES:
                model = build_residual_network(arch, dim=dim, num_layers=depth)
                clone_shared_block_weights(base, model)
                out, stats = model(x3d, return_stats=True, apply_final_norm=False)
                self.assertEqual(out.shape, x3d.shape, f"Shape mismatch for {arch} L={depth}")
                self.assertEqual(
                    len(stats["stream_rms"]),
                    depth + 1,
                    f"stream_rms length mismatch for {arch} L={depth}",
                )
                self.assertEqual(len(stats["dilution_ratio"]), depth)

                flow = measure_signal_and_gradient_flow(model, x3d, target3d)
                self.assertFalse(flow["has_nan_or_inf"], f"NaN/Inf detected for {arch} L={depth}")
                self.assertGreater(flow["input_to_output_grad_ratio"], 0.0)

                x2d = torch.randn(8, dim)
                out2d = model(x2d)
                self.assertEqual(out2d.shape, x2d.shape)

    def test_edge_cases_zero_tiny_and_huge_inputs(self):
        dim = 32
        depth = 8
        for arch in ALL_ARCHITECTURES:
            model = build_residual_network(arch, dim=dim, num_layers=depth)
            for name, x in [
                ("zeros", torch.zeros(2, 4, dim)),
                ("tiny", torch.full((2, 4, dim), 1e-7)),
                ("huge", torch.randn(2, 4, dim) * 1e4),
            ]:
                x_req = x.clone().requires_grad_(True)
                out = model(x_req)
                self.assertTrue(
                    torch.isfinite(out).all().item(),
                    f"Non-finite output for {arch} on {name}",
                )
                loss = out.pow(2).mean()
                loss.backward()
                self.assertIsNotNone(x_req.grad)
                self.assertTrue(
                    torch.isfinite(x_req.grad).all().item(),
                    f"Non-finite input grad for {arch} on {name}",
                )
                model.zero_grad(set_to_none=True)

    def test_block_attnres_bounds_memory_slots_and_activation_memory(self):
        dim = 64
        depth = 24
        x = torch.randn(8, 8, dim)
        m_pre = build_residual_network("pre_norm", dim=dim, num_layers=depth)
        m_mhc = build_residual_network("mhc", dim=dim, num_layers=depth)
        m_blk = build_residual_network("attn_res_block", dim=dim, num_layers=depth, attn_num_blocks=8)
        m_full = build_residual_network("attn_res_full", dim=dim, num_layers=depth)

        _, stats_blk = m_blk(x, return_stats=True)
        _, stats_full = m_full(x, return_stats=True)
        self.assertLessEqual(max(stats_blk["memory_slots"]), 9.0)
        self.assertEqual(max(stats_full["memory_slots"]), float(depth + 1))

        mem_pre = measure_activation_memory_mb(m_pre, x)
        mem_mhc = measure_activation_memory_mb(m_mhc, x)
        mem_blk = measure_activation_memory_mb(m_blk, x)
        mem_full = measure_activation_memory_mb(m_full, x)
        self.assertGreater(mem_mhc, mem_pre)
        self.assertGreater(mem_full, mem_blk)

    def test_controlled_alignment_generator_exact_cosine(self):
        gen = torch.Generator().manual_seed(99)
        x = torch.randn(16, 8, 64, generator=gen)
        for target_rho in [-0.8, -0.4, 0.0, 0.4, 0.8]:
            u = generate_controlled_alignment_step(
                x, target_rho=target_rho, update_scale=0.2, generator=gen
            )
            cos = torch.nn.functional.cosine_similarity(x, u, dim=-1)
            self.assertLess((cos - target_rho).abs().max().item(), 1e-4)

    def test_synthetic_datasets_and_erank(self):
        ds_a = generate_structured_deep_matrix_dataset(
            n_train=64, n_val=16, seq_len=4, dim=32, seed=1
        )
        ds_b = generate_outlier_stress_matrix_dataset(
            n_train=64, n_val=16, seq_len=4, dim=32, seed=2
        )
        self.assertEqual(ds_a.x_train.shape, (64, 4, 32))
        self.assertEqual(ds_b.x_val.shape, (16, 4, 32))
        self.assertGreater(ds_b.metadata["input_max_to_rms"], ds_a.metadata["input_max_to_rms"])
        er = normalized_erank(ds_a.x_train)
        self.assertGreater(er, 0.2)
        self.assertLessEqual(er, 1.0 + 1e-5)

    def test_no_dead_parameters_in_any_architecture(self):
        torch.manual_seed(29)
        dim = 32
        x = torch.randn(4, 4, dim)
        y = torch.randn(4, 4, dim)
        for depth in [4, 12]:
            for arch in ALL_ARCHITECTURES:
                model = build_residual_network(arch, dim=dim, num_layers=depth)
                opt = torch.optim.SGD(model.parameters(), lr=0.05)
                for _ in range(3):
                    opt.zero_grad(set_to_none=True)
                    out = model(x, apply_final_norm=True)
                    loss = torch.nn.functional.mse_loss(out, y)
                    loss.backward()
                    opt.step()
                dead = [n for n, p in model.named_parameters() if p.grad is None]
                self.assertEqual(
                    dead,
                    [],
                    f"Dead parameter(s) with grad=None in {arch} L={depth}: {dead}",
                )

    def test_gated_residual_breaks_stream_symmetry_during_training(self):
        torch.manual_seed(31)
        dim = 32
        ds = generate_markov_deep_matrix_dataset(
            n_train=128, n_val=32, seq_len=4, dim=dim, seed=31
        )
        model = build_residual_network("gated_res", dim=dim, num_layers=8)
        opt = torch.optim.AdamW(build_adamw_param_groups(model, weight_decay=1e-2), lr=2e-3)
        for step in range(10):
            opt.zero_grad(set_to_none=True)
            out = model(ds.x_train[:32], apply_final_norm=True)
            loss = torch.nn.functional.mse_loss(out, ds.y_train[:32])
            loss.backward()
            opt.step()

        with torch.no_grad():
            streams = ds.x_val[:8].unsqueeze(-2).expand(8, 4, model.n_streams, dim).clone()
            max_stream_diff = 0.0
            for norm, block, gate in zip(model.norms, model.blocks, model.gates):
                h_in, g_res, g_out = gate.compute_read_and_write_gates(streams)
                u = block(norm(h_in))
                streams = g_res * streams + g_out * u.unsqueeze(-2)
                diff = (streams - streams[..., :1, :]).abs().max().item()
                max_stream_diff = max(max_stream_diff, diff)
        self.assertGreater(max_stream_diff, 1e-4)

    def test_mhc_maintains_birkhoff_invariant_after_training(self):
        torch.manual_seed(37)
        dim = 32
        ds = generate_markov_deep_matrix_dataset(
            n_train=128, n_val=32, seq_len=4, dim=dim, seed=37
        )
        model = build_residual_network("mhc", dim=dim, num_layers=8)
        opt = torch.optim.AdamW(build_adamw_param_groups(model, weight_decay=1e-2), lr=2e-3)
        for _ in range(15):
            opt.zero_grad(set_to_none=True)
            out = model(ds.x_train[:32], apply_final_norm=True)
            loss = torch.nn.functional.mse_loss(out, ds.y_train[:32])
            loss.backward()
            opt.step()

        with torch.no_grad():
            streams = ds.x_val[:8].unsqueeze(-2).expand(8, 4, model.n_streams, dim).clone()
            for norm, block, coupling in zip(model.norms, model.blocks, model.couplings):
                h_pre, h_post, h_res = coupling.compute_matrices(streams)
                st = check_doubly_stochastic(h_res)
                self.assertLess(st["max_row_err"], 1e-4)
                self.assertLess(st["max_col_err"], 1e-4)
                self.assertLessEqual(st["max_spectral_norm"], 1.0 + 1e-4)
                h_in = torch.einsum("...n,...nd->...d", h_pre, streams)
                u = block(norm(h_in))
                streams = torch.einsum("...ij,...jd->...id", h_res, streams) + h_post.unsqueeze(-1) * u.unsqueeze(-2)

    def test_markov_dataset_and_shared_validation_slicing(self):
        full_ds = generate_markov_deep_matrix_dataset(
            n_train=128, n_val=32, seq_len=4, dim=32, seed=5
        )
        small_ds = full_ds.slice_train(32)
        self.assertEqual(small_ds.x_train.shape, (32, 4, 32))
        self.assertEqual(small_ds.y_train.shape, (32, 4, 32))
        self.assertEqual(small_ds.metadata["n_train"], 32.0)
        self.assertTrue(torch.equal(small_ds.x_val, full_ds.x_val))
        self.assertTrue(torch.equal(small_ds.y_val, full_ds.y_val))

    def test_adamw_param_groups_exclude_biases_and_norms(self):
        for arch in ALL_ARCHITECTURES:
            model = build_residual_network(arch, dim=32, num_layers=4)
            groups = build_adamw_param_groups(model, weight_decay=1e-2)
            decay_ids = {id(p) for p in groups[0]["params"]}
            no_decay_ids = {id(p) for p in groups[1]["params"]}
            for name, p in model.named_parameters():
                if name.endswith(("bias", "bias_res", "bias_pre", "bias_post", "bias_out")) or "norm" in name or p.ndim < 2:
                    self.assertIn(id(p), no_decay_ids, f"{arch}:{name} should not have weight decay")
                    self.assertNotIn(id(p), decay_ids)

    def test_zero_comments_policy_in_benchmark_package(self):
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        py_files = sorted(glob.glob(os.path.join(pkg_dir, "*.py")))
        self.assertGreaterEqual(len(py_files), 5)
        for path in py_files:
            with open(path, "rb") as f:
                comments = [
                    tok
                    for tok in tokenize.tokenize(f.readline)
                    if tok.type == tokenize.COMMENT
                ]
            self.assertEqual(
                len(comments),
                0,
                f"Found forbidden # comment(s) in {os.path.basename(path)}: {comments[:3]}",
            )


if __name__ == "__main__":
    unittest.main()
