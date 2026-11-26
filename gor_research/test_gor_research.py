import glob
import json
import os
import subprocess
import sys
import tempfile
import tokenize
import unittest
import torch

from gor_research.paper_scaling_study import _build_paper_optimizer
from gor_research.residuals import ALL_LM_RESIDUALS, GORv2Residual, _compute_gor_components, sinkhorn_knopp
from gor_research.run_colab_study import MatrixResidualBlock
from gor_research.transformer_lm import FromScratchTransformerLM, TransformerConfig, clone_transformer_backbone
from src.config import ModelArgs
from src.model import GeometricOrthogonalResidual, MiniLLM
from src.optimizer import GEOMuon


class TestGORResearch(unittest.TestCase):
    def test_gor_v2_algebraic_equivalence(self):
        torch.manual_seed(11)
        x = torch.randn(4, 16, 64)
        u = torch.randn(4, 16, 64)
        op = GORv2Residual(k=3.0, eps=1e-7)
        fast_out = op(x, u)
        xf, _, u_par, u_perp, rho, _ = _compute_gor_components(x, u, 1e-7)
        ref_out = (xf + u_perp + (2.0 * torch.sigmoid(3.0 * rho)) * u_par).to(dtype=x.dtype)
        self.assertLess((fast_out - ref_out).abs().max().item(), 1e-5)

    def test_zero_comments_in_gor_research(self):
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        for py_file in sorted(glob.glob(os.path.join(pkg_dir, "*.py"))):
            with open(py_file, "rb") as f:
                tokens = tokenize.tokenize(f.readline)
                comments = [tok.string for tok in tokens if tok.type == tokenize.COMMENT]
                self.assertEqual(comments, [], f"Found comments in {py_file}: {comments}")

        for nb_path in sorted(glob.glob(os.path.join(pkg_dir, "*.ipynb"))):
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    src = "".join(cell["source"])
                    for line in src.splitlines():
                        self.assertFalse(line.strip().startswith("#"), f"Found comment line in {nb_path}: {line}")

    def test_all_variants_forward_backward_and_causality(self):
        torch.manual_seed(7)
        ref_cfg = TransformerConfig(vocab_size=64, dim=32, n_layers=4, n_heads=4, max_seq_len=16, residual_type="pre_norm")
        ref_lm = FromScratchTransformerLM(ref_cfg)
        x = torch.randint(0, 64, (2, 8))
        y = torch.randint(0, 64, (2, 8))
        x_pert = x.clone()
        x_pert[:, 6:] = (x_pert[:, 6:] + 13) % 64

        for arch in ALL_LM_RESIDUALS:
            cfg = TransformerConfig(vocab_size=64, dim=32, n_layers=4, n_heads=4, max_seq_len=16, residual_type=arch)
            lm = FromScratchTransformerLM(cfg)
            clone_transformer_backbone(ref_lm, lm)

            logits1, loss1, geom = lm(x, targets=y, collect_geometry=True)
            logits2, _, _ = lm(x_pert, targets=y)
            self.assertEqual(logits1.shape, (2, 8, 64))
            self.assertLess((logits1[:, :6, :] - logits2[:, :6, :]).abs().max().item(), 1e-5)
            self.assertEqual(len(geom["conflict_rates"]), 8)

            loss1.backward()
            grad_norm = sum(float(p.grad.abs().sum().item()) for p in lm.parameters() if p.grad is not None)
            self.assertGreater(grad_norm, 0.0)

    def test_matrix_block_zero_param_and_smoke_cli(self):
        for zero_arch in ("pre_norm", "gor_v2", "gor_bounded", "gor_antidilute"):
            blk = MatrixResidualBlock(32, 4, zero_arch)
            self.assertEqual(blk.count_overhead_params(), 0)

        self.assertEqual(
            MatrixResidualBlock(32, 4, "dual_bank_gor_v2").count_overhead_params(),
            MatrixResidualBlock(32, 4, "std_dual_bank").count_overhead_params(),
        )
        self.assertEqual(
            MatrixResidualBlock(32, 4, "attn_res_gor_v2").count_overhead_params(),
            MatrixResidualBlock(32, 4, "attn_res_block").count_overhead_params(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            res = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "gor_research.run_colab_study",
                    "--device",
                    "cpu",
                    "--quick",
                    "--synthetic-lm",
                    "--variants",
                    "pre_norm,gor_v2,std_dual_bank,dual_bank_gor_v2,attn_res_block,attn_res_gor_v2",
                    "--output-dir",
                    tmpdir,
                ],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("Saved HTML:", res.stdout)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "gor_from_scratch_results.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "gor_from_scratch_report.png")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "gor_from_scratch_report.html")))

    def test_alba_minillm_integration_and_reasoning_smoke(self):
        torch.manual_seed(13)
        x = torch.randn(2, 8, 32)
        u = torch.randn(2, 8, 32)
        op = GeometricOrthogonalResidual(eps=1e-7, k=3.0)
        xf, _, u_par, u_perp, rho, _ = _compute_gor_components(x, u, 1e-7)
        ref = (xf + u_perp + (2.0 * torch.sigmoid(3.0 * rho)) * u_par).to(dtype=x.dtype)
        self.assertLess((op(x, u) - ref).abs().max().item(), 1e-5)

        idx = torch.randint(0, 64, (2, 8))
        idx_pert = idx.clone()
        idx_pert[:, 6:] = (idx_pert[:, 6:] + 17) % 64
        targets = torch.randint(0, 64, (2, 8))

        combos = [
            ("standard", "swiglu"),
            ("gor", "swiglu"),
            ("pre_norm", "swiglu"),
            ("gor_v2", "swiglu"),
            ("std_dual_bank", "swiglu"),
            ("dual_bank_gor_v2", "swiglu"),
            ("attn_res_block", "swiglu"),
            ("attn_res_gor_v2", "swiglu"),
            ("attn_res_gor_v2", "os_moe"),
        ]
        for res_type, ffn_type in combos:
            args = ModelArgs(
                vocab_size=64,
                dim=32,
                n_layers=4,
                n_heads=4,
                max_seq_len=16,
                residual_type=res_type,
                ffn_type=ffn_type,
                n_routed_experts=4,
                num_experts_per_tok=2,
            )
            model = MiniLLM(args)
            logits1, loss1 = model(idx, targets)
            logits2, _ = model(idx_pert, targets)
            self.assertEqual(logits1.shape, (2, 8, 64))
            self.assertLess((logits1[:, :6, :] - logits2[:, :6, :]).abs().max().item(), 1e-5)
            loss1.backward()
            grad_sum = sum(float(p.grad.abs().sum().item()) for p in model.parameters() if p.grad is not None)
            self.assertGreater(grad_sum, 0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            res = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "gor_research.alba_reasoning_study",
                    "--device",
                    "cpu",
                    "--quick",
                    "--variants",
                    "pre_norm,gor_v2,dual_bank_gor_v2,attn_res_block,attn_res_gor_v2,attn_res_gor_v2_moe",
                    "--output-dir",
                    tmpdir,
                ],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("Saved HTML:", res.stdout)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "alba_reasoning_results.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "alba_reasoning_report.png")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "alba_reasoning_report.html")))

    def test_mhc_sinkhorn_and_grad_checkpoint_parity(self):
        torch.manual_seed(21)
        logits = torch.randn(3, 5, 4, 4)
        ds = sinkhorn_knopp(logits, iters=12)
        self.assertTrue(torch.all(ds > 0).item())
        self.assertLess((ds.sum(dim=-1) - 1.0).abs().max().item(), 1e-4)
        self.assertLess((ds.sum(dim=-2) - 1.0).abs().max().item(), 1e-4)

        for arch in ("pre_norm", "mhc", "attn_res_gor_v2"):
            x = torch.randint(0, 64, (2, 8))
            y = torch.randint(0, 64, (2, 8))
            torch.manual_seed(31)
            m_no_ckpt = FromScratchTransformerLM(
                TransformerConfig(vocab_size=64, dim=32, n_layers=4, n_heads=4, max_seq_len=16, residual_type=arch, gradient_checkpointing=False)
            )
            torch.manual_seed(31)
            m_ckpt = FromScratchTransformerLM(
                TransformerConfig(vocab_size=64, dim=32, n_layers=4, n_heads=4, max_seq_len=16, residual_type=arch, gradient_checkpointing=True)
            )
            l1, loss1, _ = m_no_ckpt(x, targets=y)
            l2, loss2, _ = m_ckpt(x, targets=y)
            self.assertLess((l1 - l2).abs().max().item(), 1e-5)
            self.assertLess(abs(loss1.item() - loss2.item()), 1e-5)
            loss1.backward()
            loss2.backward()
            for (n1, p1), (_, p2) in zip(m_no_ckpt.named_parameters(), m_ckpt.named_parameters()):
                if p1.grad is not None:
                    self.assertLess((p1.grad - p2.grad).abs().max().item(), 1e-5, f"Grad mismatch in {arch}:{n1}")

    def test_paper_scaling_smoke_and_latex_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            out_dir = os.path.join(tmpdir, "out")
            res = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "gor_research.paper_scaling_study",
                    "--preset",
                    "smoke",
                    "--synthetic-lm",
                    "--seeds",
                    "42,43",
                    "--variants",
                    "pre_norm,pre_norm_isotime,mhc,gor_v2,attn_res_block,attn_res_gor_v2",
                    "--optimizer",
                    "geomuon",
                    "--device",
                    "cpu",
                    "--cache-dir",
                    cache_dir,
                    "--output-dir",
                    out_dir,
                ],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("Saved TEX:", res.stdout)
            for fname in ("paper_results.json", "paper_figures.png", "paper_figures.pdf", "paper_tables.tex", "paper_report.html"):
                self.assertTrue(os.path.exists(os.path.join(out_dir, fname)), f"Missing {fname}")
            with open(os.path.join(out_dir, "paper_results.json"), "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["summary"]["pre_norm_isotime"]["actual_layers"], 6)
            self.assertEqual(payload["summary"]["mhc"]["n_seeds"], 2)

    def test_mid_run_two_session_resume_exact_parity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            out_full = os.path.join(tmpdir, "full")
            out_split = os.path.join(tmpdir, "split")
            cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_args = [
                sys.executable,
                "-m",
                "gor_research.paper_scaling_study",
                "--preset",
                "smoke",
                "--synthetic-lm",
                "--seeds",
                "42",
                "--variants",
                "pre_norm,attn_res_gor_v2",
                "--steps",
                "12",
                "--device",
                "cpu",
                "--cache-dir",
                cache_dir,
            ]
            subprocess.run(base_args + ["--output-dir", out_full], cwd=cwd, capture_output=True, text=True, check=True)

            s1 = subprocess.run(
                base_args + ["--max-steps-per-session", "7", "--output-dir", out_split],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("[SessionPaused]", s1.stdout)
            self.assertFalse(os.path.exists(os.path.join(out_split, "paper_results.json")))
            self.assertTrue(os.path.exists(os.path.join(out_split, "step_checkpoints", "pre_norm__seed42_latest.pt")))

            subprocess.run(
                base_args + ["--max-steps-per-session", "7", "--output-dir", out_split],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                base_args + ["--max-steps-per-session", "7", "--output-dir", out_split],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertTrue(os.path.exists(os.path.join(out_split, "paper_results.json")))
            self.assertFalse(os.path.exists(os.path.join(out_split, "step_checkpoints", "pre_norm__seed42_latest.pt")))

            with open(os.path.join(out_full, "paper_results.json"), "r", encoding="utf-8") as f:
                r_full = json.load(f)
            with open(os.path.join(out_split, "paper_results.json"), "r", encoding="utf-8") as f:
                r_split = json.load(f)

            for v in ("pre_norm", "attn_res_gor_v2"):
                self.assertAlmostEqual(
                    r_full["summary"][v]["val_bpb"]["mean"],
                    r_split["summary"][v]["val_bpb"]["mean"],
                    places=5,
                )
                self.assertEqual(
                    r_full["runs"][v][0]["tokens_seen"],
                    r_split["runs"][v][0]["tokens_seen"],
                )

    def test_geomuon_moonlight_rms_scaling_and_convergence(self):
        torch.manual_seed(55)
        for shape in ((384, 384), (1024, 384), (384, 1024)):
            p_before = torch.randn(shape)
            p = torch.nn.Parameter(p_before.clone())
            p.grad = torch.randn(shape)
            opt = GEOMuon([{"params": [p], "use_muon": True, "lr": 1e-3, "weight_decay": 0.0, "rms_scale": 0.2}])
            opt.step()
            update_rms = float((p.detach() - p_before).pow(2).mean().sqrt().item())
            self.assertGreaterEqual(update_rms, 0.13e-3, f"Under-scaled RMS {update_rms} for shape {shape}")
            self.assertLessEqual(update_rms, 0.26e-3, f"Over-scaled RMS {update_rms} for shape {shape}")

        vocab, seq = 64, 32
        base = torch.arange(vocab).repeat(4)
        data = torch.stack([torch.roll(base[: seq + 1], shifts=i % 8) for i in range(128)])
        train_b, val_b = data[:96], data[96:]
        final_losses = {}
        init_loss = None
        for opt_name in ("adamw", "geomuon"):
            torch.manual_seed(77)
            cfg = TransformerConfig(
                vocab_size=vocab,
                dim=64,
                n_layers=4,
                n_heads=4,
                max_seq_len=seq,
                residual_type="attn_res_gor_v2",
            )
            lm = FromScratchTransformerLM(cfg)
            opt = _build_paper_optimizer(lm, opt_name, lr=2e-3, weight_decay=0.01, device=torch.device("cpu"))
            if init_loss is None:
                with torch.no_grad():
                    _, l0, _ = lm(val_b[:, :-1], targets=val_b[:, 1:])
                    init_loss = float(l0.item())
            for _ in range(50):
                opt.zero_grad(set_to_none=True)
                _, loss, _ = lm(train_b[:32, :-1], targets=train_b[:32, 1:])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
                opt.step()
            with torch.no_grad():
                _, l_end, _ = lm(val_b[:, :-1], targets=val_b[:, 1:])
                final_losses[opt_name] = float(l_end.item())

        print(f"\n  [Convergence 50-step] init={init_loss:.4f} | adamw={final_losses['adamw']:.4f} | geomuon={final_losses['geomuon']:.4f}")
        self.assertLess(final_losses["geomuon"], init_loss * 0.6)
        self.assertLessEqual(final_losses["geomuon"], final_losses["adamw"] + 0.15)


if __name__ == "__main__":
    unittest.main()
