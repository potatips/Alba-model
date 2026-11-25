import glob
import json
import os
import subprocess
import sys
import tempfile
import tokenize
import unittest
import torch

from gor_research.residuals import ALL_LM_RESIDUALS, GORv2Residual, _compute_gor_components
from gor_research.run_colab_study import MatrixResidualBlock
from gor_research.transformer_lm import FromScratchTransformerLM, TransformerConfig, clone_transformer_backbone
from src.config import ModelArgs
from src.model import GeometricOrthogonalResidual, MiniLLM


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


if __name__ == "__main__":
    unittest.main()
