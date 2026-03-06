"""Tests for random seed gaps fixed in mindtorch_v2.

Run with:
    python tests/run_test_v2.py -vs tests/test_random_seed_v2.py
"""
import unittest


class TestRemainingRandomGaps(unittest.TestCase):
    """Tests for all random-seed gap fixes."""

    # ------------------------------------------------------------------
    # Step 1: randn_like / rand_like forward generator
    # ------------------------------------------------------------------
    def test_randn_like_with_generator(self):
        import torch
        x = torch.zeros(3, 4)
        g1 = torch.Generator()
        g1.manual_seed(42)
        a = torch.randn_like(x, generator=g1)

        g2 = torch.Generator()
        g2.manual_seed(42)
        b = torch.randn_like(x, generator=g2)

        self.assertTrue(torch.allclose(a, b), "randn_like with same generator seed should be reproducible")

    def test_rand_like_with_generator(self):
        import torch
        x = torch.zeros(3, 4)
        g1 = torch.Generator()
        g1.manual_seed(42)
        a = torch.rand_like(x, generator=g1)

        g2 = torch.Generator()
        g2.manual_seed(42)
        b = torch.rand_like(x, generator=g2)

        self.assertTrue(torch.allclose(a, b), "rand_like with same generator seed should be reproducible")

    # ------------------------------------------------------------------
    # Step 2: randint_ — CPU seeding bug fix + generator support
    # ------------------------------------------------------------------
    def test_randint_inplace_reproducible(self):
        """Verify CPU randint_ uses managed RNG (not np.random)."""
        import torch
        torch.manual_seed(123)
        a = torch.empty(5, 5)
        a.randint_(0, 100)
        val_a = a.clone()

        torch.manual_seed(123)
        b = torch.empty(5, 5)
        b.randint_(0, 100)

        self.assertTrue(torch.equal(a, b), "randint_ should be reproducible with same global seed")

    def test_randint_inplace_with_generator(self):
        import torch
        g1 = torch.Generator()
        g1.manual_seed(99)
        a = torch.empty(4, 4)
        a.randint_(0, 50, generator=g1)

        g2 = torch.Generator()
        g2.manual_seed(99)
        b = torch.empty(4, 4)
        b.randint_(0, 50, generator=g2)

        self.assertTrue(torch.equal(a, b), "randint_ with same generator seed should be reproducible")

    # ------------------------------------------------------------------
    # Step 3: tensor.random_() method
    # ------------------------------------------------------------------
    def test_random_inplace_reproducible(self):
        import torch
        torch.manual_seed(77)
        a = torch.empty(3, 3)
        a.random_(0, 100)

        torch.manual_seed(77)
        b = torch.empty(3, 3)
        b.random_(0, 100)

        self.assertTrue(torch.equal(a, b), "random_ should be reproducible with same global seed")

    def test_random_inplace_with_generator(self):
        import torch
        g1 = torch.Generator()
        g1.manual_seed(55)
        a = torch.empty(3, 3)
        a.random_(0, 100, generator=g1)

        g2 = torch.Generator()
        g2.manual_seed(55)
        b = torch.empty(3, 3)
        b.random_(0, 100, generator=g2)

        self.assertTrue(torch.equal(a, b), "random_ with same generator seed should be reproducible")

    # ------------------------------------------------------------------
    # Step 4: torch.normal() functional form
    # ------------------------------------------------------------------
    def test_normal_functional_with_size(self):
        import torch
        result = torch.normal(0.0, 1.0, size=(3, 4))
        self.assertEqual(result.shape, (3, 4))

    def test_normal_functional_with_generator(self):
        import torch
        g1 = torch.Generator()
        g1.manual_seed(42)
        a = torch.normal(0.0, 1.0, size=(5, 5), generator=g1)

        g2 = torch.Generator()
        g2.manual_seed(42)
        b = torch.normal(0.0, 1.0, size=(5, 5), generator=g2)

        self.assertTrue(torch.allclose(a, b), "torch.normal with same generator seed should be reproducible")

    # ------------------------------------------------------------------
    # Step 7: nn.init.orthogonal_ and nn.init.sparse_ with generator
    # ------------------------------------------------------------------
    def test_orthogonal_init_with_generator(self):
        import torch
        g1 = torch.Generator()
        g1.manual_seed(42)
        a = torch.empty(4, 4)
        torch.nn.init.orthogonal_(a, generator=g1)

        g2 = torch.Generator()
        g2.manual_seed(42)
        b = torch.empty(4, 4)
        torch.nn.init.orthogonal_(b, generator=g2)

        self.assertTrue(torch.allclose(a, b), "orthogonal_ with same generator seed should be reproducible")

    def test_sparse_init_with_generator(self):
        import torch
        g1 = torch.Generator()
        g1.manual_seed(42)
        a = torch.empty(4, 4)
        torch.nn.init.sparse_(a, sparsity=0.5, generator=g1)

        g2 = torch.Generator()
        g2.manual_seed(42)
        b = torch.empty(4, 4)
        torch.nn.init.sparse_(b, sparsity=0.5, generator=g2)

        self.assertTrue(torch.allclose(a, b), "sparse_ with same generator seed should be reproducible")


if __name__ == "__main__":
    unittest.main()
