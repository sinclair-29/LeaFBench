import unittest

try:
    import torch

    from fingerprint.reef.compute_cka import CKA
except ModuleNotFoundError:
    torch = None
    CKA = None


@unittest.skipIf(torch is None, "torch is not installed in the local test interpreter")
class REEFProtocolTests(unittest.TestCase):
    def test_standardize_features_matches_released_reef_preprocessing(self):
        values = torch.tensor(
            [[1.0, 10.0], [2.0, 12.0], [4.0, 16.0], [8.0, 22.0]]
        )
        standardized = CKA.standardize_features(values)
        self.assertTrue(
            torch.allclose(standardized.mean(dim=0), torch.zeros(2), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(
                standardized.std(dim=0, unbiased=True), torch.ones(2), atol=1e-6
            )
        )

    def test_linear_cka_is_one_for_feature_rescaling(self):
        values = torch.tensor(
            [[1.0, 5.0], [2.0, 9.0], [4.0, 8.0], [8.0, 20.0]]
        )
        scaled = values * torch.tensor([7.0, 0.25]) + torch.tensor([3.0, -4.0])
        score = CKA(torch.device("cpu")).linear_CKA(values, scaled)
        self.assertAlmostEqual(float(score), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
