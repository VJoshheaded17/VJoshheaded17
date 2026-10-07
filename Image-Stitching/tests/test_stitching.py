import unittest
from unittest.mock import patch
import torch
import stitching as s


class StitchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_identity_canvas_preserves_last_row_column(self):
        img = torch.rand(3, 12, 17)
        size, transforms = s._canvas([img], [torch.eye(3)])
        self.assertEqual(size, (12, 17))
        result = s._render([img], transforms)
        expected = (img * 255).round().byte()
        self.assertTrue(torch.equal(result, expected))

    def test_valid_black_pixels_participate_in_average(self):
        black, white = torch.zeros(3, 12, 16), torch.ones(3, 12, 16)
        result = s._render([black, white], [torch.eye(3), torch.eye(3)], blend='average')
        self.assertTrue(torch.all(result == 128))

    def test_translation_direction(self):
        # Image B's x=0 lands at x=7 in reference A, not x=-7.
        a, b = torch.zeros(3, 10, 12), torch.ones(3, 10, 12)
        h = torch.eye(3);h[0, 2] = 7
        result = s._render([a, b], [torch.eye(3), h], blend='average')
        self.assertEqual(tuple(result.shape), (3, 10, 19))
        self.assertTrue(torch.all(result[:, :, :7] == 0))
        self.assertTrue(torch.all(result[:, :, 12:] == 255))

    def test_negative_translation_and_overlap(self):
        a = torch.ones(3, 20, 30)
        h = torch.eye(3);h[0, 2] = -10
        size, _ = s._canvas([a, a], [torch.eye(3), h])
        self.assertEqual(size, (20, 40))
        self.assertAlmostEqual(s._pair_overlap(a, a, h), 2/3, places=5)

    def test_non_overlap(self):
        a = torch.ones(3, 20, 30)
        h = torch.eye(3);h[0, 2] = 40
        self.assertEqual(s._pair_overlap(a, a, h), 0)

    def test_indirect_graph_composition(self):
        i = torch.eye(3)
        one = i.clone();one[0, 2] = 10
        two = i.clone();two[0, 2] = 20
        # Image 0 is central by degree tie/index. Image 3 overlaps only image 2.
        graph = {0: {1: (one, 20), 2: (two, 20)},
                 1: {0: (torch.linalg.inv(one), 20)},
                 2: {0: (torch.linalg.inv(two), 20), 3: (one, 20)},
                 3: {2: (torch.linalg.inv(one), 20)}}
        transforms = s._compose_transforms(graph, {0, 1, 2, 3}, i)
        self.assertTrue(torch.allclose(transforms[3], two @ one))
        self.assertEqual(float(transforms[3][0, 2]), 30)

    def test_largest_connected_group(self):
        graph = {0: {1: None}, 1: {0: None, 2: None}, 2: {1: None}, 3: {}, 4: {5: None}, 5: {4: None}}
        self.assertEqual(s._largest_component(graph), {0, 1, 2})

    def test_validation(self):
        for images in ({}, {'a': torch.zeros(1, 12, 12)}, {'a': torch.full((3, 12, 12), 2.0)},
                       {'a': torch.full((3, 12, 12), float('nan'))}):
            with self.assertRaises(s.StitchingError):
                s._normalize_images(images)

    def test_single_image_panorama(self):
        image = torch.randint(0, 256, (3, 12, 17), dtype=torch.uint8)
        result, overlap = s.panorama({'a': image})
        self.assertTrue(torch.equal(result, image))
        self.assertEqual(overlap.tolist(), [[1]])

    def test_unsafe_projection(self):
        h = torch.eye(3);h[2] = torch.tensor([1., 0., -5.])
        with self.assertRaises(s.StitchingError):
            s.corners(20, 20, h)
        huge = torch.eye(3);huge[0, 2] = 1e8
        with self.assertRaises(s.StitchingError):
            s._canvas([torch.ones(3, 20, 20)]*2, [torch.eye(3), huge])
        self.assertFalse(s._valid_homography(torch.zeros(3, 3)))

    def test_failed_match_does_not_become_identity_edge(self):
        images = [torch.ones(3, 20, 20)]*2
        features = [(torch.empty(0, 2, 3), torch.empty(0, 128))]*2
        with patch('stitching.matching', return_value=(None, None)):
            graph, overlap = s._alignment_graph(images, features)
        self.assertEqual(graph, {0: {}, 1: {}})
        self.assertEqual(overlap.tolist(), [[1, 0], [0, 1]])

    def test_matching_direction_with_controlled_correspondences(self):
        # Exercise real descriptor matching and RANSAC with a known B->A translation.
        points_b = torch.tensor([[4.,4.],[15.,4.],[25.,4.],[4.,16.],[15.,16.],
                                 [25.,16.],[4.,28.],[15.,28.],[25.,28.],[8.,35.]])
        points_a = points_b + torch.tensor([13., 7.])
        def frames(points):
            f = torch.zeros(len(points), 2, 3)
            f[:, 0, 0] = f[:, 1, 1] = 1
            f[:, :, 2] = points
            return f
        desc = torch.eye(len(points_b))
        torch.manual_seed(17)
        h, inliers = s.matching(desc, desc, frames(points_a), frames(points_b))
        self.assertIsNotNone(h)
        projected = h @ torch.cat([points_b.T, torch.ones(1, len(points_b))], dim=0)
        projected = (projected[:2] / projected[2:]).T
        self.assertTrue(torch.allclose(projected, points_a, atol=.05))
        self.assertGreaterEqual(int(inliers.sum()), 8)


if __name__ == '__main__':
    unittest.main()
