import random
import unittest

from utils.placement import bboxes_intersect, can_place_bbox, find_position_for_object


class PlacementTests(unittest.TestCase):
    def test_bboxes_intersect_inclusive_edges(self):
        self.assertTrue(bboxes_intersect((0, 0, 10, 10), (10, 10, 20, 20)))
        self.assertFalse(bboxes_intersect((0, 0, 10, 10), (11, 0, 20, 10)))

    def test_can_place_respects_padding_and_distance(self):
        placed = [(50, 50, 60, 60)]
        self.assertFalse(can_place_bbox((1, 1, 10, 10), [], 320, 180, edge_padding=2, min_distance=2))
        self.assertFalse(can_place_bbox((62, 50, 70, 60), placed, 320, 180, edge_padding=2, min_distance=2))
        self.assertTrue(can_place_bbox((65, 50, 70, 60), placed, 320, 180, edge_padding=2, min_distance=2))

    def test_found_position_is_valid(self):
        rng = random.Random(0)
        placed = []
        for _ in range(8):
            pos = find_position_for_object(30, 30, placed, 320, 180, max_attempts=200, edge_padding=2, min_distance=2, rng=rng)
            self.assertIsNotNone(pos)
            x, y = pos
            box = (x, y, x + 29, y + 29)
            self.assertTrue(can_place_bbox(box, placed, 320, 180, 2, 2))
            placed.append(box)

    def test_object_larger_than_canvas(self):
        rng = random.Random(0)
        self.assertIsNone(find_position_for_object(400, 30, [], 320, 180, 10, 2, 2, rng))


if __name__ == "__main__":
    unittest.main()
