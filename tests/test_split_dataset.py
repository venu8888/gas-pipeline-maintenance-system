import tempfile
import unittest
from pathlib import Path

from training.split_dataset import create_splits, split_counts


class DatasetSplitTests(unittest.TestCase):
    def test_split_counts_are_exact_for_project_dataset_size(self):
        self.assertEqual(
            split_counts(22120, {'train': 0.7, 'val': 0.2, 'test': 0.1}),
            {'train': 15484, 'val': 4424, 'test': 2212},
        )

    def test_manifests_are_paired_disjoint_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'data'
            images = root / 'images' / 'images' / 'train'
            labels = root / 'labels' / 'labels' / 'train'
            output = root / 'splits'
            images.mkdir(parents=True)
            labels.mkdir(parents=True)

            for index in range(10):
                (images / f'{index:06d}.jpg').write_bytes(b'image')
                (labels / f'{index:06d}.txt').write_text('0 0.5 0.5 0.2 0.2\n', encoding='utf-8')

            first = create_splits(root, output, seed=7)
            contents = {name: (output / f'{name}.txt').read_text(encoding='utf-8') for name in first['counts']}
            second = create_splits(root, output, seed=7)

            self.assertEqual(first['counts'], {'train': 7, 'val': 2, 'test': 1})
            self.assertEqual(first, second)
            self.assertEqual(contents, {name: (output / f'{name}.txt').read_text(encoding='utf-8') for name in first['counts']})
            listed = [line for content in contents.values() for line in content.splitlines()]
            self.assertEqual(len(listed), len(set(listed)))
            self.assertEqual(len(listed), 10)


if __name__ == '__main__':
    unittest.main()
