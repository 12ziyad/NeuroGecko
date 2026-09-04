import hashlib
import tempfile
import unittest
from pathlib import Path

from utils.sync_checkpoints import EXPECTED_FILES, validate_manifest, verify_bundle


class CheckpointSyncTests(unittest.TestCase):
    def make_manifest(self):
        return dict(schema_version=1, complete=True, num_timesteps=10,
                    files={name: dict(bytes=3, sha256=hashlib.sha256(b'abc').hexdigest())
                           for name in EXPECTED_FILES})

    def test_complete_bundle_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in EXPECTED_FILES:
                (root / name).write_bytes(b'abc')
            verify_bundle(root, self.make_manifest())
            (root / 'model.zip').write_bytes(b'abd')
            with self.assertRaises(ValueError):
                verify_bundle(root, self.make_manifest())

    def test_rejects_path_escape(self):
        manifest = self.make_manifest()
        manifest['files']['../escape'] = manifest['files'].pop('model.zip')
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_rejects_incomplete_snapshot(self):
        manifest = self.make_manifest()
        manifest['complete'] = False
        with self.assertRaises(ValueError):
            validate_manifest(manifest)


if __name__ == '__main__':
    unittest.main()
