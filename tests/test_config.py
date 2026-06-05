from __future__ import annotations

import plistlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from option_stream import config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        config.load_app_config.cache_clear()

    def tearDown(self):
        config.load_app_config.cache_clear()

    def _write_plist(self, payload):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        plist_path = Path(tempdir.name) / "config.plist"
        with plist_path.open("wb") as handle:
            plistlib.dump(payload, handle)
        return plist_path

    def test_load_app_config_reads_override_path(self):
        plist_path = self._write_plist(
            {
                "fred_api_key": "abc123",
                "illiquid_equity": {"a": 1, "b": 2},
            }
        )

        with mock.patch.object(config, "CONFIG_PLIST_PATH", plist_path):
            loaded = config.load_app_config()

        self.assertEqual(loaded["fred_api_key"], "abc123")
        self.assertEqual(loaded["illiquid_equity"], {"a": 1, "b": 2})

    def test_config_key_missing_key_raises_clear_error(self):
        plist_path = self._write_plist(
            {
                "fred_api_key": "abc123",
                "illiquid_equity": {"a": 1, "b": 2},
            }
        )

        with mock.patch.object(config, "CONFIG_PLIST_PATH", plist_path):
            with self.assertRaises(KeyError) as ctx:
                config.config_key("missing_key")

        self.assertIn("missing_key", str(ctx.exception))
        self.assertIn("Available keys", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
