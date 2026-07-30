from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data.loading import list_sources, load_source


class DataLoadingTests(unittest.TestCase):
    def test_load_source_supports_nested_fetch_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "outbrain"
            source_dir.mkdir()
            (source_dir / "outbrain-unsafe.json").write_text(
                json.dumps([{"headers": '{"h":"u"}', "request": '{"r":"u"}'}]),
                encoding="utf-8",
            )
            (source_dir / "outbrain-bot.json").write_text(
                json.dumps([{"headers": '{"h":"b"}', "request": '{"r":"b"}'}]),
                encoding="utf-8",
            )

            headers, requests, labels = load_source("outbrain", directory=tmp)

            self.assertEqual(['{"h":"u"}', '{"h":"b"}'], headers)
            self.assertEqual(['{"r":"u"}', '{"r":"b"}'], requests)
            self.assertEqual([0, 1], labels.tolist())
            self.assertEqual(["outbrain"], list_sources(directory=tmp))


if __name__ == "__main__":
    unittest.main()
