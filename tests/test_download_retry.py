import os
import unittest
from unittest.mock import Mock, patch

from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ.setdefault("MODELS_DIR", os.path.join(REPO_ROOT, "test-models"))
os.environ.setdefault("DATA_DIR", os.path.join(REPO_ROOT, "test-data"))
os.environ.setdefault("LOGS_DIR", os.path.join(REPO_ROOT, "test-logs"))

import api.downloads as downloads_api
import core.monitoring as monitoring
from core.state import downloads, downloads_lock


class DownloadRetryTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(downloads_api.bp)
        self.client = self.app.test_client()
        with downloads_lock:
            self._saved_downloads = {dl_id: dict(dl) for dl_id, dl in downloads.items()}
            downloads.clear()

    def tearDown(self):
        with downloads_lock:
            downloads.clear()
            downloads.update(self._saved_downloads)

    @patch("api.downloads.save_state")
    @patch("api.downloads._restart_existing_download")
    def test_retry_failed_download_restarts_process(self, restart_mock, _save_state_mock):
        proc = Mock(pid=4321)
        log_fh = Mock()
        restart_mock.return_value = (proc, log_fh, "/tmp/dl-test.log", None)

        with downloads_lock:
            downloads["dl-test"] = {
                "id": "dl-test",
                "repo_id": "org/model",
                "filename": "model.gguf",
                "dest_path": "/models/model",
                "status": "failed",
                "pid": 0,
                "log_file": "/tmp/old.log",
                "started_at": 1,
                "per_model_speed_limit_mbps": 0,
                "retry_attempts": 2,
                "_process": None,
                "_log_fh": None,
            }

        resp = self.client.post("/api/downloads/dl-test/retry")

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "downloading")
        self.assertEqual(data["pid"], 4321)
        self.assertEqual(data["retry_attempts"], 0)
        restart_mock.assert_called_once()

    @patch("api.downloads.save_state")
    @patch("api.downloads.cleanup_download_dir")
    def test_remove_failed_download_cleans_partial_dir(self, cleanup_mock, _save_state_mock):
        with downloads_lock:
            downloads["dl-test"] = {
                "id": "dl-test",
                "repo_id": "org/model",
                "filename": "model.gguf",
                "dest_path": "/models/model",
                "status": "failed",
                "pid": 0,
                "log_file": "/tmp/old.log",
                "started_at": 1,
                "_process": None,
                "_log_fh": None,
            }

        resp = self.client.delete("/api/downloads/dl-test/remove")

        self.assertEqual(resp.status_code, 200)
        cleanup_mock.assert_called_once_with("/models/model")

    @patch("core.monitoring.save_state")
    @patch("core.monitoring.kill_instance_process")
    @patch("core.monitoring._get_failed_download_retry_settings", return_value=(True, 3))
    @patch("api.downloads._restart_existing_download")
    def test_auto_retry_failed_download_restarts_process(
        self,
        restart_mock,
        _settings_mock,
        kill_mock,
        save_state_mock,
    ):
        proc = Mock(pid=9876)
        log_fh = Mock()
        restart_mock.return_value = (proc, log_fh, "/tmp/dl-test.log", None)

        with downloads_lock:
            downloads["dl-test"] = {
                "id": "dl-test",
                "repo_id": "org/model",
                "filename": "model.gguf",
                "dest_path": "/models/model",
                "status": "downloading",
                "pid": 1234,
                "log_file": "/tmp/old.log",
                "started_at": 1,
                "per_model_speed_limit_mbps": 0,
                "retry_attempts": 1,
                "_process": Mock(),
                "_log_fh": Mock(),
            }

        monitoring._handle_download_exit("dl-test", 1)

        with downloads_lock:
            dl = downloads["dl-test"]
            self.assertEqual(dl["status"], "downloading")
            self.assertEqual(dl["pid"], 9876)
            self.assertEqual(dl["retry_attempts"], 2)

        restart_mock.assert_called_once()
        kill_mock.assert_called_once()
        save_state_mock.assert_called_once()

    @patch("core.monitoring.save_state")
    @patch("core.monitoring.kill_instance_process")
    @patch("core.monitoring._get_failed_download_retry_settings", return_value=(True, 2))
    def test_auto_retry_failed_download_stops_at_retry_limit(
        self,
        _settings_mock,
        kill_mock,
        save_state_mock,
    ):
        with downloads_lock:
            downloads["dl-test"] = {
                "id": "dl-test",
                "repo_id": "org/model",
                "filename": "model.gguf",
                "dest_path": "/models/model",
                "status": "downloading",
                "pid": 1234,
                "log_file": "/tmp/old.log",
                "started_at": 1,
                "per_model_speed_limit_mbps": 0,
                "retry_attempts": 2,
                "_process": Mock(),
                "_log_fh": Mock(),
            }

        monitoring._handle_download_exit("dl-test", 1)

        with downloads_lock:
            dl = downloads["dl-test"]
            self.assertEqual(dl["status"], "failed")
            self.assertEqual(dl["pid"], 0)
            self.assertEqual(dl["retry_attempts"], 2)

        kill_mock.assert_called_once()
        save_state_mock.assert_called_once()


class DownloadWatchSurvivesStorageBlipTests(unittest.TestCase):
    """core/monitoring.py issue #122: a storage read failing exactly as the
    downloader exits must cost the poller one tick, not the thread."""

    def setUp(self):
        with downloads_lock:
            self._saved_downloads = {dl_id: dict(dl) for dl_id, dl in downloads.items()}
            downloads.clear()

    def tearDown(self):
        with downloads_lock:
            downloads.clear()
            downloads.update(self._saved_downloads)

    def _add_exited_download(self):
        proc = Mock()
        proc.poll.return_value = 1
        with downloads_lock:
            downloads["dl-test"] = {
                "id": "dl-test",
                "repo_id": "org/model",
                "filename": "model.gguf",
                "dest_path": "/models/model",
                "status": "downloading",
                "pid": 1234,
                "log_file": "/tmp/dl-test.log",
                "started_at": 1,
                "per_model_speed_limit_mbps": 0,
                "retry_attempts": 0,
                "_process": proc,
                "_log_fh": Mock(),
            }
        return proc

    @patch("core.monitoring._handle_download_exit", side_effect=ConnectionError("db unreachable"))
    def test_watch_downloads_survives_handler_failure(self, _handle_mock):
        self._add_exited_download()
        # Must not raise: an exception escaping here kills the poller thread
        # and leaves the download stuck "downloading" forever.
        monitoring._watch_downloads()

        with downloads_lock:
            self.assertEqual(downloads["dl-test"]["status"], "downloading")

    @patch("core.monitoring.save_state")
    @patch("core.monitoring.kill_instance_process")
    @patch("core.monitoring._get_failed_download_retry_settings")
    @patch("api.downloads._restart_existing_download")
    def test_next_tick_retries_exit_handling_once_storage_recovers(
        self, restart_mock, settings_mock, _kill_mock, _save_state_mock
    ):
        settings_mock.side_effect = [
            ConnectionError("db unreachable"),  # blip: first tick fails
            (True, 3),                          # recovered: auto-retry proceeds
        ]
        restart_mock.return_value = (Mock(pid=9876), Mock(), "/tmp/dl-test.log", None)
        self._add_exited_download()

        monitoring._watch_downloads()   # swallows the storage error
        monitoring._watch_downloads()   # same exited process detected again

        with downloads_lock:
            dl = downloads["dl-test"]
            self.assertEqual(dl["status"], "downloading")
            self.assertEqual(dl["retry_attempts"], 1)
        restart_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
