import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h3run as H

SESSION = 'h3skill-test'
BATCH = 'f' * 32
ENDPOINT = 'runtime-1'


class BatchTests(unittest.TestCase):
    def test_bad_download_cannot_overwrite_existing_file(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d)
            old = dest / 'shot.png'
            old.write_bytes(b'old')
            row = dict(file='shot.png', bytes=3, sha256=hashlib.sha256(b'new').hexdigest())
            def transfer(*args, **kw):
                Path(args[-1]).write_bytes(b'bad')
            with patch.object(H, 'colab', side_effect=transfer):
                self.assertRaises(RuntimeError, H.download_result, SESSION, BATCH, 'shot', row, dest)
            self.assertEqual(old.read_bytes(), b'old')
            self.assertEqual(list(dest.iterdir()), [old])

    def test_endpoint_still_visible_blocks_release_even_after_success_exit(self):
        with tempfile.TemporaryDirectory() as d, patch.object(H, 'KEEP', Path(d)):
            H.set_keep(SESSION, None, endpoint=ENDPOINT)
            rows = [([(SESSION, ENDPOINT, 'L4')], ''), ([('?', ENDPOINT, 'L4')], '')]
            with patch.object(H, 'runtime_rows', side_effect=rows), patch.object(H, 'colab') as cmd:
                self.assertRaises(RuntimeError, H.release, SESSION, ENDPOINT)
                cmd.assert_called_once()
            self.assertTrue(H.keep_file(SESSION).exists())

    def test_release_accepts_confirmed_absence_after_transport_error(self):
        with tempfile.TemporaryDirectory() as d, patch.object(H, 'KEEP', Path(d)):
            H.set_keep(SESSION, None, endpoint=ENDPOINT)
            rows = [([(SESSION, ENDPOINT, 'L4')], ''), ([], '')]
            with patch.object(H, 'runtime_rows', side_effect=rows), patch.object(H, 'colab', side_effect=RuntimeError) as cmd:
                H.release(SESSION, ENDPOINT)
                cmd.assert_called_once()
            self.assertFalse(H.keep_file(SESSION).exists())

    def test_poll_rejects_other_batch_status(self):
        with patch.object(H, 'remote_exec', return_value='H3STATUS {"batch_id":"other","done":true}'):
            self.assertRaises(RuntimeError, H.poll, SESSION, BATCH)

    def test_fallback_preserves_requested_specs(self):
        args = SimpleNamespace(gpu='AUTO', idle_minutes=0)
        m = {'jobs': [{'type': 'video', 'resolution': '720P'}, {'type': 'video'}]}
        with patch.object(H, 'runtime_rows', return_value=([], '')), patch.object(H, 'colab', side_effect=[RuntimeError('Allocation refused (precondition failed)'), 'READY']):
            _, gpu = H.create_runtime('A100', auto=True)
        H.apply_defaults(m, args, gpu)
        self.assertEqual(gpu, 'L4')
        self.assertEqual(args.idle_minutes, 0)
        self.assertEqual([j['resolution'] for j in m['jobs']], ['720P', '480P'])

    def test_transport_failure_does_not_provision_second_gpu(self):
        with patch.object(H, 'runtime_rows', return_value=([], '')), patch.object(H, 'colab', side_effect=RuntimeError('timed out')) as cmd:
            self.assertRaises(RuntimeError, H.create_runtime, 'A100', True)
            cmd.assert_called_once()

    def test_new_run_refuses_unnamed_runtime(self):
        rows = [('?', 'runtime-orphan', 'L4')]
        with patch.object(H, 'runtime_rows', return_value=(rows, '')) as listed, patch.object(H, 'colab') as cmd:
            self.assertRaises(RuntimeError, H.create_runtime, 'L4', False)
            listed.assert_called_once()
            cmd.assert_not_called()

    def test_lost_start_reply_keeps_accepted_batch(self):
        record = {'session': SESSION, 'endpoint': ENDPOINT, 'batch_id': BATCH}
        accepted = f'H3STATUS {{"batch_id":"{BATCH}","stage":"starting","done":false}}'
        with patch.object(H, 'remote_exec', side_effect=[RuntimeError('timed out'), accepted]) as remote, \
                patch.object(H, 'cleanup_batch') as cleanup:
            reply = H.start_batch(record)
            self.assertEqual(reply['batch_id'], BATCH)
            cleanup.assert_not_called()
            self.assertEqual(remote.call_count, 2)

    def test_watchdog_keeps_runtime_when_busy_state_is_unknown(self):
        with tempfile.TemporaryDirectory() as d, patch.object(H, 'KEEP', Path(d)):
            H.set_keep(SESSION, 1, endpoint=ENDPOINT)
            rows = [([(SESSION, ENDPOINT, 'L4')], '')]
            with patch.object(H, 'runtime_rows', side_effect=rows), \
                    patch.object(H, 'runtime_busy', side_effect=RuntimeError('查詢失敗')), \
                    patch.object(H, 'release') as release, patch.object(H.time, 'time', return_value=2):
                H.watchdog_once(SESSION)
                release.assert_not_called()
            self.assertTrue(H.keep_file(SESSION).exists())


if __name__ == '__main__':
    unittest.main()
