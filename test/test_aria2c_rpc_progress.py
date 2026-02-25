#!/usr/bin/env python3

# Allow direct execution
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import json
from unittest.mock import MagicMock, patch

from test.helper import FakeYDL
from yt_dlp.downloader.external import Aria2cFD


def _make_rpc_response(result, request_id=None):
    """Build a JSON-RPC response body that aria2c_rpc() will accept."""
    resp = {'jsonrpc': '2.0', 'id': request_id, 'result': result}
    return json.dumps(resp).encode()


class _FakeRPCResponse:
    """Context manager mimicking urlopen() return value."""

    def __init__(self, data):
        self._stream = io.BytesIO(data)

    def __enter__(self):
        return self._stream

    def __exit__(self, *args):
        pass


class TestAria2cRPC(unittest.TestCase):
    """Tests for Aria2cFD.aria2c_rpc()"""

    def _make_downloader(self):
        ydl = FakeYDL()
        return Aria2cFD(ydl, {}), ydl

    def test_rpc_sends_correct_request(self):
        dl, ydl = self._make_downloader()
        captured = {}

        def fake_urlopen(request):
            body = json.loads(request.data)
            captured.update(body)
            return _FakeRPCResponse(
                _make_rpc_response({'version': '1.37.0'}, body['id']))

        ydl.urlopen = fake_urlopen
        result = dl.aria2c_rpc(19190, 'test-secret', 'aria2.getVersion')

        self.assertEqual(captured['method'], 'aria2.getVersion')
        self.assertEqual(captured['params'], ['token:test-secret'])
        self.assertEqual(result, {'version': '1.37.0'})

    def test_rpc_passes_extra_params(self):
        dl, ydl = self._make_downloader()
        captured = {}

        def fake_urlopen(request):
            body = json.loads(request.data)
            captured.update(body)
            return _FakeRPCResponse(_make_rpc_response([], body['id']))

        ydl.urlopen = fake_urlopen
        dl.aria2c_rpc(19190, 'secret', 'aria2.tellStopped', [0, 10])

        self.assertEqual(captured['params'], ['token:secret', 0, 10])

    def test_rpc_raises_on_network_error(self):
        dl, ydl = self._make_downloader()
        ydl.urlopen = MagicMock(side_effect=OSError('Connection refused'))

        with self.assertRaises(ConnectionError):
            dl.aria2c_rpc(19190, 'secret', 'aria2.getVersion')

    def test_rpc_raises_on_invalid_json(self):
        dl, ydl = self._make_downloader()

        def fake_urlopen(request):
            return _FakeRPCResponse(b'not json')

        ydl.urlopen = fake_urlopen

        with self.assertRaises(ConnectionError):
            dl.aria2c_rpc(19190, 'secret', 'aria2.getVersion')

    def test_rpc_raises_on_id_mismatch(self):
        dl, ydl = self._make_downloader()

        def fake_urlopen(request):
            return _FakeRPCResponse(_make_rpc_response('ok', 'wrong-id'))

        ydl.urlopen = fake_urlopen

        with self.assertRaises(ConnectionError):
            dl.aria2c_rpc(19190, 'secret', 'aria2.getVersion')


class TestAria2cCallDownloader(unittest.TestCase):
    """Tests for _call_downloader enabling RPC."""

    def test_rpc_enabled_by_default(self):
        with FakeYDL() as ydl:
            dl = Aria2cFD(ydl, {})
            info_dict = {'url': 'http://example.com/video.mp4'}
            with patch('yt_dlp.downloader.external.ExternalFD._call_downloader', return_value=0):
                dl._call_downloader('test', info_dict)
            self.assertIn('__rpc', info_dict)
            self.assertIn('port', info_dict['__rpc'])
            self.assertIn('secret', info_dict['__rpc'])

    def test_rpc_disabled_via_compat_opts(self):
        with FakeYDL({'compat_opts': {'no-external-downloader-progress'}}) as ydl:
            dl = Aria2cFD(ydl, {'compat_opts': {'no-external-downloader-progress'}})
            info_dict = {'url': 'http://example.com/video.mp4'}
            with patch('yt_dlp.downloader.external.ExternalFD._call_downloader', return_value=0):
                dl._call_downloader('test', info_dict)
            self.assertNotIn('__rpc', info_dict)


def _make_mock_proc(poll_sequence, returncode=0):
    """Create a mock Popen context manager with a controlled poll() sequence."""
    mock_proc = MagicMock()
    mock_proc.__enter__ = MagicMock(return_value=mock_proc)
    mock_proc.__exit__ = MagicMock(return_value=False)
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.read = MagicMock(return_value='')
    mock_proc.returncode = returncode
    mock_proc.poll = MagicMock(side_effect=poll_sequence)
    mock_proc.wait = MagicMock(return_value=returncode)
    return mock_proc


class TestAria2cRPCProgress(unittest.TestCase):
    """Tests for the RPC progress polling loop in _call_process()."""

    def test_progress_hooks_fired_during_download(self):
        with FakeYDL() as ydl:
            dl = Aria2cFD(ydl, {})
            progress_statuses = []
            dl._hook_progress = lambda status, info: progress_statuses.append(dict(status))

            info_dict = {
                'url': 'http://example.com/video.mp4',
                '_filename': 'test.mp4',
                '__rpc': {'port': 19190, 'secret': 'test'},
            }

            call_count = [0]

            def fake_urlopen(request):
                body = json.loads(request.data)
                method = body['method']
                call_count[0] += 1

                if method == 'aria2.getVersion':
                    result = {'version': '1.37.0'}
                elif method == 'aria2.tellActive':
                    if call_count[0] < 8:
                        result = [{'completedLength': '500000', 'totalLength': '1000000',
                                   'downloadSpeed': '100000'}]
                    else:
                        result = []
                elif method == 'aria2.tellStopped':
                    if call_count[0] < 8:
                        result = []
                    else:
                        result = [{'totalLength': '1000000', 'completedLength': '1000000',
                                   'downloadSpeed': '0'}]
                elif method == 'aria2.shutdown':
                    result = 'OK'
                else:
                    result = None

                return _FakeRPCResponse(_make_rpc_response(result, body['id']))

            ydl.urlopen = fake_urlopen

            # poll: None during download, then None a few more times (loop continues),
            # then download completes via active=[] and completed=[...]
            mock_proc = _make_mock_proc([None] * 10 + [0])

            with patch('yt_dlp.downloader.external.Popen', return_value=mock_proc):
                with patch('time.sleep'):
                    _, _, retval = dl._call_process(['aria2c', '--enable-rpc'], info_dict)

            self.assertEqual(retval, 0)
            self.assertTrue(len(progress_statuses) >= 2)
            self.assertEqual(progress_statuses[0]['status'], 'downloading')
            self.assertEqual(progress_statuses[0]['downloaded_bytes'], 0)
            self.assertTrue(any(s['downloaded_bytes'] > 0 for s in progress_statuses[1:]))
            self.assertTrue(any(s.get('speed', 0) > 0 for s in progress_statuses[1:]))

    def test_fallback_when_rpc_not_ready(self):
        with FakeYDL() as ydl:
            dl = Aria2cFD(ydl, {})
            progress_statuses = []
            dl._hook_progress = lambda status, info: progress_statuses.append(dict(status))
            messages = []
            dl.to_screen = messages.append

            info_dict = {
                'url': 'http://example.com/video.mp4',
                '_filename': 'test.mp4',
                '__rpc': {'port': 19190, 'secret': 'test'},
            }

            ydl.urlopen = MagicMock(side_effect=OSError('Connection refused'))
            mock_proc = _make_mock_proc([None] * 20)

            with patch('yt_dlp.downloader.external.Popen', return_value=mock_proc):
                with patch('time.sleep'):
                    _, _, retval = dl._call_process(['aria2c', '--enable-rpc'], info_dict)

            self.assertEqual(retval, 0)
            self.assertEqual(len(progress_statuses), 0)
            self.assertTrue(any('not available' in m for m in messages))

    def test_fallback_when_process_exits_during_startup(self):
        with FakeYDL() as ydl:
            dl = Aria2cFD(ydl, {})
            progress_statuses = []
            dl._hook_progress = lambda status, info: progress_statuses.append(dict(status))
            messages = []
            dl.to_screen = messages.append

            info_dict = {
                'url': 'http://example.com/video.mp4',
                '_filename': 'test.mp4',
                '__rpc': {'port': 19190, 'secret': 'test'},
            }

            ydl.urlopen = MagicMock(side_effect=OSError('Connection refused'))
            # Process exits immediately (poll returns 1 on first call)
            mock_proc = _make_mock_proc([1], returncode=1)

            with patch('yt_dlp.downloader.external.Popen', return_value=mock_proc):
                with patch('time.sleep'):
                    _, _, retval = dl._call_process(['aria2c', '--enable-rpc'], info_dict)

            self.assertEqual(retval, 1)
            self.assertEqual(len(progress_statuses), 0)

    def test_rpc_failure_mid_download_graceful(self):
        with FakeYDL() as ydl:
            dl = Aria2cFD(ydl, {})
            progress_statuses = []
            dl._hook_progress = lambda status, info: progress_statuses.append(dict(status))
            messages = []
            dl.to_screen = messages.append

            info_dict = {
                'url': 'http://example.com/video.mp4',
                '_filename': 'test.mp4',
                '__rpc': {'port': 19190, 'secret': 'test'},
            }

            rpc_calls = [0]

            def fake_urlopen(request):
                body = json.loads(request.data)
                rpc_calls[0] += 1
                if rpc_calls[0] == 1:
                    return _FakeRPCResponse(
                        _make_rpc_response({'version': '1.37.0'}, body['id']))
                raise OSError('Connection reset')

            ydl.urlopen = fake_urlopen
            mock_proc = _make_mock_proc([None] * 5)

            with patch('yt_dlp.downloader.external.Popen', return_value=mock_proc):
                with patch('time.sleep'):
                    _, _, retval = dl._call_process(['aria2c', '--enable-rpc'], info_dict)

            self.assertEqual(retval, 0)
            # Initial hook fires, then RPC fails on first poll
            self.assertEqual(len(progress_statuses), 1)
            self.assertTrue(any('connection lost' in m.lower() for m in messages))

    def test_eta_and_total_bytes_computed(self):
        with FakeYDL() as ydl:
            dl = Aria2cFD(ydl, {})
            progress_statuses = []
            dl._hook_progress = lambda status, info: progress_statuses.append(dict(status))

            info_dict = {
                'url': 'http://example.com/video.mp4',
                '_filename': 'test.mp4',
                '__rpc': {'port': 19190, 'secret': 'test'},
            }

            call_count = [0]

            def fake_urlopen(request):
                body = json.loads(request.data)
                method = body['method']
                call_count[0] += 1

                if method == 'aria2.getVersion':
                    result = {'version': '1.37.0'}
                elif method == 'aria2.tellActive':
                    if call_count[0] < 6:
                        result = [{'completedLength': '250000', 'totalLength': '1000000',
                                   'downloadSpeed': '50000'}]
                    else:
                        result = []
                elif method == 'aria2.tellStopped':
                    if call_count[0] < 6:
                        result = []
                    else:
                        result = [{'totalLength': '1000000', 'completedLength': '1000000',
                                   'downloadSpeed': '0'}]
                elif method == 'aria2.shutdown':
                    result = 'OK'
                else:
                    result = None

                return _FakeRPCResponse(_make_rpc_response(result, body['id']))

            ydl.urlopen = fake_urlopen
            mock_proc = _make_mock_proc([None] * 10 + [0])

            with patch('yt_dlp.downloader.external.Popen', return_value=mock_proc):
                with patch('time.sleep'):
                    dl._call_process(['aria2c', '--enable-rpc'], info_dict)

            # Find a status with active download data
            active_statuses = [s for s in progress_statuses if s['downloaded_bytes'] > 0]
            self.assertTrue(len(active_statuses) > 0)
            s = active_statuses[0]
            self.assertEqual(s['downloaded_bytes'], 250000)
            self.assertEqual(s['speed'], 50000.0)
            self.assertIsNotNone(s.get('total_bytes'))
            self.assertIsNotNone(s.get('eta'))


    def test_fragmented_download_progress(self):
        with FakeYDL() as ydl:
            dl = Aria2cFD(ydl, {})
            progress_statuses = []
            dl._hook_progress = lambda status, info: progress_statuses.append(dict(status))

            info_dict = {
                'url': 'http://example.com/video.mp4',
                '_filename': 'test.mp4',
                '__rpc': {'port': 19190, 'secret': 'test'},
                'fragments': [
                    {'url': 'http://example.com/frag0'},
                    {'url': 'http://example.com/frag1'},
                    {'url': 'http://example.com/frag2'},
                ],
            }

            call_count = [0]

            def fake_urlopen(request):
                body = json.loads(request.data)
                method = body['method']
                call_count[0] += 1

                if method == 'aria2.getVersion':
                    result = {'version': '1.37.0'}
                elif method == 'aria2.tellActive':
                    if call_count[0] < 8:
                        result = [{'completedLength': '200000', 'totalLength': '500000',
                                   'downloadSpeed': '100000'}]
                    else:
                        result = []
                elif method == 'aria2.tellStopped':
                    if call_count[0] < 8:
                        result = [{'totalLength': '500000', 'completedLength': '500000',
                                   'downloadSpeed': '0'}]
                    else:
                        result = [
                            {'totalLength': '500000', 'completedLength': '500000', 'downloadSpeed': '0'},
                            {'totalLength': '500000', 'completedLength': '500000', 'downloadSpeed': '0'},
                            {'totalLength': '500000', 'completedLength': '500000', 'downloadSpeed': '0'},
                        ]
                elif method == 'aria2.shutdown':
                    result = 'OK'
                else:
                    result = None

                return _FakeRPCResponse(_make_rpc_response(result, body['id']))

            ydl.urlopen = fake_urlopen
            mock_proc = _make_mock_proc([None] * 10 + [0])

            with patch('yt_dlp.downloader.external.Popen', return_value=mock_proc):
                with patch('time.sleep'):
                    _, _, retval = dl._call_process(['aria2c', '--enable-rpc'], info_dict)

            self.assertEqual(retval, 0)
            self.assertTrue(len(progress_statuses) >= 2)
            # Fragmented: total_bytes should be None, fragment_count should be 3
            self.assertEqual(progress_statuses[0]['fragment_count'], 3)
            self.assertEqual(progress_statuses[0]['fragment_index'], 0)
            # During download, total_bytes should be None (fragmented mode)
            active_statuses = [s for s in progress_statuses if s['downloaded_bytes'] > 0]
            self.assertTrue(len(active_statuses) > 0)
            self.assertIsNone(active_statuses[0]['total_bytes'])
            self.assertIsNotNone(active_statuses[0].get('total_bytes_estimate'))


if __name__ == '__main__':
    unittest.main()
