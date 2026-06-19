import unittest
from unittest.mock import MagicMock
import sys
import os

# Import Flask app and mock the engine before loading or immediately after
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app import app, engine

class TestChat2Endpoint(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

        # Mock the engine
        self.original_is_loaded = engine.is_loaded
        self.original_generate_stream = engine.generate_stream
        self.original_generate = engine.generate

    def tearDown(self):
        engine.is_loaded = self.original_is_loaded
        engine.generate_stream = self.original_generate_stream
        engine.generate = self.original_generate

    def test_chat2_model_not_loaded(self):
        engine.is_loaded = MagicMock(return_value=False)
        response = self.app.get('/api/chat2?quirie=Hello')
        self.assertEqual(response.status_code, 503)
        self.assertIn(b"Model is not loaded yet", response.data)

    def test_chat2_missing_param(self):
        engine.is_loaded = MagicMock(return_value=True)
        response = self.app.get('/api/chat2')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Missing parameter 'quirie'", response.data)

    def test_chat2_stream_success(self):
        engine.is_loaded = MagicMock(return_value=True)
        # Mock generator returning text chunks and one metadata dict
        def mock_generate_stream(messages, max_new_tokens=None):
            yield "Hello "
            yield "world"
            yield {"__meta__": {"tokens": 2, "elapsed_sec": 0.1, "tokens_per_sec": 20}}

        engine.generate_stream = MagicMock(side_effect=mock_generate_stream)

        response = self.app.get('/api/chat2?quirie=Hello&stream=true')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'text/plain; charset=utf-8')
        self.assertEqual(response.data, b"Hello world")
        engine.generate_stream.assert_called_once_with([{"role": "user", "content": "Hello"}])

    def test_chat2_sync_success(self):
        engine.is_loaded = MagicMock(return_value=True)
        engine.generate = MagicMock(return_value="Synchronous response")

        response = self.app.get('/api/chat2?quirie=Hello&stream=false')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'text/plain; charset=utf-8')
        self.assertEqual(response.data, b"Synchronous response")
        engine.generate.assert_called_once_with([{"role": "user", "content": "Hello"}])

    def test_chat2_post_json_success(self):
        engine.is_loaded = MagicMock(return_value=True)
        engine.generate = MagicMock(return_value="JSON response")

        response = self.app.post('/api/chat2', json={"quirie": "Hello JSON", "stream": "false"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"JSON response")
        engine.generate.assert_called_once_with([{"role": "user", "content": "Hello JSON"}])

    def test_chat2_post_form_success(self):
        engine.is_loaded = MagicMock(return_value=True)
        engine.generate = MagicMock(return_value="Form response")

        response = self.app.post('/api/chat2', data={"quirie": "Hello Form", "stream": "false"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"Form response")
        engine.generate.assert_called_once_with([{"role": "user", "content": "Hello Form"}])

if __name__ == '__main__':
    unittest.main()
