"""
Unit tests for TokenizerManager helper methods.

This tests the refactored tokenization functionality including input format detection,
tokenizer input preparation, and result extraction logic.

Usage:
python3 -m unittest test_tokenizer_manager.TestInputFormatDetection
python3 -m unittest test_tokenizer_manager.TestTokenizerInputPreparation
python3 -m unittest test_tokenizer_manager.TestTokenizerResultExtraction
python3 -m unittest test_tokenizer_manager.TestTokenizerManagerIntegration
"""

import asyncio
import unittest
from typing import List, Optional, Union
from unittest.mock import AsyncMock, Mock, patch

import transformers
import types
from types import SimpleNamespace
import sys
import torch
import importlib


_triton_stub = sys.modules.setdefault("triton", types.ModuleType("triton"))
_triton_lang_stub = sys.modules.setdefault("triton.language", types.ModuleType("triton.language"))
sys.modules.setdefault("sgl_kernel", types.ModuleType("sgl_kernel"))
_sgl_kvcache_stub = sys.modules.setdefault("sgl_kernel.kvcacheio", types.ModuleType("sgl_kernel.kvcacheio"))
_triton_runtime_stub = sys.modules.setdefault("triton.runtime", types.ModuleType("triton.runtime"))
_triton_runtime_jit_stub = sys.modules.setdefault(
    "triton.runtime.jit", types.ModuleType("triton.runtime.jit")
)
_triton_runtime_stub.jit = _triton_runtime_jit_stub
_triton_stub.runtime = _triton_runtime_stub

if not hasattr(_triton_stub, "jit"):
    _triton_stub.jit = lambda fn=None, **_: fn if fn is not None else (lambda f: f)

if not hasattr(_triton_stub, "Config"):
    class _TritionConfig:
        def __init__(self, *args, **kwargs):
            pass


    _triton_stub.Config = _TritionConfig

if not hasattr(_triton_stub, "autotune"):
    def _autotune_stub(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator


    _triton_stub.autotune = _autotune_stub

for _fn_name in [
    "transfer_kv_all_layer",
    "transfer_kv_all_layer_direct_lf_pf",
    "transfer_kv_all_layer_lf_pf",
    "transfer_kv_all_layer_mla",
    "transfer_kv_all_layer_mla_lf_pf",
    "transfer_kv_direct",
    "transfer_kv_per_layer",
    "transfer_kv_per_layer_direct_pf_lf",
    "transfer_kv_per_layer_mla",
    "transfer_kv_per_layer_mla_pf_lf",
    "transfer_kv_per_layer_pf_lf",
]:
    setattr(_sgl_kvcache_stub, _fn_name, lambda *args, **kwargs: None)

if not hasattr(_triton_runtime_jit_stub, "JITFunction"):
    class _DummyJIT:
        pass


    _triton_runtime_jit_stub.JITFunction = _DummyJIT

if not hasattr(_triton_lang_stub, "constexpr"):
    _triton_lang_stub.constexpr = lambda value: value


if not hasattr(transformers, "Qwen2_5_VLProcessor"):
    class _StubQwenProcessor:  # minimal stub to satisfy optional imports
        pass


    transformers.Qwen2_5_VLProcessor = _StubQwenProcessor
    try:
        from transformers.utils import import_utils

        import_utils._import_structure.setdefault("models.qwen2_5_vl", [])
        if "Qwen2_5_VLProcessor" not in import_utils._import_structure["models.qwen2_5_vl"]:
            import_utils._import_structure["models.qwen2_5_vl"].append("Qwen2_5_VLProcessor")

        import sys
        import types

        stub_module = types.ModuleType("transformers.models.qwen2_5_vl")
        stub_module.Qwen2_5_VLProcessor = _StubQwenProcessor
        sys.modules.setdefault("transformers.models.qwen2_5_vl", stub_module)
    except Exception:
        pass

if hasattr(torch, "compile"):
    torch.compile = lambda fn=None, **_: fn if fn is not None else (lambda f: f)

utils_module = importlib.import_module("sglang.srt.utils")
utils_module.cached_triton_kernel = lambda key_fn=None: (lambda fn: fn)

from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestInputFormatDetection(unittest.TestCase):
    """Test cases for _detect_input_format method."""

    def setUp(self):
        """Set up test fixtures."""
        with patch.object(TokenizerManager, "__init__", return_value=None):
            self.tokenizer_manager = TokenizerManager(None, None)
        self.tokenizer_manager.tokenizer = Mock()
        self.tokenizer_manager.async_dynamic_batch_tokenizer = None

    def test_detect_single_string(self):
        """Test detection of single string input."""
        text = "Hello world"
        result = self.tokenizer_manager._detect_input_format(
            text, is_cross_encoder=False
        )
        self.assertEqual(result, "single_string")

    def test_detect_single_string_cross_encoder_disabled(self):
        """Test single string with cross_encoder disabled still returns single_string."""
        text = "Hello world"
        result = self.tokenizer_manager._detect_input_format(
            text, is_cross_encoder=True
        )
        self.assertEqual(result, "single_string")

    def test_detect_batch_strings(self):
        """Test detection of batch string inputs."""
        texts = ["Hello", "World", "How are you?"]
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=False
        )
        self.assertEqual(result, "batch_strings")

    def test_detect_batch_strings_cross_encoder_disabled(self):
        """Test batch strings with cross_encoder disabled."""
        texts = ["Hello", "World"]
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=True
        )
        self.assertEqual(result, "batch_strings")

    def test_detect_cross_encoder_single_pair(self):
        """Test detection of cross-encoder single pair."""
        texts = [["query text", "document text"]]
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=True
        )
        self.assertEqual(result, "cross_encoder_pairs")

    def test_detect_cross_encoder_multiple_pairs(self):
        """Test detection of cross-encoder multiple pairs."""
        texts = [["q1", "d1"], ["q2", "d2"], ["q3", "d3"]]
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=True
        )
        self.assertEqual(result, "cross_encoder_pairs")

    def test_detect_cross_encoder_disabled_with_pairs(self):
        """Test pairs with cross_encoder disabled should return batch_strings."""
        texts = [["query", "document"]]
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=False
        )
        self.assertEqual(result, "batch_strings")

    def test_detect_empty_list(self):
        """Test detection with empty list."""
        texts = []
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=True
        )
        self.assertEqual(result, "batch_strings")

    def test_detect_malformed_cross_encoder_pairs(self):
        """Test malformed cross-encoder pairs (not length 2)."""
        texts = [["query only"]]  # Single element, not a pair
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=True
        )
        self.assertEqual(result, "batch_strings")

        texts = [["query", "doc", "extra"]]  # Three elements, not a pair
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=True
        )
        self.assertEqual(result, "batch_strings")


class TestTokenizerInputPreparation(unittest.TestCase):
    """Test cases for _prepare_tokenizer_input method."""

    def setUp(self):
        """Set up test fixtures."""
        with patch.object(TokenizerManager, "__init__", return_value=None):
            self.tokenizer_manager = TokenizerManager(None, None)
        self.tokenizer_manager.tokenizer = Mock()
        self.tokenizer_manager.async_dynamic_batch_tokenizer = None

    def test_prepare_single_string_input(self):
        """Test preparation of single string input."""
        text = "Hello world"
        result = self.tokenizer_manager._prepare_tokenizer_input(text, "single_string")
        self.assertEqual(result, ["Hello world"])

    def test_prepare_batch_strings_input(self):
        """Test preparation of batch strings input."""
        texts = ["Hello", "World", "Test"]
        result = self.tokenizer_manager._prepare_tokenizer_input(texts, "batch_strings")
        self.assertEqual(result, ["Hello", "World", "Test"])

    def test_prepare_cross_encoder_pairs_input(self):
        """Test preparation of cross-encoder pairs input."""
        texts = [["query1", "doc1"], ["query2", "doc2"]]
        result = self.tokenizer_manager._prepare_tokenizer_input(
            texts, "cross_encoder_pairs"
        )
        self.assertEqual(result, [["query1", "doc1"], ["query2", "doc2"]])

    def test_prepare_cross_encoder_single_pair_input(self):
        """Test preparation of single cross-encoder pair."""
        texts = [["query text", "document text"]]
        result = self.tokenizer_manager._prepare_tokenizer_input(
            texts, "cross_encoder_pairs"
        )
        self.assertEqual(result, [["query text", "document text"]])

    def test_prepare_unknown_input_format(self):
        """Test preparation with unknown input format falls back to returning as-is."""
        texts = ["test"]
        result = self.tokenizer_manager._prepare_tokenizer_input(
            texts, "unknown_format"
        )
        self.assertEqual(result, ["test"])


class TestTokenizerResultExtraction(unittest.TestCase):
    """Test cases for _extract_tokenizer_results method."""

    def setUp(self):
        """Set up test fixtures."""
        with patch.object(TokenizerManager, "__init__", return_value=None):
            self.tokenizer_manager = TokenizerManager(None, None)

    def test_extract_single_string_results(self):
        """Test extraction for single string input."""
        input_ids = [[101, 2129, 102]]
        token_type_ids = [[0, 0, 0]]

        result_input_ids, result_token_type_ids = (
            self.tokenizer_manager._extract_tokenizer_results(
                input_ids, token_type_ids, "single_string", original_batch_size=1
            )
        )

        self.assertEqual(result_input_ids, [101, 2129, 102])
        self.assertEqual(result_token_type_ids, [0, 0, 0])

    def test_extract_single_cross_encoder_results(self):
        """Test extraction for single cross-encoder pair."""
        input_ids = [[101, 2129, 102, 4068, 102]]
        token_type_ids = [[0, 0, 0, 1, 1]]

        result_input_ids, result_token_type_ids = (
            self.tokenizer_manager._extract_tokenizer_results(
                input_ids, token_type_ids, "cross_encoder_pairs", original_batch_size=1
            )
        )

        self.assertEqual(result_input_ids, [101, 2129, 102, 4068, 102])
        self.assertEqual(result_token_type_ids, [0, 0, 0, 1, 1])

    def test_extract_batch_results(self):
        """Test extraction for batch inputs."""
        input_ids = [[101, 2129, 102], [101, 4068, 102]]
        token_type_ids = [[0, 0, 0], [0, 0, 0]]

        result_input_ids, result_token_type_ids = (
            self.tokenizer_manager._extract_tokenizer_results(
                input_ids, token_type_ids, "batch_strings", original_batch_size=2
            )
        )

        self.assertEqual(result_input_ids, [[101, 2129, 102], [101, 4068, 102]])
        self.assertEqual(result_token_type_ids, [[0, 0, 0], [0, 0, 0]])

    def test_extract_multiple_cross_encoder_results(self):
        """Test extraction for multiple cross-encoder pairs."""
        input_ids = [[101, 2129, 102, 4068, 102], [101, 7592, 102, 2088, 102]]
        token_type_ids = [[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]]

        result_input_ids, result_token_type_ids = (
            self.tokenizer_manager._extract_tokenizer_results(
                input_ids, token_type_ids, "cross_encoder_pairs", original_batch_size=2
            )
        )

        self.assertEqual(
            result_input_ids, [[101, 2129, 102, 4068, 102], [101, 7592, 102, 2088, 102]]
        )
        self.assertEqual(result_token_type_ids, [[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]])

    def test_extract_empty_results(self):
        """Test extraction with empty results."""
        input_ids = []
        token_type_ids = None

        result_input_ids, result_token_type_ids = (
            self.tokenizer_manager._extract_tokenizer_results(
                input_ids, token_type_ids, "single_string", original_batch_size=1
            )
        )

        self.assertEqual(result_input_ids, [])
        self.assertIsNone(result_token_type_ids)

    def test_extract_with_none_token_type_ids(self):
        """Test extraction when token_type_ids is None."""
        input_ids = [[101, 2129, 102]]
        token_type_ids = None

        result_input_ids, result_token_type_ids = (
            self.tokenizer_manager._extract_tokenizer_results(
                input_ids, token_type_ids, "single_string", original_batch_size=1
            )
        )

        self.assertEqual(result_input_ids, [101, 2129, 102])
        self.assertIsNone(result_token_type_ids)


class TestTokenizerManagerIntegration(unittest.TestCase):
    """Integration tests combining multiple helper methods."""

    def setUp(self):
        """Set up test fixtures."""
        with patch.object(TokenizerManager, "__init__", return_value=None):
            self.tokenizer_manager = TokenizerManager(None, None)
        self.tokenizer_manager.tokenizer = Mock()
        self.tokenizer_manager.async_dynamic_batch_tokenizer = None

    def test_full_workflow_single_string(self):
        """Test complete workflow for single string input."""
        text = "Hello world"

        # Step 1: Detect format
        input_format = self.tokenizer_manager._detect_input_format(
            text, is_cross_encoder=False
        )
        self.assertEqual(input_format, "single_string")

        # Step 2: Prepare input
        tokenizer_input = self.tokenizer_manager._prepare_tokenizer_input(
            text, input_format
        )
        self.assertEqual(tokenizer_input, ["Hello world"])

        # Step 3: Extract results (simulated tokenizer output)
        mock_input_ids = [[101, 2129, 4248, 102]]
        mock_token_type_ids = None

        result_input_ids, result_token_type_ids = (
            self.tokenizer_manager._extract_tokenizer_results(
                mock_input_ids, mock_token_type_ids, input_format, original_batch_size=1
            )
        )

        self.assertEqual(result_input_ids, [101, 2129, 4248, 102])
        self.assertIsNone(result_token_type_ids)

    def test_full_workflow_cross_encoder_pairs(self):
        """Test complete workflow for cross-encoder pairs."""
        texts = [
            ["How many people live in Berlin?", "Berlin is well known for its museums."]
        ]

        # Step 1: Detect format
        input_format = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=True
        )
        self.assertEqual(input_format, "cross_encoder_pairs")

        # Step 2: Prepare input
        tokenizer_input = self.tokenizer_manager._prepare_tokenizer_input(
            texts, input_format
        )
        self.assertEqual(tokenizer_input, texts)

        # Step 3: Extract results (simulated tokenizer output for cross-encoder)
        mock_input_ids = [[101, 2129, 2116, 102, 4068, 2003, 102]]
        mock_token_type_ids = [[0, 0, 0, 0, 1, 1, 1]]

        result_input_ids, result_token_type_ids = (
            self.tokenizer_manager._extract_tokenizer_results(
                mock_input_ids, mock_token_type_ids, input_format, original_batch_size=1
            )
        )

        self.assertEqual(result_input_ids, [101, 2129, 2116, 102, 4068, 2003, 102])
        self.assertEqual(result_token_type_ids, [0, 0, 0, 0, 1, 1, 1])

    def test_full_workflow_batch_strings(self):
        """Test complete workflow for batch strings."""
        texts = ["Hello", "World", "Test"]

        # Step 1: Detect format
        input_format = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=False
        )
        self.assertEqual(input_format, "batch_strings")

        # Step 2: Prepare input
        tokenizer_input = self.tokenizer_manager._prepare_tokenizer_input(
            texts, input_format
        )
        self.assertEqual(tokenizer_input, ["Hello", "World", "Test"])

        # Step 3: Extract results (simulated tokenizer output)
        mock_input_ids = [[101, 7592, 102], [101, 2088, 102], [101, 2774, 102]]
        mock_token_type_ids = None

        result_input_ids, result_token_type_ids = (
            self.tokenizer_manager._extract_tokenizer_results(
                mock_input_ids, mock_token_type_ids, input_format, original_batch_size=3
            )
        )

        self.assertEqual(
            result_input_ids, [[101, 7592, 102], [101, 2088, 102], [101, 2774, 102]]
        )
        self.assertIsNone(result_token_type_ids)


class TestTokenizerManagerMultimodal(unittest.TestCase):
    """Tests covering multimodal preprocessing behaviour."""

    def setUp(self):
        with patch.object(TokenizerManager, "__init__", return_value=None):
            self.tokenizer_manager = TokenizerManager(None, None)
        self.tokenizer_manager.max_req_input_len = None
        self.tokenizer_manager.mm_processor = None
        self.tokenizer_manager.tokenizer = Mock()
        self.tokenizer_manager.server_args = SimpleNamespace(disable_radix_cache=True)
        self.tokenizer_manager.async_dynamic_batch_tokenizer = None
        self.tokenizer_manager.context_len = 4096
        self.tokenizer_manager.reserve_input_token_num = 0
        self.tokenizer_manager.enable_metrics = False
        self.tokenizer_manager.log_requests = False
        self.tokenizer_manager.log_requests_level = "info"
        self.tokenizer_manager.preferred_sampling_params = None
        self.tokenizer_manager.model_config = SimpleNamespace(vocab_size=32000)

    def _make_generate_request(self, video_path: str) -> GenerateReqInput:
        req = GenerateReqInput(
            text="describe the video",
            video_data=video_path,
            sampling_params={"max_new_tokens": 1},
        )
        req.normalize_batch_and_arguments()
        return req

    def test_video_data_forwarded_to_mm_processor(self):
        request = self._make_generate_request("/tmp/sample.mp4")

        mm_processor_mock = AsyncMock(return_value={"input_ids": [1, 2, 3]})
        self.tokenizer_manager.mm_processor = Mock()
        self.tokenizer_manager.mm_processor.process_mm_data_async = mm_processor_mock

        with patch.object(
            self.tokenizer_manager,
            "_tokenize_texts",
            new=AsyncMock(return_value=([10, 11, 12], None)),
        ), patch("sglang.srt.managers.tokenizer_manager.trace_slice_end"):
            result = asyncio.run(
                self.tokenizer_manager._tokenize_one_request(request)
            )

        mm_processor_mock.assert_awaited_once()
        call_kwargs = mm_processor_mock.await_args.kwargs
        self.assertIn("video_data", call_kwargs)
        self.assertEqual(call_kwargs["video_data"], request.video_data)
        self.assertIsNotNone(result)
        self.assertEqual(result.input_ids, [1, 2, 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
