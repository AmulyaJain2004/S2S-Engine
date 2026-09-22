"""Minimal tests for P0. Run with: python -m pytest tests/test_p0.py -v
Requires network access (Hugging Face Hub) and, ideally, a GPU -- these are
NOT meant to run in a network-isolated sandbox.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import pick_device


@pytest.fixture(scope="module")
def device():
    return pick_device("cuda")


def test_wavlm_loads(device):
    from encoder.wavlm_encoder import WavLMEncoder
    encoder = WavLMEncoder(device=device)
    assert encoder.model is not None


def test_wavlm_feature_extraction(device):
    from encoder.wavlm_encoder import WavLMEncoder
    encoder = WavLMEncoder(device=device)
    dummy = torch.randn(1, 16000)  # 1 second of noise at 16kHz
    features = encoder(dummy, sample_rate=16000)
    assert features.dim() == 3
    assert features.shape[0] == 1


def test_vocos_loads(device):
    from vocoder.vocos_wrapper import VocosVocoder
    vocoder = VocosVocoder(device=device)
    assert vocoder.model is not None


def test_vocos_mel_extraction(device):
    from vocoder.vocos_wrapper import VocosVocoder
    vocoder = VocosVocoder(device=device)
    dummy = torch.randn(1, 24000)  # 1 second at 24kHz
    mel = vocoder.mel_from_waveform(dummy)
    assert mel.dim() == 3


def test_vocos_reconstruction(device):
    from vocoder.vocos_wrapper import VocosVocoder
    vocoder = VocosVocoder(device=device)
    dummy = torch.randn(1, 24000)
    reconstructed = vocoder.copy_synthesis(dummy)
    assert reconstructed.dim() == 2


def test_otospeech_sample_inspection():
    from dataprep.inspect_otospeech import inspect_one_sample
    sample = inspect_one_sample()
    assert sample is not None


def test_otospeech_channel_separation(tmp_path):
    from dataprep.preprocess_otospeech import SchemaNotConfigured, preprocess_one_session
    try:
        preprocess_one_session(repo_id="otoearth/otoSpeech-full-duplex-task-oriented-20h",
                                output_dir=str(tmp_path))
    except SchemaNotConfigured:
        pytest.skip("SCHEMA MAPPING not yet filled in dataprep/preprocess_otospeech.py -- "
                    "run inspect_otospeech.py first and fill in the real field names")
