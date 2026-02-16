def test_audio_extraction():
    import torchcodec
    import pytest

    def audio_extraction_method():
        # Simulate audio extraction logic
        raise OSError("Could not load this library: /opt/anaconda3/lib/python3.12/site-packages/torchcodec/libtorchcodec_core4.dylib")

    with pytest.raises(OSError):
        audio_extraction_method()