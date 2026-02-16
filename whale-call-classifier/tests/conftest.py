import pytest

@pytest.fixture(scope='session', autouse=True)
def setup_torchcodec():
    try:
        import torchcodec
    except ImportError:
        pytest.skip("torchcodec not available, skipping tests.")
    except OSError as e:
        pytest.skip(f"Could not load torchcodec library: {e}")