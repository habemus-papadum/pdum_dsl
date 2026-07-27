"""Example tests for dsl."""

from pdum import dsl


def test_version():
    """Test that the package has a version."""
    assert hasattr(dsl, "__version__")
    assert isinstance(dsl.__version__, str)
    assert len(dsl.__version__) > 0


def test_import():
    """Test that the package can be imported."""
    assert dsl is not None
