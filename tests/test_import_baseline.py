"""Import smoke tests for the app and CLI entry points."""


def test_app_and_main_importable() -> None:
    import app  # noqa: F401
    import main  # noqa: F401

