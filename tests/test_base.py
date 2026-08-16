"""Tests for model data management."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

from pranaam.base import Base
from pranaam.utils import SecurityError


class TestBase:
    """Test the model cache contract."""

    def test_base_without_model_directory_returns_none(self) -> None:
        """A base class without a model directory has nothing to load."""
        assert Base.load_model_data("test_file") is None

    @patch("pranaam.base.download_file")
    @patch("pranaam.base.user_cache_path")
    def test_missing_model_downloads_to_user_cache(
        self, mock_cache: Mock, mock_download: Mock, tmp_path: Path
    ) -> None:
        """Models are installed in a writable user cache, not the package tree."""
        mock_cache.return_value = tmp_path

        def install(url: str, target: str, file_name: str) -> bool:
            (Path(target) / file_name).mkdir()
            return True

        mock_download.side_effect = install

        class TestClass(Base):
            MODELFN = "model"

        result = TestClass.load_model_data("test_model")

        assert result == tmp_path / "model"
        mock_cache.assert_called_once_with("pranaam", ensure_exists=True)
        mock_download.assert_called_once()

    @patch("pranaam.base.download_file")
    @patch("pranaam.base.verify_model_files")
    @patch("pranaam.base.user_cache_path")
    def test_existing_model_is_reused(
        self,
        mock_cache: Mock,
        mock_verify: Mock,
        mock_download: Mock,
        tmp_path: Path,
    ) -> None:
        """An existing pinned model is reused by default."""
        mock_cache.return_value = tmp_path
        (tmp_path / "model" / "test_model").mkdir(parents=True)

        class TestClass(Base):
            MODELFN = "model"

        assert TestClass.load_model_data("test_model") == tmp_path / "model"
        mock_verify.assert_called_once_with(tmp_path / "model" / "test_model")
        mock_download.assert_not_called()

    @patch("pranaam.base.download_file", return_value=False)
    @patch("pranaam.base.verify_model_files")
    @patch("pranaam.base.user_cache_path")
    def test_failed_refresh_preserves_verified_cache(
        self,
        mock_cache: Mock,
        mock_verify: Mock,
        mock_download: Mock,
        tmp_path: Path,
    ) -> None:
        """A failed forced refresh keeps serving the existing bundle."""
        mock_cache.return_value = tmp_path
        cached_model = tmp_path / "model" / "test_model"
        cached_model.mkdir(parents=True)

        class TestClass(Base):
            MODELFN = "model"

        assert TestClass.load_model_data("test_model", latest=True) == (
            tmp_path / "model"
        )
        assert cached_model.is_dir()
        mock_verify.assert_called_once_with(cached_model)
        mock_download.assert_called_once()

    @patch("pranaam.base.download_file", return_value=False)
    @patch(
        "pranaam.base.verify_model_files",
        side_effect=SecurityError("checksum mismatch"),
    )
    @patch("pranaam.base.user_cache_path")
    def test_invalid_cache_is_not_used_when_refresh_fails(
        self,
        mock_cache: Mock,
        mock_verify: Mock,
        mock_download: Mock,
        tmp_path: Path,
    ) -> None:
        """A corrupt cache is never returned as a fallback."""
        mock_cache.return_value = tmp_path
        cached_model = tmp_path / "model" / "test_model"
        cached_model.mkdir(parents=True)

        class TestClass(Base):
            MODELFN = "model"

        assert TestClass.load_model_data("test_model") is None
        mock_verify.assert_called_once_with(cached_model)
        mock_download.assert_called_once()

    @patch("pranaam.base.download_file", return_value=False)
    @patch("pranaam.base.user_cache_path")
    def test_download_failure_without_cache_returns_none(
        self, mock_cache: Mock, mock_download: Mock, tmp_path: Path
    ) -> None:
        """A failed first download cannot masquerade as a usable model path."""
        mock_cache.return_value = tmp_path

        class TestClass(Base):
            MODELFN = "model"

        assert TestClass.load_model_data("test_model") is None
        mock_download.assert_called_once()

    @patch("pranaam.base.download_file")
    @patch("pranaam.base.verify_model_files")
    @patch("pranaam.base.user_cache_path")
    def test_concurrent_cache_misses_download_once(
        self,
        mock_cache: Mock,
        mock_verify: Mock,
        mock_download: Mock,
        tmp_path: Path,
    ) -> None:
        """The cache lock serializes concurrent first-time downloads."""
        mock_cache.return_value = tmp_path
        downloading = Event()
        release_download = Event()

        def install(url: str, target: str, file_name: str) -> bool:
            downloading.set()
            assert release_download.wait(timeout=2)
            (Path(target) / file_name).mkdir()
            return True

        mock_download.side_effect = install

        class TestClass(Base):
            MODELFN = "model"

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(TestClass.load_model_data, "test_model")
            assert downloading.wait(timeout=2)
            second = executor.submit(TestClass.load_model_data, "test_model")
            release_download.set()
            assert first.result(timeout=2) == tmp_path / "model"
            assert second.result(timeout=2) == tmp_path / "model"

        mock_download.assert_called_once()
        mock_verify.assert_called_once_with(tmp_path / "model" / "test_model")


class TestBaseInheritance:
    """Test class-level model directory configuration."""

    def test_subclasses_keep_independent_model_directories(self) -> None:
        """Changing one subclass does not mutate the base or a sibling."""

        class BaseA(Base):
            MODELFN = "model_a"

        class BaseB(Base):
            MODELFN = "model_b"

        assert Base.MODELFN is None
        assert BaseA.MODELFN == "model_a"
        assert BaseB.MODELFN == "model_b"
