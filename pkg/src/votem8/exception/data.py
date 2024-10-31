"""Exceptions raised by the data module."""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING as _TYPE_CHECKING

from votem8.exception import PackageError as _PackageError

if _TYPE_CHECKING:
    from pathlib import Path


class DataFileNotFoundError(_PackageError):
    """Raised when a requested package data file is not found.

    Parameters
    ----------
    path_relative
        Path to the file relative to the package's data directory.
    path_absolute
        Absolute path to the file.
    """

    def __init__(
        self,
        path_relative: str,
        path_absolute: Path,
    ):
        """Initialize the DataFileNotFoundError.

        Parameters
        ----------
        path_relative : str
            Path to the file relative to the package's data directory.
        path_absolute : Path
            Absolute path to the file.
        """
        self.path_relative = path_relative
        self.path_absolute = path_absolute
        super().__init__(
            "Could not find the requested package data file "
            f"'{path_relative}' at '{path_absolute}'.",
        )
        return
