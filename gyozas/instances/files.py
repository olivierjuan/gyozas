from pathlib import Path
from typing import Literal

from numpy.random import Generator
from pyscipopt import Model

from gyozas.instances.instance_generator import InstanceGenerator


class FileGenerator(InstanceGenerator):
    """Instance generator that loads SCIP models from files on disk.

    Parameters
    ----------
    directory
        Path to the directory containing instance files.
    pattern
        Glob pattern to match files (e.g. ``"*.mps"``).
    recursive
        If True, search subdirectories recursively.
    sampling_mode
        ``"replace"`` to sample with replacement (default), ``"remove"`` to sample without.
    rng
        Random seed or numpy Generator for reproducibility.
    """

    def __init__(
        self,
        directory: Path | str = ".",
        pattern: str = "*",
        recursive: bool = False,
        sampling_mode: Literal["remove", "replace"] = "replace",
        rng: Generator | int | None = None,
    ) -> None:
        self.directory = directory
        self.pattern = pattern
        self.recursive = recursive
        self.sampling_mode = sampling_mode
        super().__init__(rng=rng)
        self.files = self._list_files()
        self._reset_file_list()

    def _list_files(self) -> list[Path]:
        files = []
        directory = Path(self.directory)
        if self.recursive:
            it = directory.rglob(self.pattern)
        else:
            it = directory.glob(self.pattern)
        for file in it:
            if file.is_file() or (file.is_symlink() and file.exists()):
                files.append(file)
        return files

    def __next__(self) -> Model:
        if self.done():
            raise StopIteration("No more files available.")
        if self.files_remaining == 0:
            self.files_remaining = len(self.files)

        idx = self.rng.integers(low=0, high=self.files_remaining)
        if self.sampling_mode.lower() == "replace":
            return self.generate_instance(self.files[idx])
        self.files_remaining -= 1
        self.files[idx], self.files[self.files_remaining] = self.files[self.files_remaining], self.files[idx]
        return self.generate_instance(self.files[self.files_remaining])

    def seed(self, seed) -> None:
        self._reset_file_list()
        super().seed(seed)

    def done(self) -> bool:
        no_files_at_all = len(self.files) == 0
        seen_all_files = self.files_remaining == 0 and self.sampling_mode.lower() == "remove"
        return no_files_at_all or seen_all_files

    def _reset_file_list(self) -> None:
        self.files.sort()
        self.files_remaining = len(self.files)

    def generate_instance(self, filepath) -> Model:
        """Load a SCIP model from a file.

        Parameters
        ----------
        filepath
            Path to the instance file.
        """
        model = Model()
        model.readProblem(filename=str(filepath))
        return model
