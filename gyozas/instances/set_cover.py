import numpy as np
from numpy.typing import NDArray
from pyscipopt import Model, quicksum

from gyozas.instances.instance_generator import InstanceGenerator, sanitize_rng


class SetCoverGenerator(InstanceGenerator):
    """Generator for random instances of the Set Cover problem.

    Attributes:
        n_rows (int): Number of rows (elements to be covered). Default is 500.
        n_cols (int): Number of columns (sets available for covering). Default is 1000.
        density (float): Fraction of nonzero entries in the set cover matrix. Default is 0.05.
        max_coef (int): Maximum coefficient value for the set cover matrix. Default is 100.
        rng (np.random.Generator): Random number generator for reproducibility.
    """

    def __init__(self, n_rows=500, n_cols=1000, density=0.05, max_coef=100, rng=None) -> None:
        super().__init__(rng=rng)
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.density = density
        self.max_coef = max_coef

    def __next__(self) -> Model:
        return self.generate_instance(
            n_rows=self.n_rows, n_cols=self.n_cols, density=self.density, max_coef=self.max_coef, rng=self.rng
        )

    @staticmethod
    def _get_counts(indices, n_cols) -> NDArray[np.int64]:
        counts = np.zeros(n_cols, dtype=int)
        for idx in indices:
            counts[idx] += 1
        return counts

    def _get_choice_in_range(self, start_index, end_index, num_samples, rng=None) -> NDArray[np.int64]:
        if rng is None:
            rng = self.rng
        choices = np.arange(start_index, end_index)
        samples = rng.choice(choices, num_samples, replace=False)
        return samples

    @staticmethod
    def _convert_csc_to_csr(indices, indptr, n_rows, n_cols) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        indptr_csr = np.zeros(n_rows + 1, dtype=int)
        indices_csr = np.zeros_like(indices)
        for j in range(len(indices)):
            indptr_csr[indices[j] + 1] += 1
        indptr_csr = np.cumsum(indptr_csr)
        for col in range(n_cols):
            for jj in range(indptr[col], indptr[col + 1]):
                row = indices[jj]
                indices_csr[indptr_csr[row]] = col
                indptr_csr[row] += 1
        last = 0
        for row in range(n_rows + 1):
            last, indptr_csr[row] = indptr_csr[row], last
        return indptr_csr, indices_csr

    def generate_instance(self, n_rows=500, n_cols=1000, density=0.05, max_coef=100, rng=None) -> Model:
        rng = sanitize_rng(rng, default=self.rng)

        nnzrs = int(n_rows * n_cols * density)
        indices = np.zeros(nnzrs, dtype=int)

        # Force at least 2 rows per column
        first_indices = np.arange(0, 2 * n_cols) % n_cols
        indices[0 : 2 * n_cols] = first_indices

        # Assign remaining column indexes at random
        samples = self._get_choice_in_range(0, n_cols * (n_rows - 2), nnzrs - (2 * n_cols), rng=rng) % n_cols
        indices[2 * n_cols : nnzrs] = samples

        # Get counts of unique elements
        col_n_rows = self._get_counts(indices, n_cols)

        # Ensure at least 1 column per row
        perm = rng.permutation(n_rows)
        indices[0:n_rows] = perm

        i = 0
        indptr = np.zeros(n_cols + 1, dtype=int)
        indptr_idx = 1

        for _idx, n in enumerate(col_n_rows):
            if i + n <= n_rows:
                pass
            elif i >= n_rows:
                sampled_rows = self._get_choice_in_range(0, n_rows, n, rng=rng)
                indices[i : i + n] = sampled_rows
            elif i + n > n_rows:
                remaining_rows = np.setdiff1d(np.arange(n_rows), indices[i:n_rows])
                choices = rng.choice(remaining_rows, i + n - n_rows, replace=False)
                indices[n_rows : i + n] = choices
            i += n
            indptr[indptr_idx] = i
            indptr_idx += 1

        # Convert CSC indices/ptrs to CSR
        indptr_csr, indices_csr = self._convert_csc_to_csr(indices, indptr, n_rows, n_cols)

        # Sample coefficients
        c = rng.integers(1, max_coef + 1, size=n_cols)

        model = Model(problemName=f"SetCover-{n_rows}-{n_cols}")
        model.setMinimize()

        # Add variables
        b_vars = []
        for i in range(n_cols):
            b_vars.append(model.addVar(vtype="B", lb=0.0, ub=1.0, name=f"x_{i}", obj=c[i]))

        # Add constraints
        for i in range(n_rows):
            model.addCons(
                quicksum(b_vars[indices_csr[j]] for j in range(indptr_csr[i], indptr_csr[i + 1])) >= 1,
                name=f"cover_{i}",
            )

        return model
