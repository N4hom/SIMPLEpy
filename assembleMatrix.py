# assembleMatrix.py



# sweeps.py
import copy
import numpy as np

from variable import variable
from linearSolver import tdma

debugging = False


class LineByLineSolver:
    """
    Line-by-line solver wrapper for a 'variable' field.

    Responsibilities:
    - build Row/Column systems (A, b) from the field coefficients + BCs
    - solve them with TDMA
    - perform directional sweeps
    """

    def __init__(self, field: variable) -> None:
        self.field = field
        self.Nx = field.Nx
        self.Ny = field.Ny

        # Place-holders for the line-by-line procedure
                # Place-holders for the line-by-line procedure
        # Rows (x-sweeps): one set of diagonals per row
        self.row_lower = [None] * self.Ny
        self.row_diag  = [None] * self.Ny
        self.row_upper = [None] * self.Ny
        self.bxs       = [None] * self.Ny

        # Columns (y-sweeps): one set of diagonals per column
        self.col_lower = [None] * self.Nx
        self.col_diag  = [None] * self.Nx
        self.col_upper = [None] * self.Nx
        self.bys       = [None] * self.Nx


    # ------------------------------------------------------------------
    # Row assembly
    # ------------------------------------------------------------------
    def buildRow(self, j: int) -> None:
        a = self.field.a
        field = self.field.field

        if self.field.name == "pCorr":
            alpha = 1.0
        else:
            alpha = self.field.alpha

        Nx = self.Nx
        Ny = self.Ny

        # Tridiagonal diagonals
        a_diag  = np.zeros(Nx)
        a_lower = np.zeros(Nx)  # sub-diagonal; a_lower[0] unused
        a_upper = np.zeros(Nx)  # super-diagonal; a_upper[-1] unused

        bx = copy.deepcopy(a.b[:, j])

        if debugging:
            print("b at row in buildRow before BCs", j)
            print(bx)

        # First row (i = 0)
        a_diag[0] = a.p[0, j] / alpha
        a_upper[0] = -a.e[0, j]

        # Last row (i = Nx-1)
        a_diag[Nx - 1] = a.p[Nx - 1, j] / alpha
        a_lower[Nx - 1] = -a.w[Nx - 1, j]

        # Internal rows
        for l in range(1, Nx - 1):
            a_diag[l] = a.p[l, j] / alpha
            a_lower[l] = -a.w[l, j]
            a_upper[l] = -a.e[l, j]

        # EAST/WEST BC contributions (Dirichlet)
        if self.field.name != "pCorr":
            bx[0] += field[0, j + 1] * a.w[0, j]
            bx[Nx - 1] += field[Nx + 1, j + 1] * a.e[Nx - 1, j]

        # NORTH/SOUTH BC contributions
        if j == Ny - 1:
            # north boundary
            for i in range(Nx):
                bx[i] += field[i + 1, Ny + 1] * a.n[i, j]

        if j == 0:
            # south boundary
            for i in range(Nx):
                if debugging:
                    print("field[i + 1, j],", field[i + 1, j])
                    print("a.s[i, j],", a.s[i, j])
                    print("field[i + 1, j]*a.s[i, j]", field[i + 1, j] * a.s[i, j])
                bx[i] += field[i + 1, j] * a.s[i, j]

        if debugging:
            print(f"row diag for {self.field.name} at j={j}:", a_diag)
            print(f"row lower for {self.field.name} at j={j}:", a_lower)
            print(f"row upper for {self.field.name} at j={j}:", a_upper)
            print(f"bx for {self.field.name} at row {j}:\n", bx)

        self.row_diag[j]  = a_diag
        self.row_lower[j] = a_lower
        self.row_upper[j] = a_upper
        self.bxs[j]       = bx


    # ------------------------------------------------------------------
    # Column assembly
    # ------------------------------------------------------------------
    def buildColumn(self, i: int) -> None:
        a = self.field.a
        field = self.field.field

        if self.field.name == "pCorr":
            alpha = 1.0
        else:
            alpha = self.field.alpha

        Nx = self.Nx
        Ny = self.Ny

        a_diag  = np.zeros(Ny)
        a_lower = np.zeros(Ny)
        a_upper = np.zeros(Ny)

        by = copy.deepcopy(a.b[i, :])

        # First entry (j = 0)
        a_diag[0] = a.p[i, 0] / alpha
        a_upper[0] = -a.n[i, 0]

        # Last entry (j = Ny-1)
        a_diag[Ny - 1] = a.p[i, Ny - 1] / alpha
        a_lower[Ny - 1] = -a.s[i, Ny - 1]

        # Internal entries
        for l in range(1, Ny - 1):
            a_diag[l]  = a.p[i, l] / alpha
            a_lower[l] = -a.s[i, l]
            a_upper[l] = -a.n[i, l]

        if self.field.name != "pCorr":
            by[0]      += field[i + 1, 0] * a.s[i, 0]
            by[Ny - 1] += field[i + 1, Ny + 1] * a.n[i, Ny - 1]

        if i == Nx - 1:
            for j in range(Ny):
                by[j] += field[Nx + 1, j + 1] * a.e[i, j]

        if i == 0:
            for j in range(Ny):
                by[j] += field[0, j + 1] * a.w[i, j]

        if debugging:
            print(f"col diag for {self.field.name} at i={i}:", a_diag)
            print(f"col lower for {self.field.name} at i={i}:", a_lower)
            print(f"col upper for {self.field.name} at i={i}:", a_upper)
            print(f"by for {self.field.name} at column {i}:\n", by)

        self.col_diag[i]  = a_diag
        self.col_lower[i] = a_lower
        self.col_upper[i] = a_upper
        self.bys[i]       = by


    # ------------------------------------------------------------------
    # Row/column solves
    # ------------------------------------------------------------------
    def solveRow(self, row: int) -> None:
        a = self.field.a
        field = self.field.field

        Nx = self.Nx
        Ny = self.Ny

        if self.field.name == "pCorr":
            alpha = 1.0
        else:
            alpha = self.field.alpha

        a_diag  = self.row_diag[row].copy()
        a_lower = self.row_lower[row].copy()
        a_upper = self.row_upper[row].copy()
        b       = copy.deepcopy(self.bxs[row])

        if debugging:
            print("before filling b for each cell in row", row)
            print("b:", b)

        # Contributions from north/south neighbours (internal)
        if row > 0:
            for i in range(Nx):
                b[i] += field[i + 1, row] * a.s[i, row]

        if row < Ny - 1:
            for i in range(Nx):
                b[i] += field[i + 1, row + 2] * a.n[i, row]

        for i in range(Nx):
            b[i] += (1.0 / alpha - 1.0) * field[i + 1, row + 1] * a.p[i, row]

        if debugging:
            print(f"diag at row {row}:", a_diag)
            print(f"lower at row {row}:", a_lower)
            print(f"upper at row {row}:", a_upper)
            print(f"b vector at row {row}:\n", b)

        solution = tdma(a_lower, a_diag, a_upper, b)
        field[1:self.field.iMax, row + 1] = solution

        if debugging:
            print(f"solution at row {row}:", solution)

    def solveColumn(self, column: int) -> None:
        a = self.field.a
        field = self.field.field

        Nx = self.Nx
        Ny = self.Ny

        if self.field.name == "pCorr":
            alpha = 1.0
        else:
            alpha = self.field.alpha

        a_diag  = self.col_diag[column].copy()
        a_lower = self.col_lower[column].copy()
        a_upper = self.col_upper[column].copy()
        b       = copy.deepcopy(self.bys[column])

        # Contributions from west/east neighbours (internal)
        if column > 0:
            for j in range(Ny):
                b[j] += field[column, j + 1] * a.w[column, j]

        if column < Nx - 1:
            for j in range(Ny):
                b[j] += field[column + 2, j + 1] * a.e[column, j]

        for j in range(Ny):
            b[j] += (1.0 / alpha - 1.0) * field[column + 1, j + 1] * a.p[column, j]

        if debugging:
            print(f"diag at column {column}:", a_diag)
            print(f"lower at column {column}:", a_lower)
            print(f"upper at column {column}:", a_upper)
            print(f"b vector at column {column}:\n", b)

        solution = tdma(a_lower, a_diag, a_upper, b)
        field[column + 1, 1:self.field.jMax] = solution

        if debugging:
            print(f"solution at column {column}:", solution)


    # ------------------------------------------------------------------
    # Sweeps
    # ------------------------------------------------------------------
    def sweepEastToWest(self) -> None:
        if debugging:
            print(f"Sweeping {self.field.name} from east to west...")

        for column in range(self.Nx):
            self.buildColumn(column)

        for column in range(self.Nx - 1, -1, -1):
            self.solveColumn(column)

    def sweepWestToEast(self) -> None:
        if debugging:
            print(f"Sweeping {self.field.name} from west to east...")

        for column in range(self.Nx):
            self.buildColumn(column)

        for column in range(self.Nx):
            self.solveColumn(column)

    def sweepNorthToSouth(self) -> None:
        if debugging:
            print(f"Sweeping {self.field.name} from north to south...")

        for row in range(self.Ny):
            self.buildRow(row)

        for row in range(self.Ny - 1, -1, -1):
            self.solveRow(row)

    def sweepSouthToNorth(self) -> None:
        if debugging:
            print(f"Sweeping {self.field.name} from south to north...")

        for row in range(self.Ny):
            self.buildRow(row)

        for row in range(self.Ny):
            self.solveRow(row)


