# postProcess.py
import SIMPLE
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Post-processing / plotting
# ---------------------------------------------------------------------------

class PostProcessor:
    """
    Handles all visualization for a SIMPLE lid-driven cavity simulation.

    It takes a converged SimpleLidDrivenCavity instance and provides
    contour plots for u, v, p, and |U|.
    """

    def __init__(self, solver) -> None:
        self.solver = solver

    def _build_grid_for_field(self, field: np.ndarray):
        """Construct a uniform (X, Y) mesh that matches `field.shape`."""
        ny, nx = field.shape
        x = np.linspace(0.0, self.solver.grid.Lx, nx)
        y = np.linspace(0.0, self.solver.grid.Ly, ny)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def contour_u(self) -> None:
        X, Y = self._build_grid_for_field(self.solver.u.field)
        plt.figure()
        cs = plt.contourf(Y, X, self.solver.u.field, levels=50, cmap="plasma")
        plt.colorbar(cs, label="u (m/s)")
        plt.title("u-velocity")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    def contour_v(self) -> None:
        X, Y = self._build_grid_for_field(self.solver.v.field)
        plt.figure()
        cs = plt.contourf(X, Y, self.solver.v.field, levels=50, cmap="plasma")
        plt.colorbar(cs, label="v (m/s)")
        plt.title("v-velocity")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    def contour_p(self) -> None:
        X, Y = self._build_grid_for_field(self.solver.p.field)
        plt.figure()
        cs = plt.contourf(X, Y, self.solver.p.field, levels=50, cmap="plasma")
        plt.colorbar(cs, label="p (Pa)")
        plt.title("Pressure")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    def contour_speed(self) -> None:
        """Contour of |U| on a cell-centred grid from staggered u and v."""
        # interior pieces (ignore ghost cells)
        u_int = self.solver.u.field[1:-1, 1:-1]
        v_int = self.solver.v.field[1:-1, 1:-1]

        ny = min(u_int.shape[0], v_int.shape[0])
        nx = min(u_int.shape[1], v_int.shape[1])
        u_int = u_int[:ny, :nx]
        v_int = v_int[:ny, :nx]

        speed = np.sqrt(u_int ** 2 + v_int ** 2)

        x = np.linspace(
            self.solver.grid.Deltax * 0.5,
            self.solver.grid.Lx - self.solver.grid.Deltax * 0.5,
            nx,
        )
        y = np.linspace(
            self.solver.grid.Deltay * 0.5,
            self.solver.grid.Ly - self.solver.grid.Deltay * 0.5,
            ny,
        )
        Y, X = np.meshgrid(x, y)

        plt.figure()
        cs = plt.contourf(Y, X, speed, levels=50, cmap="plasma")
        plt.colorbar(cs, label="|U| (m/s)")
        plt.title("Velocity magnitude")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()
