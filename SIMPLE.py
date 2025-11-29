#!/usr/bin/env python3
import copy
import math

import matplotlib.pyplot as plt
import numpy as np

from grid import Grid2D
from variable import variable, coeff
from assembleMatrix import LineByLineSolver


debugging = False


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def power(a: float, n: int) -> float:
    """Avoid python native **"""
    """Much faster           """
    tmp = 1.0
    for _ in range(n):
        tmp *= a
    return tmp


def A(P: float) -> float:
    """QUICK-like blending function."""
    return max(0.0, power((1.0 - 0.1 * abs(P)), 5))




class FluidProperties:
    """Container for constant-property, incompressible fluid data."""

    def __init__(self, rho: float, mu: float, Re: float, Lref: float) -> None:
        self.rho = rho
        self.mu = mu
        self.Re = Re
        self.Lref = Lref

    @property
    def top_velocity(self) -> float:
        """Lid velocity from Re = rho*U*L/mu."""
        return self.Re * self.mu / (self.rho * self.Lref)


# ---------------------------------------------------------------------------
# SIMPLE-based solver for lid-driven cavity
# ---------------------------------------------------------------------------

class SimpleLidDrivenCavity:
    """
    SIMPLE algorithm for a 2D lid-driven cavity on a staggered grid.

    Responsibilities:
    - Owns grid and fluid properties.
    - Owns fields (p, pCorr, u, v).
    - Assembles momentum and pressure-correction equations.
    - Performs SIMPLE iterations and convergence checks.
    """

    def __init__(
        self,
        Nx: int,
        Ny: int,
        Lx: float,
        Ly: float,
        density: float,
        viscosity: float,
        Re: float = 100.0,
        p_tolerance: float = 1e-5,
        u_tolerance: float = 1e-6,
        v_tolerance: float = 1e-6,
    ) -> None:

        # Grid and fluid
        self.grid = Grid2D(Nx, Ny, Lx, Ly)
        self.fluid = FluidProperties(density, viscosity, Re, Lx)

        # Tolerances
        self.p_tol = p_tolerance
        self.u_tol = u_tolerance
        self.v_tol = v_tolerance

        # Convenience aliases
        Nx_p = Nx
        Ny_p = Ny
        Nx_u = Nx - 1
        Ny_u = Ny
        Nx_v = Nx
        Ny_v = Ny - 1

        top_velocity = self.fluid.top_velocity

        # Primary fields (with SIMPLE relaxation factors)
        self.p = variable("p", Nx_p, Ny_p, 0.0, 0.0, 0.0, 0.0)
        self.pCorr = variable("pCorr", Nx_p, Ny_p, 0.0, 0.0, 0.0, 0.0, 0.7, p_tolerance)
        self.u = variable("u", Nx_u, Ny_u, top_velocity, 0.0, 0.0, 0.0, 0.5, u_tolerance)
        self.v = variable("v", Nx_v, Ny_v, 0.0, 0.0, 0.0, 0.0, 0.5, v_tolerance)

        # Initialisation
        self.p.initialize(0.0)
        self.pCorr.initialize()
        self.u.initialize()
        self.v.initialize()

        # Defined line-by-line solvers for each field
        self.uSolver = LineByLineSolver(self.u)
        self.vSolver = LineByLineSolver(self.v)
        self.pCorrSolver = LineByLineSolver(self.pCorr)

    # ------------------------------------------------------------------
    # Assembly routines (momentum + pressure correction)
    # ------------------------------------------------------------------
    def assemble_u_momentum(self) -> None:
        """Build coefficients for the u-momentum equation."""

        # Local aliases
        u = self.u
        v = self.v
        p = self.p
        rho = self.fluid.rho
        mu = self.fluid.mu
        Deltax_default = self.grid.Deltax
        Deltay = self.grid.Deltay

        F = coeff()
        D = coeff()

        # Loop over internal u-cells (note: u.iMax = Nx_u + 1, with ghost cells)
        for i in range(1, u.iMax):
            for j in range(1, u.jMax):

                # Geometric adjustments near boundaries for the staggered layout
                if i == 1 or i == u.iMax - 1:
                    Deltax = 1.5 * Deltax_default
                else:
                    Deltax = Deltax_default

                dx = Deltax_default  # diffusion length in x
                dyn = Deltay
                dys = Deltay

                if j == 1:
                    dys = 0.5 * Deltay
                if j == u.jMax - 1:
                    dyn = 0.5 * Deltay

                # Diffusion coefficients
                D.e = mu * Deltay / dx
                D.w = mu * Deltay / dx
                D.n = mu * Deltax / dyn
                D.s = mu * Deltax / dys

                # Mass fluxes at cell faces
                if i == u.iMax - 1:
                    F.e = rho * Deltay * u[u.iMax, j]
                else:
                    F.e = rho * Deltay * 0.5 * (u[i, j] + u[i + 1, j])

                if i == 1:
                    F.w = rho * Deltay * u[0, j]
                else:
                    F.w = rho * Deltay * 0.5 * (u[i - 1, j] + u[i, j])

                if i == 1:
                    F.s = rho * Deltax * (v[0, j - 1] + 3.0 * v[1, j - 1] + 2.0 * v[2, j - 1]) / 6.0
                    F.n = rho * Deltax * (v[0, j] + 3.0 * v[1, j] + 2.0 * v[2, j]) / 6.0
                elif i == u.iMax - 1:
                    F.s = rho * Deltax * (
                        v[u.iMax - 1, j - 1]
                        + 3.0 * v[u.iMax, j - 1]
                        + 2.0 * v[u.iMax + 1, j - 1]
                    ) / 6.0
                    F.n = rho * Deltax * (
                        v[u.iMax - 1, j]
                        + 3.0 * v[u.iMax, j]
                        + 2.0 * v[u.iMax + 1, j]
                    ) / 6.0
                else:
                    F.s = rho * Deltax * 0.5 * (v[i, j - 1] + v[i + 1, j - 1])
                    F.n = rho * Deltax * 0.5 * (v[i, j] + v[i + 1, j])

                # Hybrid upwind / power-law coefficients
                a_n = D.n * A(F.n / D.n) + max(-F.n, 0.0)
                a_s = D.s * A(F.s / D.s) + max(F.s, 0.0)
                a_e = D.e * A(F.e / D.e) + max(-F.e, 0.0)
                a_w = D.w * A(F.w / D.w) + max(F.w, 0.0)

                # Pressure gradient source term
                source_term = Deltay * (p[i, j] - p[i + 1, j])

                # Map (i, j) in the field with ghost cells to (i-1, j-1) in coeff arrays
                u.buildCoefficients(a_n, a_s, a_w, a_e, source_term, i - 1, j - 1)

    def assemble_v_momentum(self) -> None:
        """Build coefficients for the v-momentum equation."""

        v = self.v
        u = self.u
        p = self.p
        rho = self.fluid.rho
        mu = self.fluid.mu

        Deltax_default = self.grid.Deltax
        Deltay_default = self.grid.Deltay

        F = coeff()
        D = coeff()

        for i in range(1, v.iMax):
            for j in range(1, v.jMax):

                # Effective cell height for staggered v-cells
                if j == 1 or j == v.jMax - 1:
                    Deltay = 1.5 * Deltay_default
                else:
                    Deltay = Deltay_default

                dy = Deltay_default
                dxe = Deltax_default
                dxw = Deltax_default

                if i == 1:
                    dxw = 0.5 * Deltax_default
                if i == v.iMax - 1:
                    dxe = 0.5 * Deltax_default

                D.e = mu * Deltay / dxe
                D.w = mu * Deltay / dxw
                D.n = mu * Deltax_default / dy
                D.s = mu * Deltax_default / dy

                # SOUTH/NORTH fluxes
                if j == 1:
                    F.s = rho * Deltax_default * v[i, 0]
                else:
                    F.s = rho * Deltax_default * 0.5 * (v[i, j] + v[i, j - 1])

                if j == v.jMax - 1:
                    F.n = rho * Deltax_default * v[i, v.jMax]
                else:
                    F.n = rho * Deltax_default * 0.5 * (v[i, j] + v[i, j + 1])

                # WEST/EAST fluxes
                if j == 1:
                    F.w = rho * Deltay * (u[i - 1, 0] + 3.0 * u[i - 1, 1] + 2.0 * u[i - 1, 2]) / 6.0
                    F.e = rho * Deltay * (u[i, 0] + 3.0 * u[i, 1] + 2.0 * u[i, 2]) / 6.0
                elif j == v.jMax - 1:
                    F.w = rho * Deltay * (
                        2.0 * u[i - 1, v.jMax - 1]
                        + 3.0 * u[i - 1, v.jMax]
                        + u[i - 1, v.jMax + 1]
                    ) / 6.0
                    F.e = rho * Deltay * (
                        2.0 * u[i, v.jMax - 1]
                        + 3.0 * u[i, v.jMax]
                        + u[i, v.jMax + 1]
                    ) / 6.0
                else:
                    F.w = rho * Deltay * 0.5 * (u[i - 1, j] + u[i - 1, j + 1])
                    F.e = rho * Deltay * 0.5 * (u[i, j] + u[i, j + 1])

                # Discretisation coefficients
                a_n = D.n * A(F.n / D.n) + max(-F.n, 0.0)
                a_s = D.s * A(F.s / D.s) + max(F.s, 0.0)
                a_e = D.e * A(F.e / D.e) + max(-F.e, 0.0)
                a_w = D.w * A(F.w / D.w) + max(F.w, 0.0)

                source_term = Deltax_default * (p[i, j] - p[i, j + 1])

                v.buildCoefficients(a_n, a_s, a_w, a_e, source_term, i - 1, j - 1)

    def assemble_p_correction(self) -> None:
        """Build coefficients for the pressure-correction equation."""
        pCorr = self.pCorr
        u = self.u
        v = self.v

        rho = self.fluid.rho
        Deltax = self.grid.Deltax
        Deltay = self.grid.Deltay

        for i in range(1, pCorr.iMax):
            for j in range(1, pCorr.jMax):

                a_n = 0.0
                a_s = 0.0
                a_e = 0.0
                a_w = 0.0

                if i > 1:
                    a_w = rho * (Deltay / u.a.p[i - 2, j - 1]) * Deltay
                if i < pCorr.iMax - 1:
                    a_e = rho * (Deltay / u.a.p[i - 1, j - 1]) * Deltay
                if j > 1:
                    a_s = rho * (Deltax / v.a.p[i - 1, j - 2]) * Deltax
                if j < pCorr.jMax - 1:
                    a_n = rho * (Deltax / v.a.p[i - 1, j - 1]) * Deltax

                b = rho * (
                    Deltay * (u[i - 1, j] - u[i, j]) +
                    Deltax * (v[i, j - 1] - v[i, j])
                )

                pCorr.buildCoefficients(a_n, a_s, a_w, a_e, b, i - 1, j - 1)

    # ------------------------------------------------------------------
    # Correction steps
    # ------------------------------------------------------------------
    def correct_u(self) -> None:
        u = self.u
        pCorr = self.pCorr
        Deltay = self.grid.Deltay

        for i in range(1, u.iMax):
            for j in range(1, u.jMax):
                u[i, j] = u[i, j] + Deltay / u.a.p[i - 1, j - 1] * (
                    pCorr[i, j] - pCorr[i + 1, j]
                )

    def correct_v(self) -> None:
        v = self.v
        pCorr = self.pCorr
        Deltax = self.grid.Deltax

        for i in range(1, v.iMax):
            for j in range(1, v.jMax):
                v[i, j] = v[i, j] + Deltax / v.a.p[i - 1, j - 1] * (
                    pCorr[i, j] - pCorr[i, j + 1]
                )

    def correct_p(self) -> None:
        p = self.p
        pCorr = self.pCorr
        Nx = self.grid.Nx
        Ny = self.grid.Ny

        for i in range(1, pCorr.iMax):
            for j in range(1, pCorr.jMax):
                p[i, j] = p[i, j] + pCorr.alpha * pCorr[i, j]

        # Reset correction field
        pCorr.field = np.zeros((Nx + 2, Ny + 2))

        # Re-apply Neumann pressure BCs on all boundaries
        p[:, 0] = p[:, 1]                # south
        p[:, p.jMax] = p[:, p.jMax - 1]  # north
        p[p.iMax, :] = p[p.iMax - 1, :]  # east
        p[0, :] = p[1, :]                # west

    # ------------------------------------------------------------------
    # Residuals and convergence
    # ------------------------------------------------------------------
    def _velocity_residual(self, component: str) -> float:
        """Generic residual for u or v using the assembled coefficients."""
        if component == "u":
            field = self.u
        elif component == "v":
            field = self.v
        else:
            raise ValueError("velocity component must be 'u' or 'v'")

        res_num = 0.0
        res_den = 0.0

        for i in range(1, field.iMax):
            for j in range(1, field.jMax):

                ap = field.a.p[i - 1, j - 1]
                an = field.a.n[i - 1, j - 1]
                as_ = field.a.s[i - 1, j - 1]
                ae = field.a.e[i - 1, j - 1]
                aw = field.a.w[i - 1, j - 1]
                b = field.a.b[i - 1, j - 1]

                phi_p = field[i, j]
                phi_n = field[i, j + 1]
                phi_s = field[i, j - 1]
                phi_e = field[i + 1, j]
                phi_w = field[i - 1, j]

                res_den += ap * phi_p

                res = (
                    ap * phi_p
                    - an * phi_n
                    - as_ * phi_s
                    - ae * phi_e
                    - aw * phi_w
                    - b
                )
                res_num += res

        return res_num / max(res_den, 1e-16)

    def _p_residual(self) -> float:
        """Continuity residual (mass imbalance)."""
        u = self.u
        v = self.v
        pCorr = self.pCorr

        Lref = self.grid.Lx
        Deltax = self.grid.Deltax
        Deltay = self.grid.Deltay

        res_num = 0.0
        for i in range(1, pCorr.iMax):
            for j in range(1, pCorr.jMax):
                res_num += abs(
                    Deltay * (u[i - 1, j] - u[i, j])
                    + Deltax * (v[i, j - 1] - v[i, j])
                )

        return res_num / (self.fluid.top_velocity * Lref)

    def has_converged(self) -> bool:
        """Check all convergence criteria."""
        p_res = self._p_residual()
        u_res = self._velocity_residual("u")
        v_res = self._velocity_residual("v")

        if debugging:
            print(f"Residuals: p={p_res:.3e}, u={u_res:.3e}, v={v_res:.3e}")

        return (
            p_res < self.p_tol
            and u_res < self.u_tol
            and v_res < self.v_tol
        )

    # ------------------------------------------------------------------
    # SIMPLE loop
    # ------------------------------------------------------------------
    def iterate(self) -> None:
        # Assemble momentum equations
        self.assemble_u_momentum()
        self.assemble_v_momentum()

        # Solve momentum using line-by-line solvers
        self.uSolver.sweepWestToEast()
        self.uSolver.sweepSouthToNorth()
        self.uSolver.sweepEastToWest()

        self.vSolver.sweepWestToEast()
        self.vSolver.sweepSouthToNorth()
        self.vSolver.sweepEastToWest()

        # Assemble and solve pressure-correction
        self.assemble_p_correction()

        self.pCorrSolver.sweepWestToEast()
        self.pCorrSolver.sweepSouthToNorth()
        self.pCorrSolver.sweepEastToWest()

        # Correct fields
        self.correct_u()
        self.correct_v()
        self.correct_p()

    def run(self, max_iterations: int = 10000, verbose: bool = True) -> int:
        """Run SIMPLE iterations until convergence or max_iterations is hit."""
        for it in range(max_iterations):
            if verbose:
                print(f"Iteration {it}")

            self.iterate()

            if self.has_converged():
                if verbose:
                    print(f"Converged in {it + 1} iterations.")
                return it + 1

        if verbose:
            print(
                f"WARNING: did not converge in {max_iterations} iterations "
                f"(p_res={self._p_residual():.3e}, "
                f"u_res={self._velocity_residual('u'):.3e}, "
                f"v_res={self._velocity_residual('v'):.3e})"
            )
        return max_iterations
    
    def contour_u(self) -> None:
        ny, nx = self.u.field.shape
        x = np.linspace(0.0, self.grid.Lx, nx)
        y = np.linspace(0.0, self.grid.Ly, ny)
        X, Y = np.meshgrid(x, y)
        plt.figure()
        cs = plt.contourf(Y, X, self.u.field, levels=50, cmap="plasma")
        plt.colorbar(cs, label="u (m/s)")
        plt.title("u-velocity")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()




# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import postProcess
    
    cavity = SimpleLidDrivenCavity(
        Nx=20,
        Ny=20,
        Lx=1.0,
        Ly=1.0,
        density=998.3,
        viscosity=1.002e-3,
        Re=100.0,
    )

    cavity.contour_u()

    cavity.run(max_iterations=10000, verbose=True)

    cavity.contour_u()
    