# grid.py
class Grid2D:
    """
    Simple 2D uniform grid.

    It stores the number of control volumes for the *pressure* field.
    Staggered u and v velocities use (Nx-1, Ny) and (Nx, Ny-1) respectively.
    """
    def __init__(self, Nx: int, Ny: int, Lx: float, Ly: float) -> None:
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly

        self.Deltax = Lx / Nx
        self.Deltay = Ly / Ny
