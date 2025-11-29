import numpy as np


debugging = False

class coeff(object):

	#This class is a place holder that improves readability.
	def __init__(self, n = None, s = None, w = None, e = None , p =  None , b = None):
		self.n = n
		self.s = s
		self.w = w
		self.e = e

		self.p = p

		self.b = b


class variable(object):
	"""docstring for field"""
	def __init__(
			self, 
			name : str, 
			Nx: int, 
			Ny: int, 
			top : float, 
			bottom : float, 
			left : float, 
			right : float, 
			alpha : float = 1.0, 
			tolerance : float = 1e-5):
		
		self.name = name

		self.Nx = Nx
		self.Ny = Ny

		# Discretisation coefficients a_N, a_S, a_W, a_E, a_P, b
		zeros = np.zeros((Nx, Ny))
		self.a = coeff(
		    n=zeros.copy(),
		    s=zeros.copy(),
		    w=zeros.copy(),
		    e=zeros.copy(),
		    p=zeros.copy(),
		    b=zeros.copy()
		)

		self.alpha = alpha
		self.tolerance = tolerance

		# Define coefficient matrix for the variables to be solved
		#self.a = coeff(np.zeros((Nx,Ny)) , np.zeros((Nx,Ny)) , np.zeros((Nx,Ny)) , np.zeros((Nx,Ny)), np.zeros((Nx,Ny)), np.zeros((Nx,Ny)))
		
		self.iMax = Nx + 1
		self.jMax = Ny + 1
		self.field = np.zeros((Nx+2,Ny+2))

		# Initial boundary values
		self.top = top
		self.bottom = bottom
		self.left = left
		self.right = right

	def initialize(self, value = None):
		Nx = self.Nx
		Ny = self.Ny

		if value != None:
			self.field = np.ones((Nx + 2, Ny + 2))*value
		
		self.field[Nx + 1, :] = np.ones((Ny + 2))*self.right
		self.field[0 , :] = np.ones((Ny + 2))*self.left
		self.field[:, Ny + 1] = np.ones((Nx + 2))*self.top
		self.field[:, 0] = np.ones((Nx + 2))*self.bottom
		
		# This will be handled by functions		
		# Place-holders for the line-by-line procedure
		# self.Axs = [None]*Ny
		# self.Ays = [None]*Nx
		# self.bxs = [None]*Ny
		# self.bys = [None]*Nx
	
	# ------------------------------------------------------------------
    # Coefficient assembly for cell (i, j) in coefficient arrays
    # ------------------------------------------------------------------
	def buildCoefficients(self, a_n, a_s, a_w, a_e, sourceTerm, i, j):
		"""
		Store discretisation coefficients for a given cell (i, j) in the
		coefficient arrays (which are shape (Nx, Ny)).

		Note: indices (i, j) here are 0-based in the coefficient arrays,
		corresponding to (i+1, j+1) in self.field including ghost cells.
		"""
		self.a.e[i, j] = a_e
		self.a.w[i, j] = a_w
		self.a.s[i, j] = a_s
		self.a.n[i, j] = a_n

		a_p = a_e + a_w + a_n + a_s
		self.a.p[i, j] = a_p

		self.a.b[i, j] = sourceTerm

	def __getitem__(self, index):
		return self.field[index]

	def __setitem__(self,index,value):
		self.field[index] = value
		return None

	
	def __str__(self):
		

		fmt = "{:10.4f}"
		nx, ny = self.field.shape  # nx = i-direction, ny = j-direction

		lines = [f"Field '{self.name}' (shape={self.field.shape}):"]

		for i in range(nx):
			line = "".join(fmt.format(self.field[i, j]) for j in range(ny))
			lines.append(line)

		return "\n".join(lines)
			
	

		if debugging == True:
			print(self.name , self.field )


if __name__ == "__main__":


	testField = variable("testField", 10, 8 , top=1.0, bottom=0.0, left=-1.0, right=2.0)

	print(testField)
	print("Nx, Ny:", testField.Nx, testField.Ny)
	print("iMax, jMax:", testField.iMax, testField.jMax)
	print("field.shape:", testField.field.shape)
	print("alpha, tol:", testField.alpha, testField.tolerance)
	print("Boundary values (top, bottom, left, right):", testField.top, testField.bottom, testField.left, testField.right)