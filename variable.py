import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import ticker
import copy
import math

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
	def __init__(self, name : str  , Nx, Ny , top, bottom, left, right , alpha = 1.0 , tolerance = 1e-5):
		
		self.name = name

		self.Nx = Nx
		self.Ny = Ny

		self.alpha = alpha
		self.tolerance = tolerance

		# Define coefficient matrix for the variables to be solved
		self.a = coeff(np.zeros((Nx,Ny)) , np.zeros((Nx,Ny)) , np.zeros((Nx,Ny)) , np.zeros((Nx,Ny)), np.zeros((Nx,Ny)), np.zeros((Nx,Ny)))
		
		self.iMax = Nx + 1
		self.jMax = Ny + 1
		self.field = np.zeros((Nx+2,Ny+2))


		self.top = top
		self.bottom = bottom
		self.left = left
		self.right = right
		
		# Place-holders for the line-by-line procedure
		self.Axs = [None]*Ny
		self.Ays = [None]*Nx
		self.bxs = [None]*Ny
		self.bys = [None]*Nx



			 


	def __getitem__(self, index):
		return self.field[index]

	def __setitem__(self,index,value):
		self.field[index] = value
		return None

	def initialize(self, value = None):
		Nx = self.Nx
		Ny = self.Ny

		if value != None:
			self.field = np.ones((Nx + 2, Ny + 2))*value
		
		self.field[Nx + 1, :] = np.ones((Ny + 2))*self.right
		self.field[0 , :] = np.ones((Ny + 2))*self.left
		self.field[:, Ny + 1] = np.ones((Nx + 2))*self.top
		self.field[:, 0] = np.ones((Nx + 2))*self.bottom
		

		if debugging == True:
			print(self.name , self.field )


	def buildCoefficients(self, a_n , a_s , a_w , a_e , sourceTerm , i , j):

		self.a.e[i,j] = a_e
		self.a.w[i,j] = a_w
		self.a.s[i,j] = a_s
		self.a.n[i,j] = a_n

		a_p = a_e + a_w + a_n + a_s
		self.a.p[i,j] = a_p


		self.a.b[i,j] = sourceTerm

		

	def buildRow(self,j):

		a = self.a
		field = self.field

		# Pressure correction is relaxed at correction
		if self.name == "pCorr":
			alpha = 1.0
		else:
			alpha = self.alpha

		# Number of variable on each direction
		Nx = self.Nx
		Ny = self.Ny




		# The b array of this variable needs to be updated at each row and then stored in bxs
		Ax = np.zeros((Nx,Nx))

		# a.b contains explicit source terms
		bx = copy.deepcopy(a.b[:,j])

		if debugging==True:
			print("b at row  in buildRow before considering bc" , j)
			print()
			print(bx)
			

		Ax[0,0] = a.p[0,j]/alpha
		Ax[0,1] = -a.e[0,j]
		

		Ax[Nx-1,Nx-1] = a.p[Nx-1,j]/alpha
		Ax[Nx-1,Nx-2] = -a.w[Nx-1,j]

		for l in range(1,Nx-1):
			for k in range(Nx):
				if k == l - 1:
					Ax[l,k] = -a.w[l,j]
				if k == l + 1:
					Ax[l,k] = -a.e[l,j]
				if k == l :
					Ax[l,k] = a.p[l,j]/alpha

		# EAST AND WEST CONTRIBUTION FROM BOUNDARY CONDITIONS
		if (self.name != 'pCorr'):
			
			bx[0] = bx[0] + field[0,j+1]*a.w[0,j]		
			bx[Nx-1] = bx[Nx - 1] +  field[ Nx + 1, j + 1 ]*a.e[Nx-1,j]
			
			

		# NORTH AND SOUTH CONTRIBUTIONS FROM BOUNDARY CONDITIONS
		
		if j == Ny-1:
			for i in range(Nx):
				bx[i] = bx[i] + field[i + 1 , Ny + 1]*a.n[i,j]
					
						

		
		if j ==  0:
			for i in range(Nx):
				if debugging==True:
						
					print("field[i + 1, j] ," , field[i + 1, j])
					print("a.s[i,j] ," , a.s[i,j])
					print("field[i + 1, j]*a.s[i,j] " , field[i + 1, j]*a.s[i,j])
				
				bx[i] = bx[i] + field[i + 1, j]*a.s[i,j]
					

		
		if debugging == True:
			print("Ax for " , self.name,  " at row " , j , " " , Ax)
			print("bx for " , self.name, " at row " , j , " ",  bx)
		
		# self.Axs.append(Ax)
		# self.bxs.append(bx)
		self.Axs[j] = Ax
		self.bxs[j] = bx
		

		



		

	def buildColumn(self,i):


		a = self.a
		field = self.field

		if self.name == "pCorr":
			alpha = 1.0
		else:
			alpha = self.alpha
		# Number of variable on each direction
		Nx = self.Nx
		Ny = self.Ny

		Ay = np.zeros((Ny,Ny))
		by = copy.deepcopy(a.b[i,:])


		Ay[0,0] = a.p[i,0]/alpha
		Ay[0,1] = -a.n[i,0]
		

		Ay[Ny-1,Ny-1] = a.p[i, Ny-1]/alpha
		Ay[Ny-1,Ny-2] = -a.s[i, Ny-1]



		for l in range(1,Ny-1):
			for k in range(Ny):
				if k == l - 1:
					Ay[l,k] = -a.s[i,l]
				if k == l + 1:
					Ay[l,k] = -a.n[i,l]
				if k == l:
					Ay[l,k] = a.p[i,l]/alpha
					
		if self.name != 'pCorr':
			by[0] = by[0] + field[i+1, 0]*a.s[i,0]
			by[Ny-1] = by[Ny-1] + field[i+1, Ny + 1]*a.n[i,Ny-1]
			
			
		if (i == Nx-1 ):
			for j in range(Ny):
				by[j] = by[j] + field[Nx + 1, j + 1] * a.e[i,j]
				
		if (i ==  0 ):
			for j in range(Ny):
				by[j] = by[j] + field[0 , j + 1 ] * a.w[i,j]

		if debugging == True:
			print("Ay for " , self.name, " at column " , i , " " , Ay)
			print("by for " , self.name, " at column " , i , " ",  by)

		# self.Ays.append(Ay)
		# self.bys.append(by)
		self.Ays[i] = Ay
		self.bys[i] = by

		if debugging==True:
			
			print("Ays " , np.shape(self.Ays))


	# def preCompute(self):
		
	# 	# Fill matrices and vectors before solving

	# 	for i in range(self.Ny):
	# 		self.buildRow(i)

	# 		if debugging == True:
	# 			print("precomputed Ax ", self.Axs[i])
	# 			print("precomputed bx ", self.bxs[i])
	# 			print()
			
			

	# 	for i in range(self.Nx):
	# 		self.buildColumn(i)
			
	# 		if debugging == True:
	# 			print("precomputed Ay ", self.Ays[i])
	# 			print("precomputed by ", self.bys[i])
	# 			print()

		


	def solveRow(self,row):
    	
		a = self.a

		Nx = self.Nx
		Ny = self.Ny

		
		if self.name == "pCorr":
			alpha = 1.0
		else:
			alpha = self.alpha

		A = self.Axs[row]



		b = copy.deepcopy(self.bxs[row])
		
		if debugging==True:
			print(" before filling for each cell ")
			print()
			print("b: ", b)
			
		
		if debugging  == True:
			print("Solving row ", row, "for " , self.name)
			print()
				
		field = self.field

		if row > 0:
			for i in range(Nx):
				b[i] = b[i] + field[i + 1,row]*a.s[i,row]

				if debugging==True:
					print("i: " , i , "row: " , row)
					print("field[i + 1,row]*a.s[i,row]", field[i + 1,row]*a.s[i,row])
					print("Filling b: ", b)
					
				
				
		if row < Ny -1 :
			for i in range(Nx):
				b[i] = b[i] + field[i + 1,row+2]*a.n[i,row]

				if debugging==True:
					print("i: " , i , "row: " , row)
					print("field[i + 1,row+2] , " , field[i + 1,row+2])
					print("a.n[i,row] , ", a.n[i,row])
					print("field[i + 1,row+2]*a.n[i,row] , " , field[i + 1,row+2]*a.n[i,row])
					print("Filling b: ", b)
					
					
				

		for i in range(Nx):
			b[i] = b[i] + (1/alpha-1)*field[i + 1,row + 1]*a.p[i,row]

			if debugging==True:
				
				print("b: ", b)
		
		if debugging == True:
			print()
			print("A matrix at row " , row , ":")
			print(A)
			print("b vector at row " , row, " : ")
			print(b)
			print()
		
		field[1:self.iMax,row + 1] = self.TDMAsolver(A,b)
		
		if debugging==True:
			print("solution at row " , row , field[1:self.iMax,row + 1])
				


		# b_ = copy.deepcopy(b)
		# self.T[:,row] = np.linalg.solve(A,b_)
		
	def solveColumn(self,column):

		a = self.a

		Nx = self.Nx
		Ny = self.Ny

		if self.name == "pCorr":
			alpha = 1.0
		else:
			alpha = self.alpha


		A = self.Ays[column]
		b = copy.deepcopy(self.bys[column])

		field = self.field


		# The contributions from east and west are already considered when b is precomputed
		if column > 0:
			for i in range(Ny):
				b[i] = b[i] + field[column, i + 1]*a.w[column,i]

				
		if column < Nx -1 :
			for i in range(Ny):
				b[i] = b[i] + field[column + 2 , i + 1]*a.e[column,i]

				

		for i in range(Ny):
			b[i] = b[i] + (1/alpha-1)*field[column + 1,i + 1]*a.p[column,i]
		
	
		if debugging == True:
			print()
			print("A matrix at column " , column , ":")
			print(A)
			print("b vector at column " , column, " : ")
			print(b)
			print()
		field[column + 1, 1:self.jMax] = self.TDMAsolver(A,b)

		# self.T[column,:] = np.linalg.solve(A,b)
		if debugging==True:
			print("solution at column " , column , field[column+ 1 , 1:self.jMax])





	def sweepEastToWest(self):


		if debugging==True:
			
			print("Sweeping from east to west ... ")
			print()
			
		for column in range(self.Nx):
			self.buildColumn(column)
		
		for column in range(self.Nx-1,-1,-1):
			self.solveColumn(column)

		if debugging==True:
					
			print()
			print("Solution after sweeping east to west: ")
			print(self.field)
			print()

	def sweepWestToEast(self):

		if debugging==True:
			print("Sweeping from west to east ... ")
			print()
			
		for column in range(self.Nx):
			self.buildColumn(column)

		for column in range(self.Nx):
			self.solveColumn(column)

		if debugging==True:
			
			print()
			print("Solution after sweeping west to east: ")
			print(self.field)
			print()



	def sweepNorthToSouth(self):

		if debugging==True:
			print("Sweeping from north to south ... ")
			print()
			
		for row in range(self.Ny):
			self.buildRow(row)

		for row in range(self.Ny-1,-1,-1):
			self.solveRow(row)

		if debugging==True:

			print()
			print("Solution after sweeping North to South: ")
			print(self.field)
			print()
			

	def sweepSouthToNorth(self):

		if debugging==True:
			print("Sweeping from south to north ... ")
			print()
			
		for row in range(self.Ny):
			self.buildRow(row)		

		for row in range(self.Ny):
			self.solveRow(row)

		if debugging==True:
			
			print()
			print("Solution after sweeping South to North: ")
			print(self.field)
			print()





	def TDMAsolver(self,A,b):

		N = len(b)   
		x = np.zeros(N) # Container for the result
	

		tdmaMatrix = copy.copy(A)
		tdmaVector = copy.copy(b)   		

	        # Forward elimination:
		for i in range(1,N):
	            # Normalization of the first element of the row
			m = tdmaMatrix[i,i-1]/tdmaMatrix[i-1,i-1]

			tdmaMatrix[i,i] = tdmaMatrix[i,i] - m*tdmaMatrix[i-1,i]
	        
			tdmaVector[i] = tdmaVector[i] - m*tdmaVector[i-1]
	        
	        # Back-substitution:

		x[N-1] = tdmaVector[N-1]/tdmaMatrix[N-1,N-1]
	        
		for i in range(N-2,-1,-1):
			x[i] = 1/tdmaMatrix[i,i]*(tdmaVector[i] - tdmaMatrix[i,i+1]*x[i+1])

		
		
	        
		return x


		