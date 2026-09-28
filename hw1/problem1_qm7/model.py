import numpy as np

class Module:
	def update(self,lr): pass
	def average(self,nn,a): pass
	def backward(self,DY): pass
	def forward(self,X): pass

class Sequential(Module):

	def __init__(self,modules):
		self.modules = modules
	
	def forward(self,X):
		'''Given input X, perform a full forward pass through the MLP'''
		for module in self.modules:
			X = module.forward(X) #or just module(X) if module had
			#def __call__(self, X):
        	#	return self.forward(X)

		return X

	
	def backward(self,DY): #for all layers (with learnable parameters, otherwise the inherited `Module` function will `pass`)
		'''Perform a full backward pass through the MLP. 
			DY is gradient of the loss w.r.t the final output'''
		grad = DY
		for module in reversed(self.modules):
			grad = module.backward(grad)

		return grad
		
	def update(self,lr): #after gradients are computed, update parameters for all layers (with learnable parameters, otherwise the inherited `Module` function will `pass`)
		for m in self.modules: 
			m.update(lr)
		
	def average(self,nn,a):
	#average with another network with same architecture, e.g.:
	# 		model:          Linear → Tanh → Linear → Tanh → Linear
	# 		average_model:  Linear → Tanh → Linear → Tanh → Linear

	#and m = average_model.modules[...]
		#n = model.modules[...]

		#average_model.average(model, 0.1)
		for m,n in zip(self.modules,nn.modules): 
			m.average(n,a) #works just for `Linear`, others have `pass`

class Input(Module): #no backward or update functions since no learnable params
	def __init__(self, inp):
		R, Z = inp
		sample_in = np.concatenate([R, np.expand_dims(Z, -1)], axis = -1)
		#concatenation between (B, N, 3) and (B, N, 1) to obtain (B, N, 4)
		self.nbout = sample_in.shape[-2] * sample_in.shape[-1] # N * 4

	def forward(self,inp): 
		R, Z = inp
		rz = np.concatenate([R, np.expand_dims(Z, -1)], axis = -1)
		return rz.reshape(rz.shape[0], -1) #(B, N * 4)

class Output(Module):

	def __init__(self,T):
		#T are the training targets (energies)
		self.tmean = T.mean()
		self.tstd  = T.std()
		self.nbinp = 1

	def forward(self,X):
		# un-normalize the final prediction 
		self.input_shape = X.shape #(B, 1)
		self.X = X.flatten()
		return self.X*self.tstd+self.tmean

	def backward(self,DY):
		DX = DY * self.tstd

		return DX.reshape(self.input_shape) #(B, 1) to match what fcl3 expects (a DY -- this backward output -- of (B, 1))
	
# Output.backward:
# (B,)
#  ↓ reshape
# (B, 1)

# last Linear.backward:
# DY        (B, 1)
# W.T       (1, 100)

# (B, 1) @ (1, 100)
#        ↓
#     (B, 100)

class Linear(Module):

	def __init__(self,m,n):
		"""
		m: input features
		n: output features

		xavier initialization --> np.rand/sqrt(m) so var(X) = var(XW)
		"""
		self.lr = 1 / m**.5
		self.W = np.random.normal(0,1 / m**.5,[m,n]).astype('float32')
		self.B = np.zeros([n]).astype('float32')
		

	def forward(self,X):
		self.X = X
		self.res = X @ self.W + self.B

		return self.res 
	
	def backward(self,DY):
		"""
		X = (B, m)
		W =  (m, n)
		B = (n,)
		Y = XW + B = (B, n)
		--> DY = (B, n)
		"""
		DX = DY @ self.W.T #(B, m)
		self.DW = self.X.T @ DY #(m, n)
		self.DB = DY.sum(axis = 0) #(n,)

		return DX

	def update(self,lr):
		self .W -= lr*self.lr*self.DW
		self.B -= lr*self.lr*self.DB

	def average(self,nn,a):
		self.W = a*nn.W + (1-a)*self.W
		self.B = a*nn.B + (1-a)*self.B

class Tanh(Module):
	
	def forward(self,X):
		"""
		X: (B, N, 4)
		"""
		#self.res = (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))
		self.res = np.tanh(X)

		return self.res 
		
	def backward(self,DY):
		Dtan = 1 - np.square(self.res)
		self.DX = DY * Dtan

		return self.DX  

