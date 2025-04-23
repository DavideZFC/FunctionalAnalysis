import numpy as np

class function:
    def __init__(self, N, a=0, b=1, x=None):
        self.N = N
        if x is not None:
            self.x = x
        else:
            self.x = np.linspace(a,b,N)

    def forma(self, f):
        self.y = f(self.x)

    def norm2(self):
        return np.mean(self.y**2)**0.5
    
    def norm1(self):
        return np.mean(np.abs(self.y))
    
    def norminf(self):
        return np.max(np.abs(self.y))
    
    def normalize(self):
        self.y = self.y/self.norm2()
    
    def __add__(self, other):
        new_f = function(self.N, x=self.x)
        new_f.y = self.y + other.y
        return new_f
    
    def multiply(self, c):
        new_f = function(self.N, x=self.x)
        new_f.y = c*self.y
        return new_f
    
    def __mul__(self, other):
        return np.dot(self.y, other.y)/self.N
    
    def project(self, other):
        # self è la functione che si proietta, other quella su cui si proietta
        scal = (self*other)/(other*other)
        return other.multiply(scal)
    
    def residual(self, other):
        # residual of the projection of self over other
        return self + self.project(other).multiply(-1)       
        
        
    
