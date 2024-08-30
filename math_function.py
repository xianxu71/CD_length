import numpy as np

def delta_lorentzian(x, xc, eta):

    return (eta / np.pi) / ( (x-xc) ** 2 + eta ** 2)

def delta_gauss(x,xc,eta):
    return np.exp(-(x-xc)**2/(2*(eta)**2))/(np.sqrt(2*np.pi)*eta)

def calculate_reflectivity(eps1,eps2):
    a = np.sqrt(0.5*(eps1+np.sqrt(eps1**2+eps2**2)))
    b = np.sqrt(0.5*(-eps1+np.sqrt(eps1**2+eps2**2)))
    R = ((a-1)**2+b**2)/((a+1)**2+b**2)
    return R