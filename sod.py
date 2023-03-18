import math
p_star = 0
p_i = 0
rho_i = 0
gamma = 1.4

def  F(p_star, p_i, rho_i):
    c_i = math.sqrt(gamma * p_i / rho_i)
    if p_star > p_i:
        result = (p_star - p_i)/(rho_i*c_i*math.sqrt((gamma + 1)*p_star/(gamma*2*p_i)+(gamma-1)/2/gamma))
    else:
        result = 2 * c_i / (gamma - 1) * (math.pow(p_star/p_i, (gamma - 1)/2/gamma) - 1)
    return result

print(F(0, 0, 0))
