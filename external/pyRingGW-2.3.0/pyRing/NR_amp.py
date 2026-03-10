from numpy import sqrt,exp

class Amp_KHS:
    
    """
        QNM real amplitudes fitting functions for remnant BHs resulting from the coalescence of two non-precessing progenitors in a quasi-circular orbit. References: https://arxiv.org/abs/1207.0399, https://arxiv.org/abs/1406.3201.
    """
    
    
    def __init__(self, eta, chi_eff):
        
        self.eta     = eta
        self.chi_eff = chi_eff

        self.amps = {
                        (2, 2): self.eta*0.864,
                        (2, 1): self.eta*0.864*0.43*(sqrt(1.-4.*self.eta) - self.chi_eff),
                        (3, 3): self.eta*0.864*0.44*pow((1.-4.*self.eta), 0.45),
                        (4, 4): self.eta*0.864*(0.04 + 5.4*(self.eta - 0.22)*(self.eta - 0.22))
                    }

class Amp_MMRDNP:
    
    """
        QNM complex amplitudes fitting functions for remnant BHs resulting from the coalescence of two non-precessing progenitors in a quasi-circular orbit. Reference: https://arxiv.org/abs/1801.08208.
    """
    
    # Overtones are NOT to be used (Lionel personal communication)
    
    def __init__(self, eta, chi_s, chi_a, delta):
        
        self.eta   = eta
        self.chi_s = chi_s
        self.chi_a = chi_a
        self.delta = delta

        self.amps = {
                (2, 2):
                {
                    (2, 2, 1): self.eta*(56.959224306037 * exp(0.249870658757j) * (self.eta)               +
                                         13.358851401577 * exp(5.278844614593j) * (self.chi_s*self.chi_s)  +
                                         68.627334544380 * exp(5.777931269583j) * (self.eta*self.chi_s)    +
                                         13.272497293173 * exp(1.575473940868j) * (self.delta*self.delta)  +
                                         7.037699086164  * exp(6.228126238366j) * (self.chi_a*self.delta)  ),
                    
                    (2, 2, 0): self.eta*(-0.653712262068 * (self.chi_s) +
                                         -4.007147321891                )
                },
                (2, 1):
                {
                    (2, 1, 0): self.eta*(2.348760040855 * exp(2.663056033439j) * (self.delta)            +
                                         0.801085189044 * exp(5.706986123489j) * (self.chi_a)            +
                                         3.582754813634 * exp(5.522326756943j) * (self.eta*self.delta)   +
                                         1.177427060772 * exp(0.425407431079j) * (self.chi_s*self.delta) +
                                         0.626021476259 * exp(5.345654928116j) * (self.chi_s*self.chi_a) )
                },
                (3, 3):
                {
                    (3, 3, 0): self.eta*(2.641151533089 * exp(2.988026867825j) * (self.delta)            +
                                         1.603029394414 * exp(0.665544501584j) * (self.delta*self.delta) +
                                         1.035429991023 * exp(3.609595591592j) * (self.chi_s*self.delta) +
                                         0.491081370987 * exp(4.734657474326j) * (self.chi_a*self.chi_a) ),

                    (3, 3, 1): self.eta*(33.435799877434  * exp(5.705209058888j) * (self.delta)            +
                                         60.712322561545  * exp(2.574187334609j) * (self.chi_a)            +
                                         252.183816522714 * exp(5.673332115767j) * (self.eta*self.chi_a)   +
                                         1.638725482466   * exp(5.435206166400j) * (self.chi_s*self.chi_s) +
                                         46.583291557097  * exp(5.773086185765j) * (self.chi_a*self.delta) +
                                         23.109132218151  * exp(2.532816456662j) * (self.delta*self.delta) +
                                         67.785254118884  * exp(2.423649846906j) * (self.eta*self.delta)   )
                },
                (3, 2):
                {
                    (3, 2, 0): self.eta*(2.570691974815 * exp(4.142699014294j) * (self.eta)              +
                                         9.421592014408 * exp(0.807614972952j) * (self.eta*self.eta)     +
                                         0.597281423027 * exp(2.181641820050j) * (self.eta*self.chi_s)   +
                                         0.210421544048 * exp(4.904320363371j) * (self.chi_a*self.chi_a) +
                                         0.441666578823 * exp(5.454386699537j) * (self.chi_a*self.delta) +
                                         0.943921938215 * exp(1.761351963620j) * (self.delta*self.delta) ),
                    
                    (2, 2, 0): self.eta*(1.340733486020 * exp(2.946590526436j) * (self.eta)              +
                                         0.071690697284 * exp(5.530356938749j)                           +
                                         0.106061493718 * exp(2.643212390641j) * (self.chi_s*self.chi_s) +
                                         0.989356860932 * exp(2.929356011991j) * (self.eta*self.chi_s)   +
                                         0.373522106106 * exp(3.329029215606j) * (self.chi_a*self.delta) )
                },
                (4, 4):
                {
                    (4, 4, 0): self.eta*(1.328400699459  * exp(2.683129882379j) * (self.delta*self.delta)                       +
                                         1.161895660558  * exp(0.414239904649j) * (self.delta*self.delta*self.delta)            +
                                         1.279028235412  * exp(4.722648678220j) * (self.chi_s*self.chi_a*self.chi_a*self.delta) +
                                         1.238681601988  * exp(4.561572066843j) * (self.chi_s*self.chi_a*self.chi_a*self.chi_a) +
                                         1.290934476699  * exp(2.812028014439j) * (self.chi_s*self.delta*self.delta*self.delta) +
                                         42.357472914273 * exp(6.141845955070j) * (self.eta*self.eta*self.eta*self.eta)         )
                },
                (4, 3):
                {
                    (3, 3, 0): self.eta*(0.041123756515 * exp(2.644070703232j) * (self.chi_a)            +
                                         0.048619374603 * exp(3.208492542783j) * (self.chi_s*self.chi_s) +
                                         0.807790100431 * exp(2.746110940128j) * (self.eta*self.delta)   +
                                         0.194011841224 * exp(3.029203678324j) * (self.chi_s*self.delta) +
                                         0.052877836741 * exp(3.583020069562j) * (self.chi_a*self.chi_a) +
                                         0.035764383445 * exp(0.173129332737j) * (self.delta*self.delta) ),

                    (4, 3, 0): self.eta*(0.566500061188  * exp(3.399184539111j) * (self.delta)                       +
                                         0.145653197976  * exp(4.747575463481j) * (self.chi_a)                       +
                                         0.823925735345  * exp(1.817382535089j) * (self.eta*self.chi_a)              +
                                         0.050668406375  * exp(4.749537130684j) * (self.chi_s*self.chi_a)            +
                                         0.980612981215  * exp(0.602860611615j) * (self.delta*self.delta*self.delta) +
                                         10.167779687712 * exp(6.218461358623j) * (self.eta*self.eta*self.delta)     )
                }
            }


spherical_multipoles_list_MMRDNP  = [(2, 2), (2, -2),
                                     (2, 1), (2, -1),
                                     (3, 3), (3, -3),
                                     (3, 2), (3, -2),
                                     (4, 4), (4, -4),
                                     (4, 3), (4, -3)]

spheroidal_multipoles_list_MMRDNP = [(2, 2, 0), (2, -2, 0),
                                     (2, 2, 1), (2, -2, 1),
                                     (3, 3, 0), (3, -3, 0),
                                     (3, 3, 1), (3, -3, 1),
                                     (3, 2, 0), (3, -2, 0),
                                     (4, 4, 0), (4, -4, 0),
                                     (4, 3, 0), (4, -3, 0)]


# NOTE that these functions give QNM amplitudes in STRAIN.

class Amp_MMRDNS:
    
    """
    QNM complex amplitudes fitting functions for remnant BHs resulting from the coalescence of two non-spinning progenitors in a quasi-circular orbit. Reference: https://github.com/llondon6/kerr_public/blob/master/kerr/formula/mmrdns_amplitudes.py (the ones in the original publication https://arxiv.org/abs/1404.3197 are not up to date).
    """
    
    def __init__(self, eta):
        
        self.eta = eta


        self.amps = {
                        (2,2,0):                  (  0.95846504*exp(2.99318408*1j)*self.eta + 0.47588079*exp(0.82658128*1j)*(self.eta**2) + 1.23853419*exp(2.30528861*1j)*(self.eta**3)  ),

                        (2,2,1):                  (  0.12750415*exp(0.05809736*1j)*self.eta + 1.18823931*exp(1.51798243*1j)*(self.eta**2) + 8.27086561*exp(4.42014780*1j)*(self.eta**3) + 26.23294960*exp(1.16782950*1j)*(self.eta**4)  ),

                        (3,2,0):                  (  0.19573228*exp(0.54325509*1j)*self.eta + 1.58299638*exp(4.24509590*1j)*(self.eta**2) + 5.03380859*exp(1.71003281*1j)*(self.eta**3) + 3.73662711*exp(5.14735754*1j)*(self.eta**4)  ),

                        (4,4,0):                  (  0.25309908*exp(5.16320109*1j)*self.eta + 2.40404787*exp(2.46899414*1j)*(self.eta**2) + 14.72733952*exp(5.56235208*1j)*(self.eta**3) + 67.36237809*exp(2.19824119*1j)*(self.eta**4) + 126.58579931*exp(5.41735031*1j)*(self.eta**5)  ),
                        (2,1,0): sqrt(1-4*self.eta)  * (  0.47952344*exp(5.96556090*1j)*self.eta + 1.17357614*exp(3.97472217*1j)*(self.eta**2) + 1.23033028*exp(2.17322465*1j)*(self.eta**3)  ),
                        
                        (3,3,0): sqrt(1-4*self.eta)  * (  0.42472339*exp(4.54734400*1j)*self.eta + 1.47423728*exp(2.70187807*1j)*(self.eta**2) + 4.31385024*exp(5.12815819*1j)*(self.eta**3) + 15.72642073*exp(2.25473854*1j)*(self.eta**4)  ),

                        (3,3,1): sqrt(1-4*self.eta)  * (  0.14797161*exp(2.03957081*1j)*self.eta + 1.48738894*exp(5.89538621*1j)*(self.eta**2) + 10.16366839*exp(3.28354928*1j)*(self.eta**3) + 29.47859659*exp(0.81061521*1j)*(self.eta**4)  ),

                        (4,3,0): sqrt(1-4*self.eta)  * (  0.09383417*exp(2.30765661*1j)*self.eta + 0.82734483*exp(6.10053234*1j)*(self.eta**2) + 3.33846327*exp(3.87329126*1j)*(self.eta**3) + 4.66386840*exp(1.75165690*1j)*(self.eta**4)  ),

                        (5,5,0): sqrt(1-4*self.eta)  * (  0.15477314*exp(1.06752431*1j)*self.eta + 1.50914172*exp(4.54983062*1j)*(self.eta**2) + 8.93331690*exp(1.28981042*1j)*(self.eta**3) + 42.34309620*exp(4.10035598*1j)*(self.eta**4) + 89.19466498*exp(1.02508947*1j)*(self.eta**5)  )
                    }

spheroidal_multipoles_list_MMRDNS = [(2, 2, 0), (2, -2, 0),
                                     (2, 1, 0), (2, -1, 0),
                                     (2, 2, 1), (2, -2, 1),
                                     (3, 3, 0), (3, -3, 0),
                                     (3, 3, 1), (3, -3, 1),
                                     (3, 2, 0), (3, -2, 0),
                                     (4, 4, 0), (4, -4, 0),
                                     (4, 3, 0), (4, -3, 0),
                                     (5, 5, 0), (5, -5, 0)]

