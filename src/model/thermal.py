from src.model.ecm import Thevenin1RC


class Lumped:
    def __init__(self, ecm_obj):
        if not isinstance(ecm_obj, Thevenin1RC):
            raise TypeError("ecm_obj needs to be a Thevenin 1 RC object.")
        self.ecm_obj = ecm_obj

    def reversible_heat(self, I, T):
        return I * T * (self.b_cell.elec_p.dOCPdT - self.b_cell.elec_n.dOCPdT)

    def irreversible_heat(self, I, V):
        return I * (V - (self.b_cell.elec_p.OCP - self.b_cell.elec_n.OCP))

    def heat_flux(self, T):
        return self.b_cell.h * self.b_cell.A * (T - self.b_cell.T_amb)

    def heat_balance(self, V, I):
        def func_heat_balance(T, t):
            main_coeff = 1 / (self.b_cell.rho * self.b_cell.Vol * self.b_cell.C_p)
            return main_coeff * (self.reversible_heat(I=I, T=T) + self.irreversible_heat(I=I, V=V) - self.heat_flux(T=T))
        return func_heat_balance
