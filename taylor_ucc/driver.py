from pyscf_backend import *
from contractions import *
from opt_einsum import contract
from scipy.sparse.linalg import *
from scipy.optimize import minimize
from scipy.optimize import line_search
import copy
import math
import time
import inspect
import sys
import git
class molecule():
    def __init__(self, geometry, basis, reference, charge = 0, unpaired = 0, conv_tol = 1e-12, read = False, fci = False, ccsd = False, ccsdt = False, Ca_guess = None, Cb_guess = None, loc = None, hci = False, var_only = False, eps_var = 5e-5, save = False, chkfile = None, memory = 8e3, pseudo_canonicalize = False, manual_C = None):
        start = time.time()
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        print(f"Git revision:\ngithub.com/hrgrimsl/PYSCF_UCC/commit/{sha}")
        print('Doing pyscf backend stuff...')
        self.vec_structure, self.hf, self.ci_energy, self.ccsd_energy, self.ccsdt_energy, self.hci_energy, self.Fa, self.Fb, self.Iaa, self.Iab, self.Ibb, self.noa, self.nob, self.nva, self.nvb, self.Ca, self.Cb, self.S = integrals(geometry, basis, reference, charge, unpaired, conv_tol, Ca_guess, Cb_guess, read = read, do_ci = fci, do_ccsd = ccsd, do_ccsdt = ccsdt, loc = loc, do_hci = hci, var_only = var_only, eps_var = eps_var, save = save, chkfile = chkfile, memory = memory, pseudo_canonicalize = pseudo_canonicalize, manual_C = manual_C)
        print(f'\nBackend done in {time.time()-start} seconds.\n')
        self.N = self.noa + self.nob
        self.reference = reference
        self.vec_size = 0
        self.loc = loc
        for i in self.vec_structure:
            self.vec_size += i
        x = self.tensor(np.ones(self.vec_size))
        self.Finv_count = 0
        self.F_count = 0
        self.H_count = 0
        self.ga = self.Fa[:self.noa,self.noa:]
        self.gb = self.Fb[:self.nob,self.nob:]
        self.gaa = copy.copy(self.Iaa) - contract('pqrs->pqsr', self.Iaa)
        self.gab = copy.copy(self.Iab)
        self.gbb = copy.copy(self.Ibb) - contract('pqrs->pqsr', self.Ibb) 
        self.g = self.arr([self.ga, self.gb, self.gaa[:self.noa,:self.noa, self.noa:, self.noa:], self.gab[:self.noa,:self.nob,self.noa:,self.nob:], self.gbb[:self.nob, :self.nob, self.nob:, self.nob:]])

        self.F_diag = self.F_N(np.ones(self.g.shape), diag = True)
        self.H_N_diag = self.arr(H_N_diag(self.tensor(np.ones(self.vec_size)), self))
        #H_op = LinearOperator((self.vec_size, self.vec_size), matvec = self.H_N, rmatvec = self.H_N)
        #H_N = self.build_arr(H_op)
        #print(self.H_N_diag)
        #print(np.diag(H_N))
        #exit()
        self.Finv_op = LinearOperator((self.vec_size, self.vec_size), matvec = self.Finv, rmatvec = self.Finv)


        self.M = self.Finv_op
        self.shift = 0        

        self.n_aa = self.noa*self.nva
        self.n_bb = self.nob*self.nvb
        self.n_aaaa = int(.25*self.noa*(self.noa-1)*self.nva*(self.nva-1))
        self.n_bbbb = int(.25*self.nob*(self.nob-1)*self.nvb*(self.nvb-1))
        self.n_abab = int(self.noa*self.nob*self.nva*self.nvb)
        self.n_singles = self.n_aa + self.n_bb
        self.n_doubles = self.n_aaaa + self.n_bbbb +self.n_abab
        assert len(self.g) == self.n_singles + self.n_doubles
        print(f"Reference Energy (a.u.):  {self.hf:16.12}") 
        print(f"Single excitations:       {self.n_singles:16d}")
        print(f"Alpha:                    {self.n_aa:16d}")
        print(f"Beta:                     {self.n_aa:16d}")
        print(f"Double excitations:       {self.n_doubles:16d}")
        print(f"aaaa:                     {self.n_aaaa:16d}")
        print(f"abab:                     {self.n_abab:16d}")
        print(f"bbbb:                     {self.n_bbbb:16d}")
    
    def save_H(self, filename):
        Aop = LinearOperator((self.vec_size, self.vec_size), matvec = self.H_N, rmatvec = self.H_N)
        H = self.build_arr(Aop)
        np.save(filename+"_H", H)
        np.save(filename+"_g", self.g)
        np.save(filename+"_E", self.hf)
         
    def lccsd(self, tol = 1e-6):
        #Defined as the 2nd-order approximation to CISD, including disconnected terms.  Differs from both LinCCSD and Lin\Lambda CCSD as defined by Taube and Bartlett (i.e 2nd-order
        
        self.times = [time.time()]
        print("\n***LCCSD***\n")
        print("Conjugate Gradient Convergence:\n")
        self.Finv_count = 0
        self.H_count = 0
        self.iteration = 0
        Aop = LinearOperator((self.vec_size, self.vec_size), matvec = self.H_N, rmatvec = self.H_N)
        print(f" CG Iter |   Energy (a.u.)   | Norm of Residual | Iteration Time |")
        self.times.append(time.time())
        x, info = cg(Aop, -self.g, tol = tol, M = self.Finv_op, callback = self.lccsd_cb) 
        self.times.append(time.time())
        assert info == 0
        energy = self.hf + self.g.T.dot(x)
        print("\n")
        print(f"CG Iterations:                   {self.iteration:18d}")
        print(f"Number of Hamiltonian actions:   {self.H_count:18d}")
        print(f"Number of inv(F_diag) actions:   {self.Finv_count:18d}")
        print(f"Converged LCCSD Energy (a.u.):   {self.hf + self.g.T.dot(x):18.8f}")
        print(f"Norm of residual:                {self.resid_norm:18.5e}")
        print(f"LCCSD Norm of solution:          {np.linalg.norm(x):18.5e}")
        t1a, t1b, t2aa, t2ab, t2bb = self.tensor(x)
        t1_norm = np.linalg.norm(x[:self.n_singles])
        t2_norm = np.linalg.norm(x[self.n_singles:])
        t1_diagnostic = t1_norm/np.sqrt(2*self.N)
        d1 = np.linalg.norm(t1a)
        print(f"LCCSD T1 Norm:                   {t1_norm:20.8e}")
        print(f"LCCSD T1 Diagnostic:             {t1_diagnostic:20.8e}")
        print(f"LCCSD D1 Diagnostic:             {d1:20.8e}")
        print(f"LCCSD T2 Norm:                   {t2_norm:20.8e}")
        idx = np.argsort(abs(x))         
        print(f"Largest amplitude:               {x[idx][-1]:18.5e}")
        print(f"Average iteration time (s):      {(self.times[-1]-self.times[1])/self.iteration:18.8f}")
        print(f"Total time elapsed (s):          {time.time()-self.times[0]:18.8f}") 
        return energy, x

    def lccsd_cb(self, x):
        self.iteration += 1
        self.times.append(time.time())
        energy = self.hf + self.g.T.dot(x)
        frame = inspect.currentframe().f_back
        resid_norm = np.linalg.norm(frame.f_locals['resid']) 
        print(f"{self.iteration:8d} | {energy:16.10f}  | {resid_norm:16.5e} | {self.times[-1] - self.times[-2]:14.2f} |")
        self.resid_norm = resid_norm
        return energy
    
    def enucc_cb(self, x):
        self.iteration += 1
        self.times.append(time.time())
        energy = self.hf + self.g.T.dot(x) + (2/3)*np.multiply(self.g,x).T.dot(np.multiply(x,x))
        frame = inspect.currentframe().f_back
        resid_norm = np.linalg.norm(frame.f_locals['resid']) 
        print(f"{self.iteration:8d} | {energy:16.10f}  | {resid_norm:16.5e} | {self.times[-1] - self.times[-2]:14.2f} |")
        self.resid_norm = resid_norm
        return energy

    def UCCSD_2_A(self, x):
        if self.trotter == 'sd':
            return self.UCCSD2_H_N_no_ov(x)
        else:
            return self.UCCSD2_H_N(x)

        
    def UCC_2_HATER3_energy(self, x):
        E = self.hf + 2*self.g.T.dot(x) + x.T@(self.UCCSD_2_A(x)) - (4/3)*self.g.T@(x*x*x)
        return E

    def UCC_2_HATER3_grad(self, x):
        grad = 2*self.g + 2*self.UCCSD_2_A(x) - 4*self.g*x*x
        return grad
   
    def UCC_2_HATER3_hess_action(self, x, p):
        hess = 2*self.UCCSD_2_A(p) - 8*self.g*x*p
        return hess
 
    def UCC_2_HATER3_cb(self, x):
        self.iteration += 1
        self.times.append(time.time())
        energy = self.UCC_2_HATER3_energy(x)
        grad = self.UCC_2_HATER3_grad(x)
        print(f"\nIteration: {self.iteration}", flush = True)
        print(f"Energy: {energy}", flush = True)
        print(f"Gradient norm: {np.linalg.norm(grad)}", flush = True)
        return energy

    def UCC_2_HATER3(self, guess = 'hf', trotter = False, tol = 1e-5):
        self.trotter = copy.copy(trotter)
        self.iteration = 0
        self.times = [time.time()]
        print("UCC_2 + HATER3:")
        if guess == 'hf':
            x = 0*self.g
        elif guess == 'g':
            x = copy.copy(-self.g)
        elif guess == 'enpt2':
            x = -np.divide(self.g, self.H_N_diag)
        elif guess == 'mp2':
            E, x = self.hylleraas_mp2()
        else:
            x = copy.copy(guess)

        #info = minimize(self.UCC_2_HATER3_energy, x, jac = self.UCC_2_HATER3_grad, hessp = self.UCC_2_HATER3_hess_action, callback = self.UCC_2_HATER3_cb, method = "newton-cg", options = {'maxiter': 50, 'disp': True, 'xtol': tol})
        info = minimize(self.UCC_2_HATER3_energy, x, jac = self.UCC_2_HATER3_grad, callback = self.UCC_2_HATER3_cb, method = "l-bfgs-b", options = {'maxiter': 500, 'disp': True, 'gtol': tol, 'ftol': 0})
        if info.success == False:
            print("Iterations did not converge.")
            return float('NaN'), 'mp2' 

        energy = info.fun
        x = info.x
        print("Final callback")
        self.UCC_2_HATER3_cb(x)
        print('\n')
        return energy, x 

    def UCC_2_HATERI_energy(self, x):
        #E_old = self.hf + self.g.T@np.sin(2*x) + .5*self.H_N_diag.T@np.ones(self.g.shape) - .5*self.H_N_diag.T@np.cos(2*x) + x.T@self.d_inf_shucc_H_N(x) - self.H_N_diag.T@(x*x)
        E = self.hf + self.g.T@np.sin(2*x) + self.H_N_diag.T@(np.sin(x)*np.sin(x)-x*x) + x.T@self.UCCSD_2_A(x)
        #print(str(E_old)+" "+str(E), flush = True)
        return E

    def UCC_2_HATERI_grad(self, x):
        grad = 2*self.UCCSD_2_A(x) + 2*self.g*np.cos(2*x) + self.H_N_diag*(np.sin(2*x)-2*x)
        return grad 
         
    def UCC_2_HATERI_hess_action(self, x, p):
        hess = 2*self.UCCSD_2_A(p) - 4*self.g*np.sin(2*x)*p + self.H_N_diag*(2*np.cos(2*x)*p-2*p)
        return hess

    def UCC_2_HATERI_cb(self, x):
        self.iteration += 1
        self.times.append(time.time())
        energy = self.UCC_2_HATERI_energy(x)
        grad = self.UCC_2_HATERI_grad(x)
        print(f"\nIteration: {self.iteration}")
        print(f"Energy: {energy}")
        print(f"Gradient norm: {np.linalg.norm(grad)}")
        return energy

    def UCC_2_HATERI(self, guess = 'hf', trotter = False, tol = 1e-5):
        self.trotter = copy.copy(trotter)
        self.iteration = 0
        self.times = [time.time()]
        print("UCC_2 + HATER_Infty:")
        if guess == 'hf':
            x = 0*self.g
        elif guess == 'enpt2':
            x = -np.divide(self.g, self.H_N_diag)
        elif guess == 'mp2':
            E, x = self.hylleraas_mp2()
        elif guess == 'enucc':
            E, x = self.UCC_2_HATER3(trotter = trotter)
        else:
            x = copy.copy(guess)
        
        #info = minimize(self.UCC_2_HATERI_energy, x, jac = self.UCC_2_HATERI_grad, hessp = self.UCC_2_HATERI_hess_action, callback = self.UCC_2_HATERI_cb, method = "newton-cg", options = {'maxiter': 50, 'xtol': tol, 'disp': True})
        info = minimize(self.UCC_2_HATERI_energy, x, jac = self.UCC_2_HATERI_grad, callback = self.UCC_2_HATERI_cb, method = "L-BFGS-B", options = {'disp': True, 'gtol': tol, 'maxls': 500, 'ftol': 0})          
        if info.success == False:
            print("HATERI did not converge.")
            return float("NaN"), "mp2" 
        energy = info.fun
        x = info.x
        print("Final callback")
        self.UCC_2_HATERI_cb(x)
        return energy, x 

    def HATERI_gnorm(self, x):
        grad = self.UCC_2_HATERI_grad(x)
        return grad.T@grad

    def HATERI_jac(self, x):
        grad = self.UCC_2_HATERI_grad(x)
        hessp = self.UCC_2_HATERI_hess_action(x, grad)
        return 2*hessp
   
    def HATERI_hessp(self, x, p):
        grad = self.UCC_2_HATERI_grad(x)
        hessp = self.UCC_2_HATERI_hess_action(x, p)
        hess2p = 2*self.UCC_2_HATERI_hess_action(x, hessp)
        GJp = -8*grad*p*(2*self.g*np.cos(2*x)-self.H_N_diag*np.sin(2*x))
        return hess2p + GJp

    def Brute_Force_UCC2_HATERI(self, guess = "hf", trotter = False, tol = 1e-8):
        self.trotter = copy.copy(trotter)
        self.iteration = 0
        self.times = [time.time()]
        print("UCC_2 + HATER_Infty:")
        if guess == 'hf':
            x = 0*self.g
        elif guess == 'enpt2':
            x = -np.divide(self.g, self.H_N_diag)
        elif guess == 'mp2':
            E, x = self.hylleraas_mp2()
        elif guess == 'enucc':
            E, x = self.UCC_2_HATER3(guess = 'mp2')
        else:
            x = copy.copy(guess)

        #info = minimize(self.HATERI_gnorm, x, jac = self.HATERI_jac, hessp = self.HATERI_hessp, callback = self.UCC_2_HATERI_cb, method = "newton-cg", options = {'xtol': tol, 'disp': True})
        info = minimize(self.HATERI_gnorm, x, jac = self.HATERI_jac, callback = self.UCC_2_HATERI_cb, method = "L-BFGS-B", options = {'disp': True, 'gtol': tol, 'maxls': 500, 'ftol': 0})          
        if info.success == False:
            print("HATERI did not converge.")
            return float("NaN"), "mp2" 
        x = info.x
        print("Final callback")
        self.UCC_2_HATERI_cb(x)
        energy = self.UCC_2_HATERI_energy(x)
        return energy, x 
        
    def uccsd2(self, tol = 1e-6, trotter = False, guess = False):
        self.times = [time.time()]
        print("\n***UCCSD2***\n")
        print("Conjugate Gradient Convergence:\n")
        self.Finv_count = 0
        self.H_count = 0
        self.iteration = 0
        if trotter == False:
            Aop = LinearOperator((self.vec_size, self.vec_size), matvec = self.UCCSD2_H_N)       
        elif trotter == 'sd':
            Aop = LinearOperator((self.vec_size, self.vec_size), matvec = self.UCCSD2_H_N_no_ov)       

        print(f" CG Iter |   Energy (a.u.)   | Norm of Residual | Iteration Time |")
        self.times.append(time.time())

        x0 = copy.copy(guess)
        try:
            x, info = cg(Aop, x0, maxiter = 250, tol = tol, M = self.M, callback = self.lccsd_cb)
        except:
            x, info = cg(Aop, -self.g, maxiter = 250, tol = tol, M = self.M, callback = self.lccsd_cb)
        self.times.append(time.time())
        if info != 0:
            print("CG took too many iterations.")
            return None, x
        energy = self.hf + self.g.T.dot(x)
        print("\n")
        print(f"CG Iterations:                   {self.iteration:18d}")
        print(f"Number of Hamiltonian actions:   {self.H_count:18d}")
        print(f"Number of inv(F_diag) actions:   {self.Finv_count:18d}")
        print(f"Converged UCCSD2 Energy (a.u.):  {self.hf + self.g.T.dot(x):18.8f}")
        print(f"Norm of residual:                {self.resid_norm:18.5e}")
        print(f"UCCSD2 Norm of solution:         {np.linalg.norm(x):18.5e}")
        t1a, t1b, t2aa, t2ab, t2bb = self.tensor(x)
        t1_norm = np.linalg.norm(x[:self.n_singles])
        t2_norm = np.linalg.norm(x[self.n_singles:])
        t1_diagnostic = t1_norm/np.sqrt(2*self.N)
        d1 = np.linalg.norm(t1a)
        print(f"UCCSD2 T1 Norm:          {t1_norm:20.8e}")
        print(f"UCCSD2 T1 Diagnostic:    {t1_diagnostic:20.8e}")
        print(f"UCCSD2 D1 Diagnostic:    {d1:20.8e}")
        print(f"UCCSD2 T2 Norm:          {t2_norm:20.8e}")
        idx = np.argsort(abs(x))         
        print(f"Largest amplitude:               {x[idx][-1]:18.5e}")
        print(f"Average iteration time (s):      {(self.times[-1]-self.times[1])/self.iteration:18.8f}")
        print(f"Total time elapsed (s):          {time.time()-self.times[0]:18.8f}") 
        return energy, x

    def shucc_energy(self, x):
        E = self.hf + 2*self.g.T.dot(x) + x.T@(self.UCCSD_2_A(x))
        return E

    def shucc_grad(self, x):
        grad = 2*self.g + 2*self.UCCSD_2_A(x)
        return grad
    
    def shucc_hessp(self, x, p):
        return 2*self.UCCSD_2_A(p)      
  
    def shucc_cb(self, x):
        self.iteration += 1
        self.times.append(time.time())
        energy = self.shucc_energy(x)
        grad = self.shucc_grad(x)
        print(f"\nIteration: {self.iteration}", flush = True)
        print(f"Energy: {energy}", flush = True)
        print(f"Gradient norm: {np.linalg.norm(grad)}", flush = True)
        return energy

    def shucc(self, guess = 'hf', trotter = False, tol = 1e-8):
        self.trotter = copy.copy(trotter)
        self.iteration = 0
        self.times = [time.time()]
        print("SHUCC:")
        if guess == 'hf':
            x = 0*self.g
        elif guess == 'g':
            x = copy.copy(-self.g)
        elif guess == 'enpt2':
            x = -np.divide(self.g, self.H_N_diag)
        elif guess == 'mp2':
            E, x = self.hylleraas_mp2()
        else:
            x = copy.copy(guess)

        info = minimize(self.shucc_energy, x, jac = self.shucc_grad, hessp = self.shucc_hessp, callback = self.shucc_cb, method = "newton-cg", options = {'maxiter': 500, 'disp': True, 'xtol': tol})
        #info = minimize(self.shucc_energy, x, jac = self.shucc_grad, callback = self.shucc_cb, method = "l-bfgs-b", options = {'maxiter': 500, 'disp': True, 'gtol': tol})
        if info.success == False:
            print("Iterations did not converge.")
            return float('nan'), 'mp2' 
        energy = info.fun
        x = info.x
        print('\n')
        return energy, x 
    
    def d_inf_energy(self, x):
        E = self.hf + self.g.T@np.sin(2*x) + .5*np.sum(self.H_N_diag) - .5*self.H_N_diag.T@np.cos(2*x) + x.T@self.d_inf_shucc_H_N(x) - contract('i,i,i', self.H_N_diag, x, x)
        return E

    def d_inf_cb(self, x):
        self.iteration += 1
        self.times.append(time.time())
        energy = self.d_inf_energy(x)
        resid = self.d_inf_resid(x)
        resid_norm = np.linalg.norm(resid) 
        print(f"{self.iteration:8d} | {energy:16.10f}  | {resid_norm:16.5e} | {self.times[-1] - self.times[-2]:14.2f} |")
        self.resid_norm = resid_norm
        return energy

    def d_inf_resid(self, x):
        resid = 2*np.multiply(self.g, np.cos(2*x)) + np.multiply(self.H_N_diag, np.sin(2*x)-2*x) + 2*self.d_inf_shucc_H_N(x) 
        return resid
    
    def d_inf_hessp(self, x, p):
        hessp = 2*self.d_inf_shucc_H_N(p) - 2*np.multiply(self.H_N_diag, p) - 4*contract('i,i,i', self.g, np.sin(2*x), p) + 2*contract('i,i,i', self.H_N_diag, np.cos(2*x), p)
        return hessp

    def d_inf_shucc_H_N(self, x):
        if self.trotter == 'sd':
            return self.UCCSD2_H_N_no_ov(x)
        else:
            return self.UCCSD2_H_N(x)
    
    
    def dumb_uccsd2_d_inf(self, tol = 1e-6, trotter = False, guess = 'hf'):
        self.trotter = copy.copy(trotter)
        self.times = [time.time()]
        self.iteration = 0
        print("UCCSD(2T,Inf_D)")
        if guess == 'hf':
            x = 0*self.g
        elif guess == 'enpt':
            x = .5*np.arctan(-2*np.divide(self.g, self.H_N_diag))
        elif guess == 'enpt2':
            x = -np.divide(self.g, self.H_N_diag)
        elif guess == 'mp2':
            E, x = self.hylleraas_mp2()
        elif guess == 'enucc':
            E, x = self.enucc3()
        else:
            x = copy.copy(guess)
        info = minimize(self.d_inf_energy, x, jac = self.d_inf_resid, callback = self.d_inf_cb, method = "l-bfgs-b", options = {'disp': True, 'gtol': 1e-8})
        energy = info.fun
        x = info.x
        print('\n')
        print(f"Converged UCCSD(2T,ID) Energy (a.u.):    {energy:18.8f}")
        print(f"UCCSD(2T,ID) Norm of solution:           {np.linalg.norm(x):18.5e}")
        t1a, t1b, t2aa, t2ab, t2bb = self.tensor(x)
        t1_norm = np.linalg.norm(x[:self.n_singles])
        t2_norm = np.linalg.norm(x[self.n_singles:])
        t1_diagnostic = t1_norm/np.sqrt(2*self.N)
        d1 = np.linalg.norm(t1a)
        print(f"UCCSD(2T,ID) T1 Norm:            {t1_norm:20.8e}")
        print(f"UCCSD(2T,ID) T1 Diagnostic:      {t1_diagnostic:20.8e}")
        print(f"UCCSD(2T,ID) D1 Diagnostic:      {d1:20.8e}")
        print(f"UCCSD(2T,ID) T2 Norm:            {t2_norm:20.8e}")
        idx = np.argsort(abs(x))         
        print(f"Largest amplitude:               {x[idx][-1]:18.5e}")
        print(f"Total time elapsed (s):          {time.time()-self.times[0]:18.8f}")         
        return energy, x

 
    def uccsd2_d_inf(self, tol = 1e-5, trotter = False, guess = 'enpt'):
        self.trotter = copy.copy(trotter)

        print("UCCSD(2T,Inf_D)")
        if guess == 'hf':
            x = 0*self.g
        elif guess == 'enpt':
            x = .5*np.arctan(-2*np.divide(self.g, self.H_N_diag))
        elif guess == 'enpt2':
            x = -np.divide(self.g, self.H_N_diag)
        elif guess == 'mp2':
            E, x = self.hylleraas_mp2()
        elif guess == 'enucc':
            E, x = self.enucc3()
        else:
            x = copy.copy(guess)
        resid_prev = np.linalg.norm(self.d_inf_resid(x))
        self.x0 = copy.copy(x)
        self.times = [time.time()]
        print("\n***UCCSD2 + Infinite Diagonal***\n")
        print("Conjugate Gradient Convergence:\n")
        self.Finv_count = 0
        self.H_count = 0
        self.iteration = 0


        Done = False

        Aop = LinearOperator((self.vec_size, self.vec_size), matvec = self.d_inf_shucc_H_N, rmatvec = self.d_inf_shucc_H_N)

        macro_iters = 0
        energy = self.d_inf_energy(x)
        self.macro_times = [time.time()]
        while Done == False:
            self.times = [time.time()]
            old_E = copy.deepcopy(energy)
            macro_iters += 1
            b = -np.multiply(self.g, np.cos(2*x)) - .5*np.multiply(self.H_N_diag, np.sin(2*x)) + np.multiply(self.H_N_diag, x)
            #b = 0*x
            print(f" CG Iter |   Energy (a.u.)   | Norm of Residual | Iteration Time |")
            x0 = copy.copy(x)
            self.x0 = copy.copy(x0)
            x, info = cg(Aop, b, x0 = x, tol = tol*1e-1, M = self.M, maxiter = 200, callback = self.d_inf_cb)
            inner = False

            while inner == False:
                energy = self.d_inf_energy(x)
                resid = np.linalg.norm(self.d_inf_resid(x))

                if resid/resid_prev > 1:
                    print("Damping to ensure residual decreases.")
                    x = .5*(x + x0)
                else:
                    inner = True 
                if np.linalg.norm(x-x0) < 1e-15:
                    print("Error:  Failed to decrease residual.")
                    return 0, x

            print(f"CG Iterations:                   {self.iteration:18d}")
            print(f"Number of Hamiltonian actions:   {self.H_count:18d}")
            print(f"Number of inv(F_diag) actions:   {self.Finv_count:18d}")
            print(f"Energy (a.u.):                   {energy:18.8f}")
            print(f"Norm of residual:                {resid:18.5e}")
            print(f"Norm of solution:                {np.linalg.norm(x):18.5e}")
            print(f"Norm of single amplitudes:       {np.linalg.norm(x[:self.n_singles]):18.5e}") 
            print(f"Norm of double amplitudes:       {np.linalg.norm(x[self.n_singles:]):18.5e}") 
            idx = np.argsort(abs(x))         
            print(f"Largest amplitude:               {x[idx][-1]:18.5e}")
            print(f"Average iteration time (s):      {(self.times[-1]-self.times[1])/self.iteration:18.8f}")
            print(f"Total time of iteration (s):     {time.time()-self.times[0]:18.8f}") 
            resid_prev = copy.copy(resid)
            print(f"Residual:  {resid}")
            if resid < tol:
                Done = True 
                energy = self.d_inf_energy(x)
            self.macro_times.append(time.time())
        print('\n')
        print(f"Converged UCCSD(2T,ID) Energy (a.u.):    {energy:18.8f}")
        print(f"UCCSD(2T,ID) Norm of residual:           {resid:18.5e}")
        print(f"UCCSD(2T,ID) Norm of solution:           {np.linalg.norm(x):18.5e}")
        t1a, t1b, t2aa, t2ab, t2bb = self.tensor(x)
        t1_norm = np.linalg.norm(x[:self.n_singles])
        t2_norm = np.linalg.norm(x[self.n_singles:])
        t1_diagnostic = t1_norm/np.sqrt(2*self.N)
        d1 = np.linalg.norm(t1a)
        print(f"UCCSD(2T,ID) T1 Norm:            {t1_norm:20.8e}")
        print(f"UCCSD(2T,ID) T1 Diagnostic:      {t1_diagnostic:20.8e}")
        print(f"UCCSD(2T,ID) D1 Diagnostic:      {d1:20.8e}")
        print(f"UCCSD(2T,ID) T2 Norm:            {t2_norm:20.8e}")
        idx = np.argsort(abs(x))         
        print(f"Largest amplitude:               {x[idx][-1]:18.5e}")
        print(f"Average macroiteration time (s): {(self.macro_times[-1]-self.macro_times[1])/self.iteration:18.8f}")
        print(f"Total time elapsed (s):          {time.time()-self.macro_times[0]:18.8f}")         
        return energy, x



    def acpf(self, tol = 1e-6, guess = "mp2"):
        self.macro_times = [time.time()]
        print('\n***ACPF***\n')
        Done = False
        if guess is not None and guess == "hf":

            print("Using HF as initial ACPF guess.")
            x_prev = 0*self.g
            Ec = 0
        elif guess is not None and guess == "mp2":
            if self.loc is not None: 
                print("Using Hylleraas MP2 as initial ACPF guess.")
                Ec, x_prev = self.hylleraas_mp2()
                Ec -= self.hf
            if self.loc is None: 
                print("Using Non-Iterative MP2 as initial ACPF guess.")
                Ec, x_prev = self.canonical_mp2()
                Ec -= self.hf
        else:
            print(f"Using a guess correlation energy of {guess} a.u.")
            Ec = copy.copy(guess)
            x_prev = 0
        print("\n")
        self.shift = -(2/self.N)*Ec
        Aop = LinearOperator((self.vec_size, self.vec_size), matvec = self.acpf_H_N, rmatvec = self.acpf_H_N)
        resid_prev = np.linalg.norm(self.acpf_H_N(x_prev)+self.g)
        macro_iters = 0
        self.macro_times = [time.time()]
        while Done == False:
            self.times = [time.time()]
            print(f"\nMacro Iteration {macro_iters}:\n")
            self.Finv_count = 0
            self.H_count = 0
            self.iteration = 0
            print(f" CG Iter |   Energy (a.u.)   | Norm of Residual | Iteration Time |")
            x, info = cg(Aop, -self.g, x0 = x_prev, tol = tol*1e-1, M = self.M, callback = self.lccsd_cb)
            self.times.append(time.time())
            assert info == 0
            print("\n")
            Ec = self.g.T.dot(x) 
            self.shift = -(2/self.N)*Ec
            err = self.g + Aop(x)
            resid = np.linalg.norm(err) 
            print(f"CG Iterations:                   {self.iteration:18d}")
            print(f"Number of Hamiltonian actions:   {self.H_count:18d}")
            print(f"Number of inv(F_diag) actions:   {self.Finv_count:18d}")
            print(f"Energy (a.u.):                   {self.hf + self.g.T.dot(x):18.8f}")
            print(f"Norm of residual:                {resid:18.5e}")
            print(f"Norm of solution:                {np.linalg.norm(x):18.5e}")
            print(f"Norm of single amplitudes:       {np.linalg.norm(x[:self.n_singles]):18.5e}") 
            print(f"Norm of double amplitudes:       {np.linalg.norm(x[self.n_singles:]):18.5e}") 
            idx = np.argsort(abs(x))         
            print(f"Largest amplitude:               {x[idx][-1]:18.5e}")
            print(f"Average iteration time (s):      {(self.times[-1]-self.times[1])/self.iteration:18.8f}")
            print(f"Total time of iteration (s):     {time.time()-self.times[0]:18.8f}") 


            good = False
            print("\nEnsuring no sudden increase in gradient...\n")
            while good == False:
                if resid > 1.01*resid_prev:
                    x = .5*x + .5*x_prev
                    Ec = self.g.T.dot(x)
                    self.shift = -(2/self.N)*Ec
                    err = self.g + Aop(x)
                    resid = np.linalg.norm(err)                    
                    print("Damping between x and x_prev...")
                    print(f"Energy (a.u.):                   {self.hf + self.g.T.dot(x):18.8f}")
                    print(f"Norm of residual:                {np.linalg.norm(resid):18.5e}")
                    print(f"Norm of solution:                {np.linalg.norm(x):18.5e}")
                    print(f"Norm of single amplitudes:       {np.linalg.norm(x[:self.n_singles]):18.5e}") 
                    print(f"Norm of double amplitudes:       {np.linalg.norm(x[self.n_singles:]):18.5e}") 
                    idx = np.argsort(abs(x))         
                    print(f"Largest amplitude:               {x[idx][-1]:18.5e}")
                else:
                    good = True
            if abs(resid) < tol:
                Done = True
            if math.isnan(Ec) or abs(Ec) > 1e16 or (macro_iters > 10000):
                print("Failed to converge.")
                exit()
            x_prev = copy.copy(x)
            resid_prev = copy.copy(resid)
            macro_iters += 1
            self.macro_times.append(time.time())
        print('\n')
        print(f"Converged ACPF Energy (a.u.):    {self.hf + self.g.T.dot(x):18.8f}")
        print(f"Norm of residual:                {np.linalg.norm(resid):18.5e}")
        print(f"ACPF Norm of solution:           {np.linalg.norm(x):18.5e}")
        t1a, t1b, t2aa, t2ab, t2bb = self.tensor(x)
        t1_norm = np.linalg.norm(x[:self.n_singles])
        t2_norm = np.linalg.norm(x[self.n_singles:])
        t1_diagnostic = t1_norm/np.sqrt(2*self.N)
        d1 = np.linalg.norm(t1a)
        print(f"ACPF T1 Norm:          {t1_norm:20.8e}")
        print(f"ACPF T1 Diagnostic:    {t1_diagnostic:20.8e}")
        print(f"ACPF D1 Diagnostic:    {d1:20.8e}")
        print(f"ACPF T2 Norm:          {t2_norm:20.8e}")

        idx = np.argsort(abs(x))         
        print(f"Largest amplitude:               {x[idx][-1]:18.5e}")
        print(f"Average macroiteration time (s): {(self.macro_times[-1]-self.macro_times[1])/self.iteration:18.8f}")
        print(f"Total time elapsed (s):          {time.time()-self.macro_times[0]:18.8f}")         
        return self.hf + self.g.T.dot(x), x
    
    def aqcc(self, tol = 1e-6, guess = "mp2"):
        self.macro_times = [time.time()]
        print('\n***AQCC***\n')
        Done = False
        if guess is not None and guess == "hf":
            print("Using HF as initial AQCC guess.")
            x_prev = 0*self.g
            Ec = 0
        elif guess is not None and guess == "mp2":
            if self.loc is not None: 
                print("Using Hylleraas MP2 as initial AQCC guess.")
                Ec, x_prev = self.hylleraas_mp2()
                Ec -= self.hf
            if self.loc is None: 
                print("Using Non-Iterative MP2 as initial AQCC guess.")
                Ec, x_prev = self.canonical_mp2()
                Ec -= self.hf
        else:
            print(f"Using a guess correlation energy of {guess} a.u.")
            Ec = copy.copy(guess)
            x_prev = 0
        print("\n")
        self.shift = -(1-((self.N-3)*(self.N-2))/((self.N-1)*self.N))*Ec
        Aop = LinearOperator((self.vec_size, self.vec_size), matvec = self.acpf_H_N, rmatvec = self.acpf_H_N)
        resid_prev = np.linalg.norm(self.acpf_H_N(x_prev)+self.g)
        macro_iters = 0
        self.macro_times = [time.time()]
        while Done == False:
            self.times = [time.time()]
            print(f"\nMacro Iteration {macro_iters}:\n")
            self.Finv_count = 0
            self.H_count = 0
            self.iteration = 0
            print(f" CG Iter |   Energy (a.u.)   | Norm of Residual | Iteration Time |")
            x, info = cg(Aop, -self.g, x0 = x_prev, tol = tol*1e-1, M = self.M, callback = self.lccsd_cb)
            self.times.append(time.time())
            assert info == 0
            print("\n")
            Ec = self.g.T.dot(x) 
            self.shift = -(1-((self.N-3)*(self.N-2))/((self.N-1)*self.N))*Ec
            err = self.g + Aop(x)
            resid = np.linalg.norm(err) 
            print(f"CG Iterations:                   {self.iteration:18d}")
            print(f"Number of Hamiltonian actions:   {self.H_count:18d}")
            print(f"Number of inv(F_diag) actions:   {self.Finv_count:18d}")
            print(f"Energy (a.u.):                   {self.hf + self.g.T.dot(x):18.8f}")
            print(f"Norm of residual:                {resid:18.5e}")
            print(f"Norm of solution:                {np.linalg.norm(x):18.5e}")
            print(f"Norm of single amplitudes:       {np.linalg.norm(x[:self.n_singles]):18.5e}") 
            print(f"Norm of double amplitudes:       {np.linalg.norm(x[self.n_singles:]):18.5e}") 
            idx = np.argsort(abs(x))         
            print(f"Largest amplitude:               {x[idx][-1]:18.5e}")
            print(f"Average iteration time (s):      {(self.times[-1]-self.times[1])/self.iteration:18.8f}")
            print(f"Total time of iteration (s):     {time.time()-self.times[0]:18.8f}") 

            good = False
            print("\nEnsuring no sudden increase in gradient...\n")
            while good == False:
                if resid > 1.01*resid_prev:
                    x = .5*x + .5*x_prev
                    Ec = self.g.T.dot(x)
                    self.shift = -(1-((self.N-3)*(self.N-2))/((self.N-1)*self.N))*Ec
                    err = self.g + Aop(x)
                    resid = np.linalg.norm(err)                    
                    print("Damping between x and x_prev...")
                    print(f"Energy (a.u.):                   {self.hf + self.g.T.dot(x):18.8f}")
                    print(f"Norm of residual:                {np.linalg.norm(resid):18.5e}")
                    print(f"Norm of solution:                {np.linalg.norm(x):18.5e}")
                    print(f"Norm of single amplitudes:       {np.linalg.norm(x[:self.n_singles]):18.5e}") 
                    print(f"Norm of double amplitudes:       {np.linalg.norm(x[self.n_singles:]):18.5e}") 
                    idx = np.argsort(abs(x))         
                    print(f"Largest amplitude:               {x[idx][-1]:18.5e}")
                else:
                    good = True
            if abs(resid) < tol:
                Done = True
            if math.isnan(Ec) or abs(Ec) > 1e16 or (macro_iters > 10000):
                print("Failed to converge.")
                exit()
            x_prev = copy.copy(x)
            resid_prev = copy.copy(resid)
            macro_iters += 1
            self.macro_times.append(time.time())
        print("\n")
        print(f"Converged AQCC Energy (a.u.):    {self.hf + self.g.T.dot(x):18.8f}")
        print(f"Norm of residual:                {np.linalg.norm(resid):18.5e}")
        print(f"AQCC Norm of solution:           {np.linalg.norm(x):18.5e}")
        t1a, t1b, t2aa, t2ab, t2bb = self.tensor(x)
        t1_norm = np.linalg.norm(x[:self.n_singles])
        t2_norm = np.linalg.norm(x[self.n_singles:])
        t1_diagnostic = t1_norm/np.sqrt(2*self.N)
        d1 = np.linalg.norm(t1a)
        print(f"AQCC T1 Norm:          {t1_norm:20.8e}")
        print(f"AQCC T1 Diagnostic:    {t1_diagnostic:20.8e}")
        print(f"AQCC D1 Diagnostic:    {d1:20.8e}")
        print(f"AQCC T2 Norm:          {t2_norm:20.8e}")

        idx = np.argsort(abs(x))         
        print(f"Largest amplitude:               {x[idx][-1]:18.5e}")
        print(f"Average macroiteration time (s): {(self.macro_times[-1]-self.macro_times[1])/self.iteration:18.8f}")
        print(f"Total time elapsed (s):          {time.time()-self.macro_times[0]:18.8f}")         
        return self.hf + self.g.T.dot(x), x

    def ENPT_inf(self):
        print("\n***Infinite Order Diagonal Approximation to UCCSD***\n")
        E = self.hf + .5*np.sum(self.H_N_diag - np.sqrt(4*np.multiply(self.g, self.g) + np.multiply(self.H_N_diag, self.H_N_diag)))
        print(f"ENPT(inf) energy (a.u.): {E:20.16f}")
        #x = .5*np.arctan(-2*np.divide(self.g, self.H_N_diag))
        #print(self.H_N_diag)
        #E = self.hf + self.g.T@np.sin(2*x) +  .5*np.sum(self.H_N_diag) - .5*self.H_N_diag.T@np.cos(2*x)
        #print(f"Trig way (a.u.): {E:20.16f}")
        return E

    def ENPT2(self):
        print("\nEpstein-Nesbet PT2***\n")
        E = self.hf - np.sum(np.divide(np.multiply(self.g, self.g), self.H_N_diag))
        print(f"ENPT2 energy (a.u.): {E:20.16f}")
        return E

    def enucc3(self, tol = 1e-5, guess = "mp2", trotter = False):
        self.macro_times = [time.time()]
        print('\n***ENUCC(3)***\n')
        Done = False
        if guess == "mp2":
            print("Using Hylleraas MP2 as initial ENUCC(3) guess.")
            Ec, x_prev = self.hylleraas_mp2()
            Ec -= self.hf
        elif guess == "hf":
            print("Using HF as initial ENUCC(3) guess.")
            x_prev = 0*self.g
            Ec = 0
        elif guess == "grad":
            print("Taking a small step in direction associated w/ imaginary time evolution")
            x_prev = -self.g
            Ec = self.g.T.dot(x_prev) + (2/3)*np.multiply(self.g, x_prev).T.dot(np.multiply(x_prev,x_prev))
        elif guess == "cisd":
            print("Using CISD as initial ENUCC(3) guess.")
            Ec, x_prev = self.cisd()
            Ec -= self.hf
        else:
            print("Using a manual guess.")
            x_prev = guess
            Ec = self.g.T.dot(x_prev) + (2/3)*np.multiply(self.g, x_prev).T.dot(np.multiply(x_prev,x_prev))

        print("\n")
        self.shift = -2*np.multiply(self.g, x_prev)
        if trotter == False:
            Aop = LinearOperator((self.vec_size, self.vec_size), matvec = self.shifted_H_N, rmatvec = self.shifted_H_N)
        elif trotter == 'sd':
            Aop = LinearOperator((self.vec_size, self.vec_size), matvec = self.shifted_H_N_no_ov, rmatvec = self.shifted_H_N_no_ov)
        resid_prev = np.linalg.norm(self.shifted_H_N(x_prev)+self.g)
        macro_iters = 0
        self.macro_times = [time.time()]
        while Done == False:
            self.times = [time.time()]
            print(f"\nMacro Iteration {macro_iters}:\n")
            self.Finv_count = 0
            self.H_count = 0
            self.iteration = 0
            print(f" CG Iter |   Energy (a.u.)   | Norm of Residual | Iteration Time |")
            x, info = cg(Aop, -self.g, x0 = x_prev, tol = tol*1e-2, M = self.M, callback = self.enucc_cb)
            self.times.append(time.time())
            assert info == 0
            print("\n")
            self.shift = -2*np.multiply(self.g, x)
            err = self.g + Aop(x)
            resid = np.linalg.norm(err) 
            print(f"CG Iterations:                   {self.iteration:18d}")
            print(f"Number of Hamiltonian actions:   {self.H_count:18d}")
            print(f"Number of inv(F_diag) actions:   {self.Finv_count:18d}")
            print(f"Energy (a.u.):                   {self.hf + self.g.T.dot(x) + (2/3)*np.multiply(self.g, x).T.dot(np.multiply(x,x)):18.8f}")
            print(f"Norm of residual:                {resid:18.5e}")
            print(f"Norm of solution:                {np.linalg.norm(x):18.5e}")
            print(f"Norm of single amplitudes:       {np.linalg.norm(x[:self.n_singles]):18.5e}") 
            print(f"Norm of double amplitudes:       {np.linalg.norm(x[self.n_singles:]):18.5e}") 
            idx = np.argsort(abs(x))         
            print(f"Largest amplitude:               {x[idx][-1]:18.5e}")
            print(f"Average iteration time (s):      {(self.times[-1]-self.times[1])/self.iteration:18.8f}")
            print(f"Total time of iteration (s):     {time.time()-self.times[0]:18.8f}") 

            good = False
            while good == False:
                #if resid > 1.01*resid_prev or macro_iters > 20:
                if macro_iters > 20:
                    x = .5*x + .5*x_prev
                    self.shift = -2*np.multiply(self.g, x)
                    err = self.g + Aop(x)
                    resid = np.linalg.norm(err)                    
                    print("Damping between x and x_prev...")
                    print(f"Energy (a.u.):               {self.hf + self.g.T.dot(x) + (2/3)*np.multiply(self.g, x).T.dot(np.multiply(x,x)):18.8f}")
                    print(f"Norm of residual:            {np.linalg.norm(resid):18.5e}")
                    print(f"Norm of solution:            {np.linalg.norm(x):18.5e}")
                    print(f"Norm of single amplitudes:   {np.linalg.norm(x[:self.n_singles]):18.5e}") 
                    print(f"Norm of double amplitudes:   {np.linalg.norm(x[self.n_singles:]):18.5e}") 
                    idx = np.argsort(abs(x))         
                    print(f"Largest amplitude:           {x[idx][-1]:18.5e}")
                    #if resid < 1.01*resid_prev:
                    #    good = True
                    good = True
                else:
                    good = True
            if abs(resid) < tol:
                Done = True

            if macro_iters > 100:
                print("Failed to converge.")
                return None, None
            x_prev = copy.copy(x)
            resid_prev = copy.copy(resid)
            macro_iters += 1
            self.macro_times.append(time.time())
        print("\n")
        print(f"Converged ENUCC(3) Energy (a.u.): {self.hf + self.g.T.dot(x)+(2/3)*np.multiply(self.g,x).T.dot(np.multiply(x,x)):18.8f}")
        print(f"Norm of residual:                 {np.linalg.norm(resid):18.5e}")
        print(f"ENUCC(3) Norm of solution:        {np.linalg.norm(x):18.5e}")
        t1a, t1b, t2aa, t2ab, t2bb = self.tensor(x)
        t1_norm = np.linalg.norm(x[:self.n_singles])
        t2_norm = np.linalg.norm(x[self.n_singles:])
        t1_diagnostic = t1_norm/np.sqrt(2*self.N)
        d1 = np.linalg.norm(t1a)
        print(f"ENUCC(3) T1 Norm:          {t1_norm:20.8e}")
        print(f"ENUCC(3) T1 Diagnostic:    {t1_diagnostic:20.8e}")
        print(f"ENUCC(3) D1 Diagnostic:    {d1:20.8e}")
        print(f"ENUCC(3) T2 Norm:          {t2_norm:20.8e}")
        idx = np.argsort(abs(x))         
        print(f"Largest amplitude:                {x[idx][-1]:18.5e}")
        print(f"Average macroiteration time (s):  {(self.macro_times[-1]-self.macro_times[1])/self.iteration:18.8f}")
        print(f"Total time elapsed (s):           {time.time()-self.macro_times[0]:18.8f}")         
        return self.hf + self.g.T.dot(x) + (2/3)*np.multiply(self.g, x).T.dot(np.multiply(x,x)), x


    def canonical_mp2(self):
        print("\n***Canonical MP2***\n")

        x = -self.Finv_op(self.g)
        energy = self.hf + self.g.T.dot(x)
        print(f"Converged MP2 Energy (a.u.):     {energy:18.8f}")
        print(f"MP2 Norm of solution:            {np.linalg.norm(x):18.5e}")
        t1a, t1b, t2aa, t2ab, t2bb = self.tensor(x)
        t1_norm = np.linalg.norm(x[:self.n_singles])
        t2_norm = np.linalg.norm(x[self.n_singles:])
        t1_diagnostic = t1_norm/np.sqrt(2*self.N)
        d1 = np.linalg.norm(t1a)
        print(f"MP2 T1 Norm:                     {t1_norm:20.8e}")
        print(f"MP2 T1 Diagnostic:               {t1_diagnostic:20.8e}")
        print(f"MP2 D1 Diagnostic:               {d1:20.8e}")
        print(f"MP2 T2 Norm:                     {t2_norm:20.8e}")
        idx = np.argsort(abs(x))         
        print(f"Largest amplitude:               {x[idx][-1]:18.5e}")
        return energy, x
   
    def CISD_H(self, x):
        if np.linalg.norm(x) == 0:
            x[0] = 1
        Hx = np.zeros(x.shape)
        norm = np.sqrt((x.T)@x)
        c0 = x[0]/norm
        x = x[1:]/norm
        #t'H_Nt
        Hx[0] = (self.g.T)@x + self.hf*c0
        Hx[1:] = self.H_N(x, ov = True) + c0*self.g + self.hf*x
        return Hx 
        
    def cisd(self):
        print("Doing CISD.")
        #Do CISD
        Aop = LinearOperator((self.vec_size+1, self.vec_size+1), matvec = self.CISD_H, rmatvec = self.CISD_H)
        E, v = scipy.sparse.linalg.eigsh(Aop, k = 1, which = 'SA')
        v = v[1:,0]/v[0]
        print(f"CISD energy:  {E[0]:20.16f}")
        return E[0], v
      
    def hylleraas_mp2(self, tol = 1e-12): 
        #Does iterative MP2
        self.times = [time.time()]
        print("\n***Hylleraas MP2***\n")
        print("Conjugate Gradient Convergence:\n")
        self.F_count = 0
        self.Finv_count = 0
        self.H_count = 0
        self.iteration = 0
        Aop = LinearOperator((self.vec_size, self.vec_size), matvec = self.F_N)                      
        print(f" CG Iter |   Energy (a.u.)   | Norm of Residual | Iteration Time |")
        self.times.append(time.time())

        x, info = cg(Aop, -self.g, tol = tol, M = self.M, callback = self.lccsd_cb)         
        xten = self.tensor(x)
        singg = self.g[:self.n_singles]
        singx = x[:self.n_singles]

        sing_E = singg.T.dot(singx)
        aag = self.g[self.n_singles:self.n_aaaa+self.n_singles]
        aax = x[self.n_singles:self.n_aaaa+self.n_singles]
        aa_E = aag.T.dot(aax)
        abg = self.g[self.n_singles+self.n_aaaa:self.n_singles+self.n_aaaa+self.n_abab]
        abx = x[self.n_singles+self.n_aaaa:self.n_singles+self.n_aaaa+self.n_abab]
        ab_E = abg.T.dot(abx)
        bbg = self.g[self.n_singles+self.n_aaaa+self.n_abab:]
        bbx = x[self.n_singles+self.n_aaaa+self.n_abab:]
        bb_E = bbg.T.dot(bbx)
        
        self.times.append(time.time())
        assert info == 0
        energy = self.hf + self.g.T.dot(x)

        print("\n")
        print(f"CG Iterations:                   {self.iteration:18d}")
        print(f"Number of Fock actions:          {self.F_count:18d}")
        print(f"Number of inv(F_diag) actions:   {self.Finv_count:18d}")
        print(f"Energy from singles:             {sing_E:18.8f}")
        print(f"Energy from aaaa doubles:        {aa_E:18.8f}")
        print(f"Energy from abab doubles:        {ab_E:18.8f}")
        print(f"Energy from bbbb doubles:        {bb_E:18.8f}")
        print(f"Total correlation energy:        {self.g.T.dot(x):18.8f}")
        print(f"Converged MP2 Energy (a.u.):     {self.hf + self.g.T.dot(x):18.8f}")
        print(f"Norm of residual:                {self.resid_norm:18.5e}")
        print(f"MP2 Norm of solution:            {np.linalg.norm(x):18.5e}")
        t1a, t1b, t2aa, t2ab, t2bb = self.tensor(x)
        t1_norm = np.linalg.norm(x[:self.n_singles])
        t2_norm = np.linalg.norm(x[self.n_singles:])
        t1_diagnostic = t1_norm/np.sqrt(2*self.N)
        d1 = np.linalg.norm(t1a)
        print(f"MP2 T1 Norm:                     {t1_norm:18.5e}")
        print(f"MP2 T1 Diagnostic:               {t1_diagnostic:18.5e}")
        print(f"MP2 D1 Diagnostic:               {d1:18.5e}")
        print(f"MP2 T2 Norm:                     {t2_norm:18.5e}")
        idx = np.argsort(abs(x))         
        print(f"Largest amplitude:               {x[idx][-1]:18.5e}")
        print(f"Average iteration time (s):      {(self.times[-1]-self.times[1])/self.iteration:18.8f}")
        print(f"Total time elapsed (s):          {time.time()-self.times[0]:18.8f}") 
        return energy, x


    def uhf_mp2_natural(self, tol = 1e-10):
        print("Getting UHF MP2-natural orbitals.")
        mp2_E, x = self.hylleraas_mp2()
        x = self.tensor(x)
        taa = x[2]
        tab = x[3]
        tbb = x[4]

        #Unrelaxed density a:
        P_ij_a = np.zeros((self.noa, self.noa))
        P_ij_a -= .5*contract('jkab,ikab->ij', taa, taa)
        P_ij_a -= contract('jkab,ikab->ij', tab, tab)
        P_ab_a = np.zeros((self.nva, self.nva))
        P_ab_a += .5*contract('ijac,ijbc->ab', taa, taa)
        P_ab_a += contract('ijac,ijbc->ab', tab, tab)
        #Unrelaxed density b:
        P_ij_b = np.zeros((self.nob, self.nob))
        P_ij_b -= .5*contract('jkab,ikab->ij', tbb, tbb)
        P_ij_b -= contract('kjab,kiab->ij', tab, tab)
        P_ab_b = np.zeros((self.nvb, self.nvb))
        P_ab_b += .5*contract('ijac,ijbc->ab', tbb, tbb)
        P_ab_b += contract('ijca,ijcb->ab', tab, tab)
        Pa = np.zeros((self.noa+self.nva, self.noa+self.nva))
        Pa[:self.noa,:self.noa] = P_ij_a + np.eye(self.noa)
        Pa[self.noa:,self.noa:] = P_ab_a
        Pb = np.zeros((self.nob+self.nvb, self.nob+self.nvb))
        Pb[:self.nob,:self.nob] = P_ij_b + np.eye(self.nob)
        Pb[self.nob:,self.nob:] = P_ab_b
       
        #X
        X_a = np.zeros((self.nva, self.noa))
        X_a += -contract('kija,jk->ai', self.gaa[:self.noa,:self.noa,:self.noa,self.noa:], P_ij_a)
        X_a += -contract('ikaj,jk->ai', self.Iab[:self.noa,:self.nob,self.noa:,:self.nob], P_ij_b)
        X_a += -contract('icab,bc->ai', self.gaa[:self.noa,self.noa:,self.noa:,self.noa:], P_ab_a)
        X_a += -contract('icab,bc->ai', self.Iab[:self.noa,self.nob:,self.noa:,self.nob:], P_ab_b)
        X_a += .5*contract('jabc,ijbc->ai', self.gaa[:self.noa,self.noa:,self.noa:,self.noa:], taa) 
        X_a -= contract('ajbc,ijbc->ai', self.Iab[self.noa:,:self.nob,self.noa:,self.nob:], tab)
        X_a += .5*contract('jkib,jkab->ai', self.gaa[:self.noa,:self.noa,:self.noa,self.noa:], taa)
        X_a += contract('jkib,jkab->ai', self.Iab[:self.noa,:self.nob,:self.noa,self.nob:], tab)
        
        X_b = np.zeros((self.nvb, self.nob))
        X_b += -contract('kija,jk->ai', self.gbb[:self.nob,:self.nob,:self.nob,self.nob:], P_ij_b)
        X_b += -contract('kija,jk->ai', self.Iab[:self.noa,:self.nob,:self.noa,self.nob:], P_ij_a)
        X_b += -contract('icab,bc->ai', self.gbb[:self.nob,self.nob:,self.nob:,self.nob:], P_ab_b)
        X_b += -contract('ciba,bc->ai', self.Iab[self.noa:,:self.nob,self.noa:,self.nob:], P_ab_a)
        X_b += .5*contract('jabc,ijbc->ai', self.gbb[:self.nob,self.nob:,self.nob:,self.nob:], tbb) 
        X_b -= contract('jacb,jicb->ai', self.Iab[:self.noa,self.nob:,self.noa:,self.nob:], tab)
        X_b += .5*contract('jkib,jkab->ai', self.gbb[:self.nob,:self.nob,:self.nob,self.nob:], tbb)
        X_b += contract('jkbi,jkba->ai', self.Iab[:self.noa,:self.nob,self.noa:,:self.nob], tab)

        #Solve
        Aop = LinearOperator((self.noa*self.nva+self.nob*self.nvb, self.noa*self.nva+self.nob*self.nvb), matvec = self.uhf_A, rmatvec = self.uhf_A)
        X = np.array(list(X_a.flatten())+list(X_b.flatten()))

        x, info = cg(Aop, X)

        if info != 0:
            print("WARN:  CPHF equations not necessarily solved to desired precision.")
        OVa = x[:self.noa*self.nva].reshape((self.nva, self.noa))
        OVb = x[self.noa*self.nva:].reshape((self.nvb, self.nob))
        Pa[:self.noa,self.noa:] = OVa.T
        Pa[self.noa:,:self.noa] = OVa
        Pb[:self.nob,self.nob:] = OVb.T
        Pb[self.nob:,:self.nob] = OVb
        Ca = scipy.linalg.sqrtm(self.S).real@self.Ca 
        Cb = scipy.linalg.sqrtm(self.S).real@self.Cb
        P = Ca@Pa@Ca.T + Cb@Pb@Cb.T

        #P = Pa + Pb
        occ, no = scipy.linalg.eigh(P)
        occ = occ[::-1]
        no = no[:, ::-1]      
        print("Natural Orbital Occupations:")
        for i in range(0, len(no)):
            print(f"{i:4d} {occ[i]:20.5f}")
        print("Returning MO coefficient matrix for natural orbitals.")
        #return self.Ca.dot(no)
        
        return scipy.linalg.sqrtm(np.linalg.inv(self.S)).real@no

 
    def mp2_natural(self, tol = 1e-10):
        try:
            assert(self.reference == 'rhf')
        except:
            print("MP2-natural orbitals only implemented for RHF in this method.  Use uhf_mp2_natural.")
            exit()
        print("Getting MP2-natural orbitals.")
        print("Doing canonical MP2 to get t2 amplitudes.")
        mp2_E, x = self.canonical_mp2()
        x = self.tensor(x)
        taa = x[2]
        tab = x[3]
        #Unrelaxed density:
        P_ij = np.zeros((self.noa, self.noa))
        P_ij -= .5*contract('jkab,ikab->ij', taa, taa)
        P_ij -= contract('jkab,ikab->ij', tab, tab)
        P_ab = np.zeros((self.nva, self.nva))
        P_ab += .5*contract('ijac,ijbc->ab', taa, taa)
        P_ab += contract('ijac,ijbc->ab', tab, tab)
        P = np.zeros((self.noa+self.nva, self.noa+self.nva))
        P[:self.noa,:self.noa] = 2*P_ij + 2*np.eye(self.noa)
        P[self.noa:,self.noa:] = 2*P_ab
        #X from CPHF equations.  (See Salter, '87 or L in Fritsch '90):
        X = np.zeros((self.nva, self.noa))
        #I think Fritsch et al made a sign error on these next 4 maybe?
        X += -contract('kija,jk->ai', self.gaa[:self.noa,:self.noa,:self.noa,self.noa:], P_ij)
        X += -contract('kija,jk->ai', self.Iab[:self.noa,:self.noa,:self.noa,self.noa:], P_ij)
        X += -contract('icab,bc->ai', self.gaa[:self.noa,self.noa:,self.noa:,self.noa:], P_ab)
        X += -contract('icab,bc->ai', self.Iab[:self.noa,self.noa:,self.noa:,self.noa:], P_ab)
        X += .5*contract('jabc,ijbc->ai', self.gaa[:self.noa,self.noa:,self.noa:,self.noa:], taa) 
        X -= contract('ajbc,ijbc->ai', self.Iab[self.noa:,:self.noa,self.noa:,self.noa:], tab)
        X += .5*contract('jkib,jkab->ai', self.gaa[:self.noa,:self.noa,:self.noa,self.noa:], taa)
        X += contract('jkib,jkab->ai', self.Iab[:self.noa,:self.noa,:self.noa,self.noa:], tab)
        Aop = LinearOperator((self.noa*self.nva, self.noa*self.nva), matvec = self.A, rmatvec = self.A)

        x, info = cg(Aop, X.flatten(), tol = tol)

        assert info == 0
        Dov = x.reshape((self.nva, self.noa))
        P[:self.noa,self.noa:] = 2*Dov.T    
        P[self.noa:,:self.noa] = 2*Dov

        occ, no = scipy.linalg.eigh(P)
        occ = occ[::-1]
        no = no[:, ::-1]             
        print("Natural Orbital Occupations:")
        for i in range(0, len(no)):
            print(f"{i:4d} {occ[i]:20.5f}")
        print("Returning MO coefficient matrix for natural orbitals.")
        return self.Ca.dot(no)

    def A(self, D):
        #Find action of A_{bj,ai}(eps_j - eps_b) on a trial density
        D = D.reshape((self.nva, self.noa))
        AD = -contract('ii,ai->ai', self.Fa[:self.noa, :self.noa], D)
        AD += contract('aa,ai->ai', self.Fa[self.noa:, self.noa:], D)
        AD += contract('abij,bj->ai', self.gaa[self.noa:,self.noa:,:self.noa,:self.noa], D)
        AD += contract('abij,bj->ai', self.Iab[self.noa:,self.noa:,:self.noa,:self.noa], D)
        AD += contract('ibaj,bj->ai', self.gaa[:self.noa,self.noa:,self.noa:,:self.noa], D)
        AD += contract('ibaj,bj->ai', self.Iab[:self.noa,self.noa:,self.noa:,:self.noa], D)

        return(AD.flatten())

    def uhf_A(self, D):
        Da = D[:self.nva*self.noa]
        Da = Da.reshape((self.nva, self.noa))
        Db = D[self.nva*self.noa:]
        Db = Db.reshape((self.nvb, self.nob))
        AD_a = -contract('ii,ai->ai', self.Fa[:self.noa, :self.noa], Da)
        AD_a += contract('aa,ai->ai', self.Fa[self.noa:, self.noa:], Da)
        AD_a += contract('abij,bj->ai', self.gaa[self.noa:,self.noa:,:self.noa,:self.noa], Da)
        AD_a += contract('abij,bj->ai', self.Iab[self.noa:,self.nob:,:self.noa,:self.nob], Db)
        AD_a += contract('ibaj,bj->ai', self.gaa[:self.noa,self.noa:,self.noa:,:self.noa], Da)
        AD_a += contract('ibaj,bj->ai', self.Iab[:self.noa,self.nob:,self.noa:,:self.nob], Db)
        
        AD_b = -contract('ii,ai->ai', self.Fb[:self.nob, :self.nob], Db)
        AD_b += contract('aa,ai->ai', self.Fb[self.nob:, self.nob:], Db)
        AD_b += contract('abij,bj->ai', self.gbb[self.nob:,self.nob:,:self.nob,:self.nob], Db)
        AD_b += contract('baji,bj->ai', self.Iab[self.noa:,self.nob:,:self.noa,:self.nob], Da)
        AD_b += contract('ibaj,bj->ai', self.gbb[:self.nob,self.nob:,self.nob:,:self.nob], Db)
        AD_b += contract('ibaj,bj->ai', self.Iab[:self.noa,self.nob:,self.noa:,:self.nob], Da)
        action = np.array(list(AD_a.flatten()) + list(AD_b.flatten()))

        return action
 
    def build_arr(self, op):
        arr = np.zeros((self.g.shape[0], self.g.shape[0]))
        for i in range(0, len(self.g)):
            xi = 0*self.g
            xi[i] = 1
            Hxi = op(xi)
            for j in range(0, len(self.g)):
                if arr[i,j] == 0:
                    arr[i,j] = Hxi[j]
                else:
                    assert(abs(arr[i,j]-Hxi[j])<1e-8)
        return arr
        
    def arr(self, ten):
        ta, tb, taa, tab, tbb = ten
        
        va = ta.flatten()


        vb = tb.flatten() 
        occ_aa = np.triu_indices(self.noa, k = 1)       
        vir_aa = np.triu_indices(self.nva, k = 1)       
        occ_bb = np.triu_indices(self.nob, k = 1)       
        vir_bb = np.triu_indices(self.nvb, k = 1)
        vaa = taa[occ_aa][(slice(None),)+vir_aa].flatten()
        vab = tab.reshape(self.noa*self.nob*self.nva*self.nvb)
        vbb = tbb[occ_bb][(slice(None),)+vir_bb].flatten()        

        return np.concatenate((va, vb, vaa, vab, vbb))

    def tensor(self, arr):
        s0 = self.vec_structure[0]
        s1 = self.vec_structure[1]+s0
        s2 = self.vec_structure[2]+s1
        s3 = self.vec_structure[3]+s2
        va, vb, vaa, vab, vbb = arr[0:s0], arr[s0:s1], arr[s1:s2], arr[s2:s3], arr[s3:]        
        ta = va.reshape(self.noa, self.nva)
        tb = vb.reshape(self.nob, self.nvb)
        occ_aa = np.triu_indices(self.noa, k = 1)       
        vir_aa = np.triu_indices(self.nva, k = 1)       
        occ_bb = np.triu_indices(self.nob, k = 1)       
        vir_bb = np.triu_indices(self.nvb, k = 1)
        taa = np.zeros((self.noa, self.noa, self.nva, self.nva))
        aa = vaa.reshape(taa[occ_aa][(slice(None),)+vir_aa].shape)
        taa[occ_aa[0][:,None], occ_aa[1][:,None], vir_aa[0], vir_aa[1]] = aa      
        taa[occ_aa[1][:,None], occ_aa[0][:,None], vir_aa[0], vir_aa[1]] = -aa      
        taa[occ_aa[0][:,None], occ_aa[1][:,None], vir_aa[1], vir_aa[0]] = -aa      
        taa[occ_aa[1][:,None], occ_aa[0][:,None], vir_aa[1], vir_aa[0]] = aa          
        tab = vab.reshape((self.noa, self.nob, self.nva, self.nvb)) 
        tbb = np.zeros((self.nob, self.nob, self.nvb, self.nvb))
        bb = vbb.reshape(tbb[occ_bb][(slice(None),)+vir_bb].shape)
        tbb[occ_bb[0][:,None], occ_bb[1][:,None], vir_bb[0], vir_bb[1]] = bb   
        tbb[occ_bb[1][:,None], occ_bb[0][:,None], vir_bb[0], vir_bb[1]] = -bb      
        tbb[occ_bb[0][:,None], occ_bb[1][:,None], vir_bb[1], vir_bb[0]] = -bb      
        tbb[occ_bb[1][:,None], occ_bb[0][:,None], vir_bb[1], vir_bb[0]] = bb          
        return ta, tb, taa, tab, tbb
        
    def H_N(self, x, diag = False, ov = True):        
        Fx = self.F_N(x, diag = diag, ov = ov) 
        Hx = Fx + self.arr(V_N(self.tensor(x), self))
        self.H_count += 1    
        return Hx

    def F_N(self, x, diag = False, ov = False):
        self.F_count += 1
        Fx = self.arr(F_N(self.tensor(x), self, diag = diag, singles = True, ov = ov))
        return Fx
 
    def F_N_no_singles(self, x, diag = False):
        self.F_count += 1
        Fx = self.arr(F_N(self.tensor(x), self, diag = diag, singles = False))
        return Fx
 
    def exact_Finv(self, x):
        Aop = LinearOperator((self.vec_size, self.vec_size), matvec = self.F_N)     
        sol, info = cg(Aop, x, tol = 1e-5, M = self.M) 
        assert(info == 0)
        return sol
 
    def shifted_H_N(self, x):
        return self.UCCSD2_H_N(x) + np.multiply(self.shift, x)

    def shifted_H_N_no_ov(self, x):
        return self.UCCSD2_H_N_no_ov(x) + np.multiply(self.shift, x)

    def acpf_H_N(self, x):
        return self.H_N(x) + self.shift*x     

    def aqcc_H_N(self, x): 
        E_c = self.g.T.dot(x)
        return self.H_N(x) - E_c*(1 - (self.N-3)*(self.N-2)/(self.N*(self.N-1)))*x

    def en_H_N(self, x):
        return self.UCCSD2_H_N(x)-2*np.multiply(np.multiply(self.g, x),x)

    def UCCSD2_H_N(self, x, ov = True):
        return self.H_N(x, ov = ov) + self.arr(UCC(self.tensor(x), self, ov = ov))

    def UCCSD2_H_N_no_ov(self, x, ov = False):
        return self.H_N(x, ov = ov) + self.arr(UCC(self.tensor(x), self, ov = ov))
         
    def Finv(self, x): 
        self.Finv_count += 1
        return np.divide(x, self.F_diag, out = np.zeros(len(x)), where = abs(self.F_diag) > 1e-12)    
 
    def var_energy(self, x, Aop, b):
        return 2*b.dot(x) + x.T.dot(Aop.dot(x)) - (4/3)*np.multiply(x, x).T.dot(np.multiply(b, x))

    def var_res_norm(self, x, Aop, b):
        return np.linalg.norm(b + Aop.dot(x))

    def callback(self, x):
        print(x)

    def jac(self, x, Aop, b):
        return 2*b + 2*Aop.dot(x) - 4*np.multiply(b, (np.multiply(x, x))) 
