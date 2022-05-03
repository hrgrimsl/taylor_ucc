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
    """
    A class to represent a molecule.

    ...

    Attributes
    ----------

    vec_structure : list
        List of number of alpha singles, beta singles, aa doubles, ab doubles, and bb doubles respectively

    hf, ccsd_energy, ccsdt_energy : float
        HF, CCSD, and CCSD(T) energies

    Fa, Fb, Iaa, Iab, Ibb : numpy array
        Alpha and Beta Fock matrices and the aa, ab, and bb 2-electron repulsion integrals

    noa, nob, nva, nvb : int
        Number of alpha electrons, beta electrons, alpha virtuals, and beta virtuals

    Ca, Cb : numpy array
        Alpha and beta MO coefficients

    N : int
        Total number of electrons

    reference : string
        Reference (RHF is the only one that I am convinced works right now.)
    
    vec_size : total number of excitations

    Finv_count, F_count, H_count:  int
         Counts of F inverse, F, and H applications
   
    ga, gb, gaa, gab, gbb: numpy array
         1- and 2- electron (antisymmetrized) integrals
    
    g:  list
        List of ga-gbb

    F_diag, H_N_diag : numpy array
        Diagonal part of F and H_N

    Finv_op: LinearOperator
        Inverse Fock operator

    n_singles, n_doubles: int
        Number of single and double exciations
       


    Methods
    -------
    canonical_mp2()
        Computes the MP2 energy assuming canonical orbitals

    hylleraas_mp2()
        Computes the MP2 energy iteratively

    cisd()
        Computes the CISD energy

    lccsd(tol = 1e-6):
        Computes the LCCSD energy.
    
    o2d2_uccsd(tol = 1e-6, trotter = False):
        Computes the O2D2_UCCSD energy.

    o2d3_uccsd(guess = 'hf', trotter = False, tol = 1e-5):
        Computes the O2D3_UCCSD energy.

    o2di_uccsd(guess = 'hf', trotter = False, tol = 1e-5):
    """

    def __init__(self, geometry, basis, reference, charge = 0, unpaired = 0, conv_tol = 1e-12, read = False, ccsd = False, ccsdt = False, chkfile = None, semi_canonical = False, manual_C = None, loc = False):

        #Git info dump
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        print(f"Git revision:\ngithub.com/hrgrimsl/PYSCF_UCC/commit/{sha}")

        #PySCF computation
        self.vec_structure, self.hf, self.ccsd_energy, self.ccsdt_energy, self.Fa, self.Fb, self.Iaa, self.Iab, self.Ibb, self.noa, self.nob, self.nva, self.nvb, self.Ca, self.Cb = integrals(geometry, basis, reference, charge, unpaired, conv_tol, read = read, do_ccsd = ccsd, do_ccsdt = ccsdt, chkfile = chkfile, semi_canonical = semi_canonical, manual_C = manual_C)

        self.N = self.noa + self.nob
        self.reference = reference
        self.vec_size = 0
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
        self.Finv_op = LinearOperator((self.vec_size, self.vec_size), matvec = self.Finv, rmatvec = self.Finv)
        n_aa = self.noa*self.nva
        n_bb = self.nob*self.nvb
        n_aaaa = int(.25*self.noa*(self.noa-1)*self.nva*(self.nva-1))
        n_bbbb = int(.25*self.nob*(self.nob-1)*self.nvb*(self.nvb-1))
        n_abab = int(self.noa*self.nob*self.nva*self.nvb)
        self.n_singles = n_aa + n_bb
        self.n_doubles = n_aaaa + n_bbbb + n_abab
        assert len(self.g) == self.n_singles + self.n_doubles
        print(f"Reference Energy (a.u.):  {self.hf:16.12}") 
        print(f"Single excitations:       {self.n_singles:16d}")
        print(f"Alpha:                    {n_aa:16d}")
        print(f"Beta:                     {n_aa:16d}")
        print(f"Double excitations:       {self.n_doubles:16d}")
        print(f"aaaa:                     {n_aaaa:16d}")
        print(f"abab:                     {n_abab:16d}")
        print(f"bbbb:                     {n_bbbb:16d}")
    
    #ACTUAL CHEMISTRY METHODS
    #------------------------

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

        x, info = cg(Aop, -self.g, tol = tol, M = self.Finv_op, callback = self.lccsd_cb)         
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

    def cisd(self):
        print("Doing CISD.")
        #Do CISD
        Aop = LinearOperator((self.vec_size+1, self.vec_size+1), matvec = self.CISD_H, rmatvec = self.CISD_H)
        E, v = scipy.sparse.linalg.eigsh(Aop, k = 1, which = 'SA')
        v = v[1:,0]/v[0]
        print(f"CISD energy:  {E[0]:20.16f}")
        return E[0], v

    def lccsd(self, tol = 1e-6):        
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

    def o2d2_uccsd(self, tol = 1e-6, trotter = False):
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
        x, info = cg(Aop, -self.g, maxiter = 250, tol = tol, M = self.Finv_op, callback = self.lccsd_cb)

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

    def o2d3_uccsd(self, guess = 'hf', trotter = False, tol = 1e-5):
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

        info = minimize(self.o2d3_uccsd_energy, x, jac = self.o2d3_uccsd_grad, callback = self.o2d3_uccsd_cb, method = "l-bfgs-b", options = {'maxiter': 500, 'disp': True, 'gtol': tol, 'ftol': 0})
        if info.success == False:
            print("Iterations did not converge.")
            return float('NaN'), 'mp2' 

        energy = info.fun
        x = info.x
        print("Final callback")
        self.o2d3_uccsd_cb(x)
        print('\n')
        return energy, x 

    def o2di_uccsd(self, guess = 'hf', trotter = False, tol = 1e-5):
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
            E, x = self.o2d3_uccsd(trotter = trotter)
        else:
            x = copy.copy(guess)
        
        info = minimize(self.o2di_uccsd_energy, x, jac = self.o2di_uccsd_grad, callback = self.o2di_uccsd_cb, method = "L-BFGS-B", options = {'disp': True, 'gtol': tol, 'maxls': 500, 'ftol': 0})          
        if info.success == False:
            print("HATERI did not converge.")
            return float("NaN"), "mp2" 
        energy = info.fun
        x = info.x
        print("Final callback")
        self.o2di_uccsd_cb(x)
        return energy, x 





    #CALLBACK FUNCTIONS
    #------------------
 
    def lccsd_cb(self, x):
        self.iteration += 1
        self.times.append(time.time())
        energy = self.hf + self.g.T.dot(x)
        frame = inspect.currentframe().f_back
        resid_norm = np.linalg.norm(frame.f_locals['resid']) 
        print(f"{self.iteration:8d} | {energy:16.10f}  | {resid_norm:16.5e} | {self.times[-1] - self.times[-2]:14.2f} |")
        self.resid_norm = resid_norm
        return energy
    
    def o2d3_uccsd_cb(self, x):
        self.iteration += 1
        self.times.append(time.time())
        energy = self.o2d3_uccsd_energy(x)
        grad = self.o2d3_uccsd_grad(x)
        print(f"\nIteration: {self.iteration}", flush = True)
        print(f"Energy: {energy}", flush = True)
        print(f"Gradient norm: {np.linalg.norm(grad)}", flush = True)
        return energy

    def o2di_uccsd_cb(self, x):
        self.iteration += 1
        self.times.append(time.time())
        energy = self.o2di_uccsd_energy(x)
        grad = self.o2di_uccsd_grad(x)
        print(f"\nIteration: {self.iteration}")
        print(f"Energy: {energy}")
        print(f"Gradient norm: {np.linalg.norm(grad)}")
        return energy 



    #ENERGIES AND GRADIENTS
    #----------------------

    def o2d3_uccsd_energy(self, x):
        E = self.hf + 2*self.g.T.dot(x) + x.T@(self.UCCSD_2_A(x)) - (4/3)*self.g.T@(x*x*x)
        return E

    def o2d3_uccsd_grad(self, x):
        grad = 2*self.g + 2*self.UCCSD_2_A(x) - 4*self.g*x*x
        return grad

    def o2di_uccsd_energy(self, x):
        E = self.hf + self.g.T@np.sin(2*x) + self.H_N_diag.T@(np.sin(x)*np.sin(x)-x*x) + x.T@self.UCCSD_2_A(x)
        return E

    def o2di_uccsd_grad(self, x):
        grad = 2*self.UCCSD_2_A(x) + 2*self.g*np.cos(2*x) + self.H_N_diag*(np.sin(2*x)-2*x)
        return grad 




    #MATRIX ACTIONS
    #--------------


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

    def UCCSD_2_A(self, x):
        if self.trotter == 'sd':
            return self.UCCSD2_H_N_no_ov(x)
        else:
            return self.UCCSD2_H_N(x)

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
        sol, info = cg(Aop, x, tol = 1e-5, M = self.Finv_op) 
        assert(info == 0)
        return sol
 
    def UCCSD2_H_N(self, x, ov = True):
        return self.H_N(x, ov = ov) + self.arr(UCC(self.tensor(x), self, ov = ov))

    def UCCSD2_H_N_no_ov(self, x, ov = False):
        return self.H_N(x, ov = ov) + self.arr(UCC(self.tensor(x), self, ov = ov))
         
    def Finv(self, x): 
        self.Finv_count += 1
        return np.divide(x, self.F_diag, out = np.zeros(len(x)), where = abs(self.F_diag) > 1e-12)    





    #MP2 Natural Orbital Stuff - May be broken, hard to say?
    #-------------------------------------------------------

        
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



    #ARRAY/TENSOR CONVERSIONS
    #------------------------ 
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


