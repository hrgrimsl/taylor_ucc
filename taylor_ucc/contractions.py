import numpy as np
import copy
from opt_einsum import contract 


def H_N_diag(tensor, mol):
    xa, xb, xaa, xab, xbb = tensor
    sa = 0*xa
    sb = 0*xb
    saa = 0*xaa
    sab = 0*xab
    sbb = 0*xbb  
    Fa = copy.copy(mol.Fa)
    Fb = copy.copy(mol.Fb)

    #<ijab|F_N|ijab>
    saa += contract('bb,ijab->ijab', Fa[mol.noa:, mol.noa:], xaa)
    saa += contract('aa,ijab->ijab', Fa[mol.noa:, mol.noa:], xaa)

    sab += contract('aa,ijab->ijab', Fa[mol.noa:, mol.noa:], xab)
    sab += contract('bb,ijab->ijab', Fb[mol.nob:, mol.nob:], xab)

    sbb += contract('bb,ijab->ijab', Fb[mol.nob:, mol.nob:], xbb)
    sbb += contract('aa,ijab->ijab', Fb[mol.nob:, mol.nob:], xbb)
    
    saa -= contract('jj,ijab->ijab', Fa[:mol.noa,:mol.noa], xaa) 
    saa -= contract('ii,ijab->ijab', Fa[:mol.noa,:mol.noa], xaa) 

    sab -= contract('jj,ijab->ijab', Fb[:mol.nob,:mol.nob], xab) 
    sab -= contract('ii,ijab->ijab', Fa[:mol.noa,:mol.noa], xab) 

    sbb -= contract('jj,ijab->ijab', Fb[:mol.nob,:mol.nob], xbb) 
    sbb -= contract('ii,ijab->ijab', Fa[:mol.nob,:mol.nob], xbb) 

    #<ia|F_N T1|ia>
    sa += contract('aa,ia->ia', Fa[mol.noa:, mol.noa:], xa)
    sa -= contract('ii,ia->ia', Fa[:mol.noa, :mol.noa], xa)
    
    sb += contract('aa,ia->ia', Fb[mol.nob:, mol.nob:], xb)
    sb -= contract('ii,ia->ia', Fb[:mol.nob, :mol.nob], xb)

    #V_N
    #xS->sS
    sa += contract('iaai,ia->ia', mol.gaa[:mol.noa, mol.noa:, mol.noa:, :mol.noa], xa)
    sb += contract('iaai,ia->ia', mol.gbb[:mol.nob, mol.nob:, mol.nob:, :mol.nob], xb)

 
    #xD->sD
    saa += contract('abab,ijab->ijab', mol.gaa[mol.noa:, mol.noa:, mol.noa:, mol.noa:], xaa)
    sab += contract('abab,ijab->ijab', mol.Iab[mol.noa:, mol.nob:, mol.noa:, mol.nob:], xab)    
    sbb += contract('abab,ijab->ijab', mol.gbb[mol.nob:, mol.nob:, mol.nob:, mol.nob:], xbb)

    saa -= contract('ajaj,ijab->ijab', mol.gaa[mol.noa:, :mol.noa, mol.noa:, :mol.noa], xaa)
    saa -= contract('aiai,ijab->ijab', mol.gaa[mol.noa:, :mol.noa, mol.noa:, :mol.noa], xaa)
    saa -= contract('bibi,ijab->ijab', mol.gaa[mol.noa:, :mol.noa, mol.noa:, :mol.noa], xaa)
    saa -= contract('bjbj,ijab->ijab', mol.gaa[mol.noa:, :mol.noa, mol.noa:, :mol.noa], xaa)

    sab -= contract('ajaj,ijab->ijab', mol.Iab[mol.noa:, :mol.nob, mol.noa:, :mol.nob], xab)
    sab -= contract('bjbj,ijab->ijab', mol.gbb[mol.nob:, :mol.nob, mol.nob:, :mol.nob], xab)
    sab -= contract('aiai,ijab->ijab', mol.gaa[mol.noa:, :mol.noa, mol.noa:, :mol.noa], xab)
    sab -= contract('ibib,ijab->ijab', mol.Iab[:mol.noa, mol.nob:, :mol.noa, mol.nob:], xab)

    sbb -= contract('aiai,ijab->ijab', mol.gbb[mol.nob:, :mol.nob, mol.nob:, :mol.nob], xbb)
    sbb -= contract('ajaj,ijab->ijab', mol.gbb[mol.nob:, :mol.nob, mol.nob:, :mol.nob], xbb)
    sbb -= contract('bibi,ijab->ijab', mol.gbb[mol.nob:, :mol.nob, mol.nob:, :mol.nob], xbb)
    sbb -= contract('bjbj,ijab->ijab', mol.gbb[mol.nob:, :mol.nob, mol.nob:, :mol.nob], xbb)

    saa += contract('ijij,ijab->ijab', mol.gaa[:mol.noa, :mol.noa, :mol.noa, :mol.noa], xaa)
    sab += contract('ijij,ijab->ijab', mol.Iab[:mol.noa, :mol.nob, :mol.noa, :mol.nob], xab)
    sbb += contract('ijij,ijab->ijab', mol.gbb[:mol.nob, :mol.nob, :mol.nob, :mol.nob], xbb)

    return sa, sb, saa, sab, sbb

def F_N(tensor, mol, diag = False, singles = True, ov = False):
    xa, xb, xaa, xab, xbb = tensor
    sa = 0*xa
    sb = 0*xb
    saa = 0*xaa
    sab = 0*xab
    sbb = 0*xbb  
    Fa = copy.copy(mol.Fa)
    Fb = copy.copy(mol.Fb)

    if diag == True:
        Fa = np.diag(np.diag(Fa))
        Fb = np.diag(np.diag(Fb))

    #<ijab|F_N T2|0> 
    saa += contract('bc,ijac->ijab', Fa[mol.noa:, mol.noa:], xaa)
    saa -= contract('ac,ijbc->ijab', Fa[mol.noa:, mol.noa:], xaa)

    sab += contract('ac,ijcb->ijab', Fa[mol.noa:, mol.noa:], xab)
    sab += contract('bc,ijac->ijab', Fb[mol.nob:, mol.nob:], xab)

    sbb += contract('bc,ijac->ijab', Fb[mol.nob:, mol.nob:], xbb)
    sbb -= contract('ac,ijbc->ijab', Fb[mol.nob:, mol.nob:], xbb)
    
    saa -= contract('kj,ikab->ijab', Fa[:mol.noa,:mol.noa], xaa) 
    saa -= contract('ki,kjab->ijab', Fa[:mol.noa,:mol.noa], xaa) 

    sab -= contract('kj,ikab->ijab', Fb[:mol.nob,:mol.nob], xab) 
    sab -= contract('ki,kjab->ijab', Fa[:mol.noa,:mol.noa], xab) 

    sbb -= contract('kj,ikab->ijab', Fb[:mol.nob,:mol.nob], xbb) 
    sbb -= contract('ki,kjab->ijab', Fa[:mol.nob,:mol.nob], xbb) 
    
    if singles == False:
        return sa, sb, saa, sab, sbb
 
    #<ia|F_N T1|0>
    sa += contract('ab,ib->ia', Fa[mol.noa:, mol.noa:], xa)
    sa -= contract('ij,ja->ia', Fa[:mol.noa, :mol.noa], xa)
    
    sb += contract('ab,ib->ia', Fb[mol.nob:, mol.nob:], xb)
    sb -= contract('ij,ja->ia', Fb[:mol.nob, :mol.nob], xb)

    if ov == False:
        return sa, sb, saa, sab, sbb

    #<ia|F_N T2|0>
    sa += contract('jb,ijab->ia', Fa[:mol.noa, mol.noa:], xaa)
    sa += contract('jb,ijab->ia', Fb[:mol.nob, mol.nob:], xab)

    sb += contract('jb,jiba->ia', Fa[:mol.noa, mol.noa:], xab)
    sb += contract('jb,ijab->ia', Fb[:mol.nob, mol.nob:], xbb)

    #<ijab|F_N T1|0>
    saa += contract('ia,jb->ijab', Fa[:mol.noa, mol.noa:], xa)
    saa -= contract('ib,ja->ijab', Fa[:mol.noa, mol.noa:], xa)
    saa -= contract('ja,ib->ijab', Fa[:mol.noa, mol.noa:], xa)
    saa += contract('jb,ia->ijab', Fa[:mol.noa, mol.noa:], xa)

    sab += contract('ia,jb->ijab', Fa[:mol.noa, mol.noa:], xb)
    sab += contract('jb,ia->ijab', Fb[:mol.nob, mol.nob:], xa)

    sbb += contract('jb,ia->ijab', Fb[:mol.nob, mol.nob:], xb)
    sbb -= contract('ja,ib->ijab', Fb[:mol.nob, mol.nob:], xb)
    sbb -= contract('ib,ja->ijab', Fb[:mol.nob, mol.nob:], xb)
    sbb += contract('ia,jb->ijab', Fb[:mol.nob, mol.nob:], xb)

    return sa, sb, saa, sab, sbb  

def V_N(tensor, mol):
    xa, xb, xaa, xab, xbb = tensor

    sa = 0*xa
    sb = 0*xb
    saa = 0*xaa
    sab = 0*xab
    sbb = 0*xbb


    #xS->sS
    sa += contract('kaci,kc->ia', mol.gaa[:mol.noa, mol.noa:, mol.noa:, :mol.noa], xa)
    sa += contract('akic,kc->ia', mol.Iab[mol.noa:, :mol.nob, :mol.noa, mol.nob:], xb)
    sb += contract('kaci,kc->ia', mol.gbb[:mol.nob, mol.nob:, mol.nob:, :mol.nob], xb)
    sb += contract('kaci,kc->ia', mol.Iab[:mol.noa, mol.nob:, mol.noa:, :mol.nob], xa)    

    #xS->sD
    saa -= contract('akij,kb->ijab', mol.gaa[mol.noa:, :mol.noa, :mol.noa, :mol.noa], xa) 
    saa += contract('bkij,ka->ijab', mol.gaa[mol.noa:, :mol.noa, :mol.noa, :mol.noa], xa) 
    sab -= contract('akij,kb->ijab', mol.Iab[mol.noa:, :mol.nob, :mol.noa, :mol.nob], xb)
    sab -= contract('kbij,ka->ijab', mol.Iab[:mol.noa, mol.nob:, :mol.noa, :mol.nob], xa)
    sbb -= contract('akij,kb->ijab', mol.gbb[mol.nob:, :mol.nob, :mol.nob, :mol.nob], xb) 
    sbb += contract('bkij,ka->ijab', mol.gbb[mol.nob:, :mol.nob, :mol.nob, :mol.nob], xb) 

    saa += contract('abcj,ic->ijab', mol.gaa[mol.noa:, mol.noa:, mol.noa:, :mol.noa], xa)
    saa -= contract('abci,jc->ijab', mol.gaa[mol.noa:, mol.noa:, mol.noa:, :mol.noa], xa)
    sab += contract('abcj,ic->ijab', mol.Iab[mol.noa:, mol.nob:, mol.noa:, :mol.nob], xa)
    sab += contract('abic,jc->ijab', mol.Iab[mol.noa:, mol.nob:, :mol.noa, mol.nob:], xb)
    sbb += contract('abcj,ic->ijab', mol.gbb[mol.nob:, mol.nob:, mol.nob:, :mol.nob], xb)
    sbb -= contract('abci,jc->ijab', mol.gbb[mol.nob:, mol.nob:, mol.nob:, :mol.nob], xb)

    #xD->sS
    sa += .5*contract('akcd,ikcd->ia', mol.gaa[mol.noa:, :mol.noa, mol.noa:, mol.noa:], xaa)
    sa += contract('akcd,ikcd->ia', mol.Iab[mol.noa:, :mol.nob, mol.noa:, mol.nob:], xab)
    sb += .5*contract('akcd,ikcd->ia', mol.gbb[mol.nob:, :mol.nob, mol.nob:, mol.nob:], xbb)
    sb += contract('kadc,kidc->ia', mol.Iab[:mol.noa, mol.nob:, mol.noa:, mol.nob:], xab)

    sa -= .5*contract('klic,klac->ia', mol.gaa[:mol.noa, :mol.noa, :mol.noa, mol.noa:], xaa)
    sa -= contract('klic,klac->ia', mol.Iab[:mol.noa, :mol.nob, :mol.noa, mol.nob:], xab)
    sb -= .5*contract('klic,klac->ia', mol.gbb[:mol.nob, :mol.nob, :mol.nob, mol.nob:], xbb)
    sb -= contract('lkci,lkca->ia', mol.Iab[:mol.noa, :mol.nob, mol.noa:, :mol.nob], xab)

    #xD->sD
    saa += .5*contract('abcd,ijcd->ijab', mol.gaa[mol.noa:, mol.noa:, mol.noa:, mol.noa:], xaa)
    sab += contract('abcd,ijcd->ijab', mol.Iab[mol.noa:, mol.nob:, mol.noa:, mol.nob:], xab)    
    sbb += .5*contract('abcd,ijcd->ijab', mol.gbb[mol.nob:, mol.nob:, mol.nob:, mol.nob:], xbb)

    saa -= contract('akcj,ikcb->ijab', mol.gaa[mol.noa:, :mol.noa, mol.noa:, :mol.noa], xaa)
    saa += contract('akci,jkcb->ijab', mol.gaa[mol.noa:, :mol.noa, mol.noa:, :mol.noa], xaa)
    saa += contract('bkcj,ikca->ijab', mol.gaa[mol.noa:, :mol.noa, mol.noa:, :mol.noa], xaa)
    saa -= contract('bkci,jkca->ijab', mol.gaa[mol.noa:, :mol.noa, mol.noa:, :mol.noa], xaa)
    saa -= contract('akjc,ikbc->ijab', mol.Iab[mol.noa:, :mol.nob, :mol.noa, mol.nob:], xab)
    saa += contract('akic,jkbc->ijab', mol.Iab[mol.noa:, :mol.nob, :mol.noa, mol.nob:], xab)
    saa += contract('bkjc,ikac->ijab', mol.Iab[mol.noa:, :mol.nob, :mol.noa, mol.nob:], xab)
    saa -= contract('bkic,jkac->ijab', mol.Iab[mol.noa:, :mol.nob, :mol.noa, mol.nob:], xab)
    sab -= contract('akcj,ikcb->ijab', mol.Iab[mol.noa:, :mol.nob, mol.noa:, :mol.nob], xab)
    sab += contract('kbcj,ikac->ijab', mol.Iab[:mol.noa, mol.nob:, mol.noa:, :mol.nob], xaa)
    sab -= contract('bkcj,ikac->ijab', mol.gbb[mol.nob:, :mol.nob, mol.nob:, :mol.nob], xab)
    sab -= contract('akci,kjcb->ijab', mol.gaa[mol.noa:, :mol.noa, mol.noa:, :mol.noa], xab)
    sab += contract('akic,kjcb->ijab', mol.Iab[mol.noa:, :mol.nob, :mol.noa, mol.nob:], xbb)
    sab -= contract('kbic,kjac->ijab', mol.Iab[:mol.noa, mol.nob:, :mol.noa, mol.nob:], xab)
    sbb -= contract('akcj,ikcb->ijab', mol.gbb[mol.nob:, :mol.nob, mol.nob:, :mol.nob], xbb)
    sbb += contract('akci,jkcb->ijab', mol.gbb[mol.nob:, :mol.nob, mol.nob:, :mol.nob], xbb)
    sbb += contract('bkcj,ikca->ijab', mol.gbb[mol.nob:, :mol.nob, mol.nob:, :mol.nob], xbb)
    sbb -= contract('bkci,jkca->ijab', mol.gbb[mol.nob:, :mol.nob, mol.nob:, :mol.nob], xbb)
    sbb -= contract('kacj,kicb->ijab', mol.Iab[:mol.noa, mol.nob:, mol.noa:, :mol.nob], xab)
    sbb += contract('kaci,kjcb->ijab', mol.Iab[:mol.noa, mol.nob:, mol.noa:, :mol.nob], xab)
    sbb += contract('kbcj,kica->ijab', mol.Iab[:mol.noa, mol.nob:, mol.noa:, :mol.nob], xab)
    sbb -= contract('kbci,kjca->ijab', mol.Iab[:mol.noa, mol.nob:, mol.noa:, :mol.nob], xab)

    saa += .5*contract('klij,klab->ijab', mol.gaa[:mol.noa, :mol.noa, :mol.noa, :mol.noa], xaa)
    sab += contract('klij,klab->ijab', mol.Iab[:mol.noa, :mol.nob, :mol.noa, :mol.nob], xab)
    sbb += .5*contract('klij,klab->ijab', mol.gbb[:mol.nob, :mol.nob, :mol.nob, :mol.nob], xbb)
    return sa, sb, saa, sab, sbb

def UCC(tensor, mol, ov = True):
    xa, xb, xaa, xab, xbb = tensor

    sa = 0*xa
    sb = 0*xb
    saa = 0*xaa
    sab = 0*xab
    sbb = 0*xbb
  
    sa += contract('ijab,jb->ia', mol.gaa[:mol.noa, :mol.noa, mol.noa:, mol.noa:], xa)
    sa += contract('ijab,jb->ia', mol.Iab[:mol.noa, :mol.nob, mol.noa:, mol.nob:], xb)
    sb += contract('ijab,jb->ia', mol.gbb[:mol.nob, :mol.nob, mol.nob:, mol.nob:], xb)
    sb += contract('jiba,jb->ia', mol.Iab[:mol.noa, :mol.nob, mol.noa:, mol.nob:], xa)

    if ov == False:
        return sa, sb, saa, sab, sbb

    sa -= .5*contract('jb,ijab->ia', mol.Fa[:mol.noa, mol.noa:], xaa)
    sa -= .5*contract('jb,ijab->ia', mol.Fb[:mol.nob, mol.nob:], xab)
    sb -= .5*contract('jb,ijab->ia', mol.Fb[:mol.nob, mol.nob:], xbb)
    sb -= .5*contract('jb,jiba->ia', mol.Fa[:mol.noa, mol.noa:], xab)

    saa -= .5*contract('ia,jb->ijab', mol.Fa[:mol.noa, mol.noa:], xa)
    saa += .5*contract('ja,ib->ijab', mol.Fa[:mol.noa, mol.noa:], xa)
    saa += .5*contract('ib,ja->ijab', mol.Fa[:mol.noa, mol.noa:], xa)
    saa -= .5*contract('jb,ia->ijab', mol.Fa[:mol.noa, mol.noa:], xa)

    sab -= .5*contract('ia,jb->ijab', mol.Fa[:mol.noa, mol.noa:], xb)
    sab -= .5*contract('jb,ia->ijab', mol.Fb[:mol.noa, mol.noa:], xa)

    sbb -= .5*contract('ia,jb->ijab', mol.Fb[:mol.nob, mol.nob:], xb)
    sbb += .5*contract('ja,ib->ijab', mol.Fb[:mol.nob, mol.nob:], xb)
    sbb += .5*contract('ib,ja->ijab', mol.Fb[:mol.nob, mol.nob:], xb)
    sbb -= .5*contract('jb,ia->ijab', mol.Fb[:mol.nob, mol.nob:], xb)

    return sa, sb, saa, sab, sbb
         
