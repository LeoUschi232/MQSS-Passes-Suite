#!/usr/bin/env python

'''
This example collects tricks that can be used in the pyscf input script.
'''

import pyscf

from pyscf import __all__

mol = pyscf.gto.M(atom='H 0 0 0; F 0 0 1.1', basis='6-311g')
print(mol.HF())
print(mol.KS().ddCOSMO())
print(mol.TDHF())
print(mol.MP2(frozen=2))

mol.RHF().run(conv_tol=1e-7).MP2(frozen=2).run(max_memory=100).Gradients().run()

mol.KS() \
    .set(conv_tol=1e-6, xc='blyp') \
    .density_fit() \
    .apply(pyscf.scf.addons.remove_linear_dep_) \
    .run() \
    .TDA() \
    .run(nstates=5)

hf_scan = mol.RHF().as_scanner()
hf_scan(mol)
hf_scan('H 0 0 -1; F 0 0 1')
hf_grad_scan = mol.RHF().nuc_grad_method().as_scanner()
hf_grad_scan(mol)

de = mol.RHF().nuc_grad_method()
geom_opt = mol.RHF().nuc_grad_method().optimizer()
geom_opt.run()
