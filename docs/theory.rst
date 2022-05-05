Theory
======
The energy of Unitary Coupled Cluster with Singles and Doubles (UCCSD) theory can be defined by minimizing the energy of the functional:

.. math:: E = \bra{\Psi}\hat{H}\ket{\Psi}

where the ansatz is given as:

.. math:: \ket{\Psi} = \exp\left(\sum_{ia}t_i^a\left(\hat{a}_i^a - \hat{a}_a^i\right) + \sum_{\substack{i<j\\a<b}}t_{ij}^{ab}\left(\hat{a}_{ij}^{ab} - \hat{a}_{ab}^{ij}\right)\right)\ket{\phi_0}

One can approximate the UCCSD energy functional with a Taylor series:

.. math:: E = E_0 + \sum_\alpha t_\alpha \frac{\partial E}{\partial t_\alpha} + \frac{1}{2}\sum_{\alpha,\beta}t_\alpha t_\beta \frac{\partial^2 E}{\partial t_\alpha \partial t_\beta} + \dots

Truncating this expression at second order and variationally minimizing it gives the O2D2-UCCSD method.  Inclusion of the "diagonal" terms to third order gives the O2D3-UCCSD method.  Inclusion of the "diagonal" terms to full order gives the O2D∞-UCCSD method.  For further explanation, please consult our paper.
