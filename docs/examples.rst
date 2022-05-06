Examples
========

To build the molecule object, we need some basic input information:
::

    geom = "H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3"
    basis = "sto-3g"
    reference = "rhf"
    mol = molecule(geom, basis, reference)
    
Once our molecule object is constructed, we can call our different methods like so:
::

    o2d2_E, o2d2_x = mol.o2d2_uccsd()
    o2d3_E, o2d3_x = mol.o2d3_uccsd()
    o2di_E, o2di_x = mol.o2di_uccsd()

For more involved examples, consult the "examples" directory.    
