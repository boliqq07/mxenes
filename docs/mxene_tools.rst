Tools of MXene
==============

mxene includes some tools for batch generation of mxene structures and extraction of crystal structure features, which
can be used as the input of VASP for subsequent calculation, and can also analyze and extract some calculation results
to achieve data mining in universities. In addition, mxene can also be used to check the rationality of vasp calculation
process and results.

=============================================================== ========================================================
Name                                                            Application
_______________________________________________________________ ________________________________________________________
:class:`mxene.core.mxenes.MXene`                                MXene structure object.
:func:`mxene.core.mxenes.MXene.get_similar_layer_atoms`         Get all same layer atoms by z0 site.
:func:`mxene.core.mxenes.MXene.get_next_layer_sites_xy`         Obtain the atomic position of the next layer.
:func:`mxene.core.mxenes.MXene.from_standard`                   Generate ideal single atom doping MXenes.
:func:`mxene.core.mxenes.MXene.get_structure_message`           Obtaining bond and face information
:func:`mxene.core.mxenes.MXene.add_absorb`                      Add adsorbent atoms.
:func:`mxene.core.mxenes.MXene.add_face_random`                 Add atoms at randomly.
:func:`mxene.core.mxenes.MXene.non_equivalent_site`             Obtain 16 equivalent positions.
=============================================================== ========================================================