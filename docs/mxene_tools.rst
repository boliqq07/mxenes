Tools of MXene
===============

:class:`mxene.core.mxenes.MXene` are inherited from ``Structure`` from pymatgen,
beside the basic functions of ``Structure``, it customizes some specialized functions for **MXenes** structure.


=============================================================== ========================================================
 Name                                                           Application
--------------------------------------------------------------- --------------------------------------------------------
 :class:`mxene.core.mxenes.MXene`                                MXene structure object.
 :func:`mxene.core.mxenes.MXene.get_similar_layer_atoms`         Get all same layer atoms by z0 site.
 :func:`mxene.core.mxenes.MXene.get_next_layer_sites_xy`         Obtain the atomic position of the next layer.
 :func:`mxene.core.mxenes.MXene.from_standard`                   Generate ideal single atom doping MXenes.
 :func:`mxene.core.mxenes.MXene.get_structure_message`           Obtaining bond and face information
 :func:`mxene.core.mxenes.MXene.add_absorb`                      Add adsorbent atoms.
 :func:`mxene.core.mxenes.MXene.add_face_random`                 Add atoms at randomly.
 :func:`mxene.core.mxenes.MXene.non_equivalent_site`             Obtain 16 equivalent positions.
=============================================================== ========================================================


.. note::
    The background of **MXene**  refers to :doc:`Guide/background`

    The background and data of ``Structure``  refer to :doc:`Guide/mxene_materials`