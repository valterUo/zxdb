# ZXdb: ZX-calculus on graph databases

This implementation provides the core simplification rules from PyZX implemented as openCypher queries:
* Spider fusion
* Identity removal
* Local complementation
* Gadget fusion
* Pivot rule
* Pivot gadget rule
* Pivot boundary rule
* Bialgebra simplification

Queries can be found from ``query_collections`` folder. It is easiest to install Memgraph and import the collections to Memgraph Lab. Then one can demonstrate the usage of the platform with the instances that are in the ``test`` folder.

This package also translates ZX-diagrams into Quimb tensor networks. It seems like this way, one can simulate a lot larger diagrams than with the tensor networks in PyZX. For this, see ``demo_quimb.ipynb``.

## Styles for MemGraph Lab

Apply the following style definitions in MemGraph Leb to match the style with PyZX:

```
@NodeStyle Equals(Property(node, "t"), 0) {
  color: #000000
  color-hover: #000000
  color-selected: #000000
}

@NodeStyle Equals(Property(node, "t"), 1) {
  color: #d6fccf
  color-hover: #d6fccf
  color-selected: #d6fccf
}

@NodeStyle Equals(Property(node, "t"), 2) {
  color: #f0908a
  color-hover: #f0908a
  color-selected: #f0908a
}

@NodeStyle Equals(Property(node, "t"), 3) {
  color: #ffd700
  color-hover: #ffd700
  color-selected: #ffd700
}
```

## Notes on differences compared to PyZX:

* There is possibility to perform spider fusion when red and green spider are connected with a Hadamard edge. As far as I know, ``spider_simp`` in PyZX does not detect these cases. Evaluating these cases is not implemented but it is noted.
* ``Bialg_simp`` is not clearly explained in the PyZX. I understand that it is the ''inverse'' of the bialgebra rule applied to all non-interacting instances.
* ``gadget_simp`` might have some side effects that depend on the phases?