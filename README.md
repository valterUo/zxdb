# ZXdb: scalable ZX-calculus on graph databases

## Installation

## Styles

Apply the following style definitions for MemGraph to match style with PyZX:

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
```

## Experimental evaluation ideas

Experimental evaluation could have at least two dimensions: storing large circuits and processing them.

### Storage
* Import time
* Database size with respect to size of the circuits
* Comparing the size of graphs to Networkx, RustworkX, QASM files
* Export time
* What other data processing related metrics could be measured?

### Processing
* What are the benchmarks here?
* Real use cases such as QAOA or VQE compilation?

## Open questions
* How to perform qubit mapping on a graph database?

## TODO
* Circuit extraction algorithm, Backens. There and back again.
* Basic rewrite rules
* 


## Ideas on formalizing ZX-calculus rules with categorical spans (has this already been done?)

[Sesqui-pushout rewriting](https://www.ti.inf.uni-due.de/publications/koenig/icgt06b.pdf) could be applied here (see Def. 1).

* ZX rewrite rules. These are not formally proper graph rewriting rules because the three dots (...) necessarily make the diagrams shorthand notation which represents a larger set of rules.
* Identify the key primitive rules in these ''derived'' basic rules:
  * Primitive rules on nodes:
    * Node removal
    * Node creation
  * Primitive rules on edges:
    * Edge removal
    * Edge creation
    * Edge source/target update, i.e., redirecting

Rewrite rules can be found https://arxiv.org/pdf/2012.13966

