# ZXdb: scalable ZX-calculus on graph databases

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