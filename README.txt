[Work In Progress]
Low Level Virtual Machine (LLVM)
SSA Partial Redundancy Elimination pass
================================

The pass is based on paper "A new algorithm for partial redundancy elimination
based on SSA form" by Fred Chow, Sun Chan, Robert Kennedy Shin-Ming Liu,
Raymond Lo and Peng Tu.

There are some adjustments been made in respect to LLVM realities. A few 
notions in the algorithms also been augmented due to some vagueness of their
meaning in the paper:
  - Real Occurance, the phrase is used to signify two concepts: a definition and
    a use of this definition. Any definition without a use is a dead code and
    will be removed. A use must be regarded as an implicit definition just before
    the use(if there is none in the block of use) and then the actual use of the
    defined value

There are a few entities(or cases) omitted from the paper:
  - It is quite possible to already have a "materialized" Factor in a form of a
    PHI function that joins results of the two expressions of the same type(class).
    Thus every Factor must have a predicate 'Materialized' which indicates a 
    presence of such PHI function and a certain link between them.
  - The second case directly follows from the first. Though there might be many
    PHI functins of this kind we might not have a matching Factor for them, e.g.:

    -----------    -----------
     y <-           x <-
     a = x + y      a = x + y
     x <-           y <-
     b = x + y      b = x + y
    -----------    -----------
             \      /
              \    /
           ------------
            phi(a,a) <- Shadowed
            phi(b,b) <- Factored
           ------------

    The Factor of a's will be shadowed by the Factor of b's since b will be at
    top of the renaming stack by the time we reach the join. This can be result
    of another optimization that reduced previously different expressions into
    a same one. Here we need to create a Factor in reverse order to include
    the earlier expressions into next processing steps.


Predicates
==========

1. DownSafety or Anticibability

