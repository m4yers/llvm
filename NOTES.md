Paper vs. Reality
================

## Terminology

### Real Occurrence and Prototype
Paper does not give the name to expressions it works on, it actually does, but
"real occurrence" as they put it is more like a thing that present in the code,
but there is no name to an entity that represents several real occurrences of
the expression. I prefer to use term prototype(and not class) because this
entity is a real expression(a clone) that holds references for all necessary
operands(real occurrences) in the code but it is not present in that code.

## Expression Types(or Kind)

### Variables and Constants
Variables and Constants are expressions and require separate types but not
processed as the rest of the types.

### Ignored and Unknown
Not all expressions qualify to be processed by the pass, some we just cannot
move or even replace, some we know nothing about. These kind of expressions
require special treatment and they have types Ignored and Unknown respectfully.

### Bottom(⊥) and Top(⊤)
Bottom is used the same way as paper does it till Finalize(in code it is
roughly Factor Bottom-Up walk) phase, after all possible Bottom insertions are
done any occurrence of it as a Factor operand automatically forces this
Factor's kill, such Factor in turn also produces Bottom value. Its occurrence
as instruction's operand is an error.

Top value is not in the paper and basically acts like the Bottom value with one
critical property, if there is a real instruction that is to be replaced by a
Factor result and this Factor is to be replaced by a Top value, the real
instruction stays without a change. This is necessary to connect certain pass
steps neatly, if I find a better solution it will be removed.

## Existing PHIs
The paper does not mention how to deal with already existing PHIs that qualify
as Factors. This pass addresses the issue by trying to identify such PHIs and
assign them appropriate expression prototype. This is achieved via semi-lattice
based solver, that tries to match as many as possible PHIs to a single
expression prototype. Such Factors are called *"materialized"*.

## Cycles
Generally the algorithm in paper can handle cycles, but result usually is not
what you would expect. The pass tries more aggressive approach and recognizes
cycled Factors(details in the code) and induction expressions.

## F Operands(TODO)
In the paper the definition of the operand of an expression precedes this
expression and this forces it to have a larger version than the previous
expressions. Before such expression definition and after such operand
definition the expression has version ⊥. Basically every new operand definition
forces version increment for the expression. Well, this is still a TODO.

Though I have some doubts it is really necessary. Potentially this approach
will combine several distinct expressions under a single prototype if we just
use operand protos and not their real llvm-instructions, but you cannot freely
substitute these expressions with one another, you still restricted by the real
instruction operands they use.

Also, in the paper, versions appear neatly one after another, in reality
nothing forbids versions to exist in parallel. While this is fine if we use
real instructions as operands, handling it with prototypes and operand versions
creates certain challenges:
```
          ---------------
            a1 <-
            b1 <- a1 + 1
            c1 <- b1 + 2
          ---------------
              /     \
  --------------  --------------
   a2 <-           a3 <-
   b2 <- a2 + 1    b3 <- a3 + 1
  --------------  --------------
              \     /
          ---------------
           b4 <- F(b2,b3)
           c? <- b1 + 2
          ---------------
```
Following the paper's algorithm the second **c** will get version **2** because
of its operand b definition right before it. Using just instructions it will
get version **1**.

## Available Definitions, Save and Restore
In the Finalize step the algorithm populates AvailDef table and then sets Save
and Restore flags, succeeding CodeMotion step supposed to preserve/delete
instruction according to these. As was mentioned before multiple versions of
the same expression can be live at one point in a program which makes using
these constructions a bit problematic. I use substitution chains that gives a
more relaxed notion of available definitions and the Save flag is now a
counter, that stores the number an instruction is actually used, if it reaches
**0** the instruction is deleted.



Current State
=============
In the current state the pass cannot offer much since it does only move simple
instructions, but it is easy to extend it. It will require specialized
knowledge of other instruction types and whether it can actually move them or
even delete.

## Pass does not move(basically TODO)
 - Memory operations
 - Function calls
 - Everything it does not know about


Benchmarks
==========
Well, since the pass presents a quite humble set of instructions it can work
with at the moment, the results are also feeble. Some result you can spot only
for previously non-optimized code. You can still get some speed improvements
for already optimized code but those are well within noise range (0.5-1% I
think), and it even makes it worse for some tests.

## NBench

Processor: 2.13 GHz Intel Core 2 Duo<br>
Memory:    4 GB 1067 MHz DDR3<br>
Clang:     5.0.0 (https://github.com/llvm-mirror/clang.git ba524690b91049ad6491d1639d500eb787607c1d)<br>

**More is better**(Iter/sec)

### -O0

```
                  |             -  :                       -  |
------------------+----------------+--------------------------+
NUMERIC SORT      |        410.43  :          413.24(+00.7%)  |
STRING SORT       |        145.49  :          145.68(+00.1%)  |
BITFIELD          |    1.7614e+08  :      1.7592e+08(-00.1%)  |
FP EMULATION      |        29.049  :          29.058( 00.0%)  |
FOURIER           |         40770  :           45511(+11.6%)  |
ASSIGNMENT        |        12.241  :          12.271(+00.3%)  |
IDEA              |        1358.8  :          1358.8( 00.0%)  |
HUFFMAN           |        615.24  :          654.71(+06.4%)  |
NEURAL NET        |        12.273  :          12.339(+00.5%)  |
LU DECOMPOSITION  |        353.82  :           357.5(+01.0%)  |
```

### -O2

```
                  |             -  :            before adce  :          after adce   |
------------------+----------------+-------------------------:-----------------------+
NUMERIC SORT      |        1503.9  :         1508.1(+00.3%)  :       1507.5(+00.2%)  |
STRING SORT       |        260.06  :         261.01(+00.4%)  :       261.45(+00.5%)  |
BITFIELD          |    3.9668e+08  :     3.9676e+08( 00.0%)  :     3.97e+08(+00.1%)  |
FP EMULATION      |        283.24  :         274.45(-03.1%)  :          275(-03.0%)  |
FOURIER           |         51174  :          51158( 00.0%)  :        51249(+00.2%)  |
ASSIGNMENT        |        37.893  :         37.784(-00.3%)  :       37.865(-00.1%)  |
IDEA              |        6158.1  :         6153.2(-00.1%)  :       6176.8(+00.3%)  |
HUFFMAN           |        2387.8  :         2374.4(-00.5%)  :       2386.2(-00.1%)  |
NEURAL NET        |         47.76  :         47.755( 00.0%)  :       47.806(+00.1%)  |
LU DECOMPOSITION  |        1590.6  :         1593.7(+00.2%)  :       1588.2(-00.2%)  |
```
