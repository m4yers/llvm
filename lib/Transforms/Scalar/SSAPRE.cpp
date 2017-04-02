//===---- SSAPRELegacy.cpp - SSA PARTIAL REDUNDANCY ELIMINATION -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/SSAPRE.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BreakCriticalEdges.h"
#include "llvm/Analysis/IteratedDominanceFrontier.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/CFG.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::ssapre;

#define DEBUG_TYPE "ssapre"

STATISTIC(SSAPREInstrSaved,    "Number of instructions saved");
STATISTIC(SSAPREInstrReloaded, "Number of instructions reloaded");
STATISTIC(SSAPREInstrInserted, "Number of instructions inserted");
STATISTIC(SSAPREInstrDeleted,  "Number of instructions deleted");
STATISTIC(SSAPREBlocksAdded,   "Number of blocks deleted");

// Anchor methods.
namespace llvm {
namespace ssapre {
Expression::~Expression() = default;
IgnoredExpression::~IgnoredExpression() = default;
UnknownExpression::~UnknownExpression() = default;
BasicExpression::~BasicExpression() = default;
PHIExpression::~PHIExpression() = default;
FactorExpression::~FactorExpression() = default;
}
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

// This is used as ⊥ version
Expression BExpr(ET_Buttom, ~2U, false);

std::pair<unsigned, unsigned> SSAPRE::
AssignDFSNumbers(BasicBlock *B, unsigned Start,
                 InstrToOrderType *M, OrderedInstrType *V) {
  unsigned End = Start;
  // if (MemoryAccess *MemPhi = MSSA->getMemoryAccess(B)) {
  //   InstrDFS[MemPhi] = End++;
  //   DFSToInstr.emplace_back(MemPhi);
  // }

  for (auto &I : *B) {
    if (M) (*M)[&I] = End++;
    if (V) V->emplace_back(&I);
  }

  // All of the range functions taken half-open ranges (open on the end side).
  // So we do not subtract one from count, because at this point it is one
  // greater than the last instruction.
  return std::make_pair(Start, End);
}

unsigned int SSAPRE::
GetRank(const Value *V) const {
  // Prefer undef to anything else
  if (isa<UndefValue>(V))
    return 0;
  if (isa<Constant>(V))
    return 1;
  else if (auto *A = dyn_cast<Argument>(V))
    return 2 + A->getArgNo();

  // Need to shift the instruction DFS by number of arguments + 3 to account for
  // the constant and argument ranking above.
  unsigned Result = InstrDFS.lookup(V);
  if (Result > 0)
    return 3 + NumFuncArgs + Result;
  // Unreachable or something else, just return a really large number.
  return ~0;
}

bool SSAPRE::
ShouldSwapOperands(const Value *A, const Value *B) const {
  // Because we only care about a total ordering, and don't rewrite expressions
  // in this order, we order by rank, which will give a strict weak ordering to
  // everything but constants, and then we order by pointer address.
  return std::make_pair(GetRank(A), A) > std::make_pair(GetRank(B), B);
}

bool SSAPRE::
FillInBasicExpressionInfo(Instruction &I, BasicExpression *E) {
  bool AllConstant = true;
  if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
    E->setType(GEP->getSourceElementType());
  } else {
    E->setType(I.getType());
  }

  E->setOpcode(I.getOpcode());

  for (auto &O : I.operands()) {
    if (auto *C = dyn_cast<Constant>(O)) {
      AllConstant &= true;
      // This is the first time we see this Constant
      if (!ValueToCOExp[C]) {
        auto COExp = CreateConstantExpression(*C);
        ExpToValue[COExp] = C;
        ValueToExp[C] = COExp;
        COExpToValue[COExp] = C;
        ValueToCOExp[C] = COExp;
      }
    } else {
      AllConstant = false;
    }
    E->addOperand(O);
  }

  return AllConstant;
}

bool SSAPRE::
VariableOrConstant(const Expression &E) {
  return E.getExpressionType() == ET_Variable ||
         E.getExpressionType() == ET_Constant;
}

bool SSAPRE::
NotStrictlyDominates(const Expression *Def, const Expression *Use) {
  assert (Def && Use && "Def or Use is null");
  // If the expression is a Factor we need to use the first non-phi instruction
  // of the block it belongs to
  auto IDef = FactorExpression::classof(Def)
                ? FactorToBlock[(FactorExpression *)Def]->getFirstNonPHI()
                : VExprToInst[Def];

  auto IUse = FactorExpression::classof(Use)
                ? FactorToBlock[(FactorExpression *)Use]->getFirstNonPHI()
                : VExprToInst[Use];

  assert (IDef && IUse && "IDef or IUse is null");

  // Not Strictly
  if (IDef == IUse)
    return true;

  return DT->dominates(IDef, IUse);
}

bool SSAPRE::
OperandsDominate(Expression *Exp, const FactorExpression *Factor) {
  for (auto &O : VExprToInst[Exp]->operands()) {
    auto E = ValueToExp[O];
    if (VariableOrConstant(*E)) continue;
    if (!NotStrictlyDominates(E, Factor))
      return false;
  }
  return true;
}

Expression * SSAPRE:: GetBottom() { return &BExpr; }

bool SSAPRE::
IsBottom(const Expression &E) {
  return &E == GetBottom() || VariableOrConstant(E);
}

bool SSAPRE::
FactorHasRealUse(const FactorExpression *F) {
  // If Factor is linked with a PHI we need to check its uses.
  if (auto PHI = FactorToPHI[F]) {
    if (PHI->getNumUses() != 0) {
      return true;
    }
  }

  bool HasRealUse = false;
  auto &Versions = PExprToVersions[F->getPExpr()];
  for (auto V : Versions[F->getVersion()]) {
    if (!FactorExpression::classof(V)
        && VExprToInst[V]->getNumUses() != 0) {
      HasRealUse = true;
      break;
    }
  }
  return HasRealUse;
}

Expression *SSAPRE::
CheckSimplificationResults(Expression *E, Instruction &I, Value *V) {
  if (!V)
    return nullptr;

  if (auto *C = dyn_cast<Constant>(V)) {
    DEBUG(dbgs() << "Simplified " << I << " to "
                 << " constant " << *C << "\n");
    assert(isa<BasicExpression>(E) &&
           "We should always have had a basic expression here");

    // cast<BasicExpression>(E)->deallocateOperands(ArgRecycler);
    // ExpressionAllocator.Deallocate(E);
    // return CreateConstantExpression(*C);
    // FIXME ignore contants for now
    return CreateIgnoredExpression(I);
  } else if (isa<Argument>(V) || isa<GlobalVariable>(V)) {
    DEBUG(dbgs() << "Simplified " << I << " to "
                 << " variable " << *V << "\n");
    // cast<BasicExpression>(E)->deallocateOperands(ArgRecycler);
    // ExpressionAllocator.Deallocate(E);
    return CreateVariableExpression(*V);
  }

  return nullptr;
}

ConstantExpression *SSAPRE::
CreateConstantExpression(Constant &C) {
  auto *E = new ConstantExpression(C);
  E->setOpcode(C.getValueID());
  E->setVersion(-2);
  return E;
}

VariableExpression *SSAPRE::
CreateVariableExpression(Value &V) {
  auto *E = new VariableExpression(V);
  E->setOpcode(V.getValueID());
  E->setVersion(-3);
  return E;
}

Expression * SSAPRE::
CreateIgnoredExpression(Instruction &I) {
  // auto *E = new (ExpressionAllocator) UnknownExpression(I);
  auto *E = new IgnoredExpression(&I);
  E->setOpcode(I.getOpcode());
  return E;
}

Expression * SSAPRE::
CreateUnknownExpression(Instruction &I) {
  // auto *E = new (ExpressionAllocator) UnknownExpression(I);
  auto *E = new UnknownExpression(&I);
  E->setOpcode(I.getOpcode());
  return E;
}

Expression * SSAPRE::
CreateBasicExpression(Instruction &I) {
  // auto *E = new (ExpressionAllocator) BasicExpression(I->getNumOperands());
  auto *E = new BasicExpression();

  bool AllConstant = FillInBasicExpressionInfo(I, E);

  if (I.isCommutative()) {
    // Ensure that commutative instructions that only differ by a permutation
    // of their operands get the same expression map by sorting the operand value
    // numbers.  Since all commutative instructions have two operands it is more
    // efficient to sort by hand rather than using, say, std::sort.
    assert(I.getNumOperands() == 2 && "Unsupported commutative instruction!");
    if (ShouldSwapOperands(E->getOperand(0), E->getOperand(1)))
      E->swapOperands(0, 1);
  }

  // Perform simplificaiton
  // We do not actually require simpler instructions but rather require them be
  // in a canonical form. Mainly we are interested in instructions that we
  // ignore, such as constants and variables.
  // TODO: Right now we only check to see if we get a constant result.
  // We may get a less than constant, but still better, result for
  // some operations.
  // IE
  //  add 0, x -> x
  //  and x, x -> x
  // We should handle this by simply rewriting the expression.
  if (auto *CI = dyn_cast<CmpInst>(&I)) {
    // Sort the operand value numbers so x<y and y>x get the same value
    // number.
    CmpInst::Predicate Predicate = CI->getPredicate();
    if (ShouldSwapOperands(E->getOperand(0), E->getOperand(1))) {
      E->swapOperands(0, 1);
      Predicate = CmpInst::getSwappedPredicate(Predicate);
    }
    E->setOpcode((CI->getOpcode() << 8) | Predicate);
    // TODO: 25% of our time is spent in SimplifyCmpInst with pointer operands
    assert(I.getOperand(0)->getType() == I.getOperand(1)->getType() &&
           "Wrong types on cmp instruction");
    assert((E->getOperand(0)->getType() == I.getOperand(0)->getType() &&
            E->getOperand(1)->getType() == I.getOperand(1)->getType()));
    Value *V = SimplifyCmpInst(Predicate, E->getOperand(0), E->getOperand(1),
                               *DL, TLI, DT, AC);
    if (auto *SE = CheckSimplificationResults(E, I, V))
      return SE;
  } else if (isa<SelectInst>(I)) {
    if (isa<Constant>(E->getOperand(0)) ||
        E->getOperand(0) == E->getOperand(1)) {
      assert(E->getOperand(1)->getType() == I.getOperand(1)->getType() &&
             E->getOperand(2)->getType() == I.getOperand(2)->getType());
      Value *V = SimplifySelectInst(E->getOperand(0), E->getOperand(1),
                                    E->getOperand(2), *DL, TLI, DT, AC);
      if (auto *SE = CheckSimplificationResults(E, I, V))
        return SE;
    }
  } else if (I.isBinaryOp()) {
    Value *V = SimplifyBinOp(E->getOpcode(), E->getOperand(0), E->getOperand(1),
                             *DL, TLI, DT, AC);
    if (auto *SE = CheckSimplificationResults(E, I, V))
      return SE;
  } else if (auto *BI = dyn_cast<BitCastInst>(&I)) {
    Value *V = SimplifyInstruction(BI, *DL, TLI, DT, AC);
    if (auto *SE = CheckSimplificationResults(E, I, V))
      return SE;
  } else if (isa<GetElementPtrInst>(I)) {
    Value *V = SimplifyGEPInst(E->getType(), E->getOperands(), *DL, TLI, DT, AC);
    if (auto *SE = CheckSimplificationResults(E, I, V))
      return SE;
  } else if (AllConstant) {
    // We don't bother trying to simplify unless all of the operands
    // were constant.
    // TODO: There are a lot of Simplify*'s we could call here, if we
    // wanted to.  The original motivating case for this code was a
    // zext i1 false to i8, which we don't have an interface to
    // simplify (IE there is no SimplifyZExt).

    SmallVector<Constant *, 8> C;
    for (Value *Arg : E->getOperands())
      C.emplace_back(cast<Constant>(Arg));

    if (Value *V = ConstantFoldInstOperands(&I, C, *DL, TLI))
      if (auto *SE = CheckSimplificationResults(E, I, V))
        return SE;
  }

  return E;
}

Expression *SSAPRE::
CreatePHIExpression(PHINode &I) {
  auto *E = new PHIExpression(I.getParent());
  FillInBasicExpressionInfo(I, E);
  return E;
}

FactorExpression *SSAPRE::
CreateFactorExpression(const Expression &E, const BasicBlock &B) {
  auto FE = new FactorExpression(B);
  size_t C = 0;
  for (auto S = pred_begin(&B), EE = pred_end(&B); S != EE; ++S) {
    FE->addPred(*S, C++);
  }
  FE->setPExpr(&E);

  return FE;
}

Expression * SSAPRE::
CreateExpression(Instruction &I) {
  if (I.isTerminator()) {
    return CreateIgnoredExpression(I);
  }

  Expression * E = nullptr;
  switch (I.getOpcode()) {
  case Instruction::ExtractValue:
  case Instruction::InsertValue:
    // E = performSymbolicAggrValueEvaluation(I);
    break;
  case Instruction::PHI:
    E = CreatePHIExpression(cast<PHINode>(I));
    break;
  case Instruction::Call:
    // E = performSymbolicCallEvaluation(I);
    break;
  case Instruction::Store:
    // E = performSymbolicStoreEvaluation(I);
    break;
  case Instruction::Load:
    // E = performSymbolicLoadEvaluation(I);
    break;
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::BitCast:
    E = CreateBasicExpression(I);
    break;
  case Instruction::ICmp:
  case Instruction::FCmp:
    // E = performSymbolicCmpEvaluation(I);
    break;
  case Instruction::Add:
  case Instruction::FAdd:
  case Instruction::Sub:
  case Instruction::FSub:
  case Instruction::Mul:
  case Instruction::FMul:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::FRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    E = CreateBasicExpression(I);
    break;
  case Instruction::Select:
  case Instruction::ExtractElement:
  case Instruction::InsertElement:
  case Instruction::ShuffleVector:
  case Instruction::GetElementPtr:
    E = CreateBasicExpression(I);
    break;
  default:
    E = CreateUnknownExpression(I);
  }

  if (!E)
    E = CreateUnknownExpression(I);

  return E;
}

bool SSAPRE::
IgnoreExpression(const Expression &E) {
  auto ET = E.getExpressionType();
  return ET == ET_Ignored  ||
         ET == ET_Unknown  ||
         ET == ET_Variable ||
         ET == ET_Constant;
}

void SSAPRE::
PrintDebug(const std::string &Caption) {
  dbgs() << "\n" << Caption;
  dbgs() << "--------------------------------------";
  dbgs() << "\nPExprToInts\n";
  for (auto &P : PExprToInsts) {
    dbgs() << "(" << P.getSecond().size() << ")";
    auto &PE = P.getFirst();
    PE->printInternal(dbgs());
    for (auto VE : PExprToVExprs[PE]) {
      dbgs() << "\n\t";
      if (VE->getSave() || VE->getReload())
        VE->printInternal(dbgs());
      else
        dbgs() << "(deleted)";
    }
    dbgs() << "\n";
  }

  dbgs() << "\nORDERS DFS/SDFS";
  for (auto &V : DFSToInstr) {
    auto I = dyn_cast<Instruction>(V);
    dbgs() << "\n" << InstrDFS[I];
    dbgs() << "\t" << InstrSDFS[I];
    if (ValueToExp[I]->getSave() && I->getParent())
      dbgs() << "\t" << *I;
    else
      dbgs() << "\t(deleted)";
  }

  dbgs() << "\nBlockToFactors\n";
  for (auto &P : BlockToFactors) {
    dbgs() << "(" << P.getSecond().size() << ")";
    P.getFirst()->printAsOperand(dbgs(), false);
    dbgs() << ":";
    for (const auto &F : P.getSecond()) {
      dbgs() << "\n";
      F->printInternal(dbgs());
    }
    dbgs() << "\n";
  }

  dbgs() << "\nBlockToInserts\n";
  for (auto &P : BlockToInserts) {
    dbgs() << "(" << P.getSecond().size() << ")";
    P.getFirst()->printAsOperand(dbgs(), false);
    dbgs() << ":";
    for (const auto &I : P.getSecond()) {
      auto VE = InstToVExpr[I];
      dbgs() << "\n";
      VE->printInternal(dbgs());
    }
    dbgs() << "\n";
  }

  dbgs() << "---------------------------------------------\n";
}

void SSAPRE::
Init(Function &F) {
  for (auto &A : F.args()) {
    auto VAExp = CreateVariableExpression(A);
    ExpToValue[VAExp] = &A;
    ValueToExp[&A] = VAExp;
    VAExpToValue[VAExp] = &A;
    ValueToVAExp[&A] = VAExp;
  }

  unsigned ICount = 0;
  // DFSToInstr.emplace_back(nullptr);

  DenseMap<const DomTreeNode *, unsigned> RPOOrdering;
  unsigned Counter = 0;
  for (auto &B : *RPOT) {
    if (!B->getSinglePredecessor()) {
      JoinBlocks.insert(B);
    }

    auto *Node = DT->getNode(B);
    assert(Node && "RPO and Dominator tree should have same reachability");

    // Assign each block RPO index
    RPOOrdering[Node] = ++Counter;

    // Collect all the expressions
    for (auto &I : *B) {
      // Create ProtoExpresison, this expression will not be versioned and used
      // to bind Versioned Expressions of the same kind/class.
      Expression *PE = CreateExpression(I);
      for (auto &P : PExprToInsts) {
        auto EP = P.getFirst();
        if (PE->equals(*EP))
          PE = (Expression *)EP;
      }

      if (!PE->getProto() && !IgnoreExpression(*PE)) {
        PE->setProto(I.clone());
      }
      // This is the real versioned expression
      Expression *VE = CreateExpression(I);

      assert(PE && VE && "Oh No!");

      ExpToValue[VE] = &I;
      ValueToExp[&I] = VE;
      BlockToFactors.insert({B, {}});
      InstToVExpr[&I] = VE;
      VExprToInst[VE] = &I;
      VExprToPExpr[VE] = PE;
      Substitutions[VE] = VE;

      if (!PExprToVersions.count(PE)) {
        PExprToVersions.insert({PE,
            DenseMap<unsigned,SmallPtrSet<Expression *, 5>>()});
      }

      if (!PExprToVExprs.count(PE)) {
        PExprToVExprs.insert({PE, {VE}});
      } else {
        PExprToVExprs[PE].insert(VE);
      }

      // Map Proto-to-Reals and Proto-to-Blocks
      if (!PExprToInsts.count(PE)) {
        PExprToInsts.insert({PE, {&I}});
        PExprToBlocks.insert({PE, {B}});
      } else {
        PExprToInsts[PE].insert(&I);
        PExprToBlocks[PE].insert(B);
      }
    }
  }

  // Sort dominator tree children arrays into RPO.
  for (auto &B : *RPOT) {
    auto *Node = DT->getNode(B);
    if (Node->getChildren().size() > 1) {
      std::sort(Node->begin(), Node->end(),
                [&RPOOrdering](const DomTreeNode *A, const DomTreeNode *B) {
                  return RPOOrdering[A] < RPOOrdering[B];
                });
    }
  }

  // Assign each instruction a DFS order number. This will be the main order
  // we traverse DT in.
  auto DFI = df_begin(DT->getRootNode());
  for (auto DFE = df_end(DT->getRootNode()); DFI != DFE; ++DFI) {
    auto B = DFI->getBlock();
    auto BlockRange = AssignDFSNumbers(B, ICount, &InstrDFS, &DFSToInstr);
    ICount += BlockRange.second - BlockRange.first;
  }

  // Now we need to create Reverse Sorted Dominator Tree, where siblings sorted
  // in the opposite to RPO order. This order will give us a clue, when during
  // the normal traversal we go up the tree. For example:
  //
  //   CFG:    DT:
  //
  //    a       a     RPO(CFG): { a, c, b, d, e } // normal cfg rpo
  //   / \    / | \   DFS(DT):  { a, b, d, e, c } // before reorder
  //  b   c  b  d  c  DFS(DT):  { a, c, b, d, e } // after reorder
  //   \ /      |
  //    d       e     SDFS(DT): { a, d, e, b, c } // after reverse reorder
  //    |             SDFSO(DFS(DT),SDFS(DT)): { 1, 5, 4, 2, 3 }
  //    e                                          <  >  >  <
  //
  // So this SDFSO which maps our RPOish DFS(DT) onto SDFS order gives us points
  // where we must backtrace our context(stack or whatever we keep updated).
  // These are the places where the next SDFSO is less than the previous one.
  //
  for (auto &B : *RPOT) {
    auto *Node = DT->getNode(B);
    if (Node->getChildren().size() > 1) {
      std::sort(Node->begin(), Node->end(),
                [&RPOOrdering](const DomTreeNode *A, const DomTreeNode *B) {
                  // NOTE here we are using the reversed operator
                  return RPOOrdering[A] > RPOOrdering[B];
                });
    }
  }

  // Calculate Instruction-to-SDFS map
  ICount = 1;
  DFI = df_begin(DT->getRootNode());
  for (auto DFE = df_end(DT->getRootNode()); DFI != DFE; ++DFI) {
    auto B = DFI->getBlock();
    auto BlockRange = AssignDFSNumbers(B, ICount, &InstrSDFS, nullptr);
    ICount += BlockRange.second - BlockRange.first;
  }
}

// FIXME Do proper memory management
void SSAPRE::
Fini() {
  JoinBlocks.clear();

  ExpToValue.clear();
  ValueToExp.clear();

  VAExpToValue.clear();
  ValueToVAExp.clear();

  COExpToValue.clear();
  ValueToCOExp.clear();

  InstrDFS.clear();
  InstrSDFS.clear();

  FactorToPHI.clear();
  PHIToFactor.clear();
  DFSToInstr.clear();

  InstToVExpr.clear();
  VExprToInst.clear();
  VExprToPExpr.clear();
  PExprToVersions.clear();
  PExprToInsts.clear();
  PExprToBlocks.clear();
  PExprToVExprs.clear();

  BlockToFactors.clear();
  FactorToBlock.clear();

  FExprs.clear();

  AvailDef.clear();
  BlockToInserts.clear();
  Substitutions.clear();
  ReloadList.clear();
  KillList.clear();
}

void SSAPRE::
FactorInsertion() {
  // We examine each join block first to find already "materialized" Factors
  for (auto B : JoinBlocks) {
    for (auto &I : *B) {
      // When we reach first non-phi instruction we stop
      if (&I == B->getFirstNonPHI()) break;

      if (auto *PHI = dyn_cast<PHINode>(&I)) {
        const Expression * PE = nullptr;
        bool Same = true;
        for (auto &Use : PHI->operands()) {
          auto UVE = ValueToExp[Use.get()];

          // A variable or a constant regarded as ⊥ value
          if (VariableOrConstant(*UVE)) continue;

          auto UPE = VExprToPExpr[UVE];

          // If the first operand, just set PE and continue
          if (!PE) {
            PE = UPE;
            continue;
          }

          // The Protos mismatch, nothing to do here
          if (PE != UPE) {
            Same = false;
            break;
          }
        }

        // If all the operands are the same we have a materialized Factor here,
        // so we need to create a Factor instance and link it with this PHI.
        if (Same) {
          auto F = CreateFactorExpression(*PE, *B);

          F->setIsLinked(true);
          F->setPExpr(PE);
          FactorToPHI[F] = PHI;
          PHIToFactor[PHI] = F;

          // Set already know expression versions
          for (unsigned i = 0, l = PHI->getNumOperands(); i < l; ++i) {
            auto B = PHI->getIncomingBlock(i);
            auto J = F->getPredIndex(B);
            auto UVE = ValueToExp[PHI->getOperand(i)];
            F->setVExpr(J, UVE);

            // ??? Careful here, this might no be the total case
            // // This must be init value, thus it is a cycle
            // if (VariableOrConstant(*UVE))
            //   F->setIsCycle(true);
          }

          BlockToFactors[B].insert({F});
          FactorToBlock[F] = B;
          Substitutions[F] = F;
          FExprs.insert(F);
        }
      }
    }
  }

  // Factors are inserted in two cases:
  //   - for each block in expressions IDF
  //   - for each phi of expression operand, which indicates expression alteration
  for (auto &P : PExprToInsts) {
    auto &PE = P.getFirst();
    if (IgnoreExpression(*PE) || PHIExpression::classof(PE))
      continue;

    // Each Expression occurrence's DF requires us to insert a Factor function,
    // which is much like PHI function but for expressions.
    SmallVector<BasicBlock *, 32> IDF;
    ForwardIDFCalculator IDFs(*DT);
    IDFs.setDefiningBlocks(PExprToBlocks[PE]);
    // IDFs.setLiveInBlocks(BlocksWithDeadTerminators);
    IDFs.calculate(IDF);

    for (const auto &B : IDF) {
      // We only need to insert a Factor at a merge point when it reaches a later
      // occurance(a DEF at this point) of the expression. A later occurance will
      // have a bigger DFS number;
      bool ShouldInsert = false;
      // Starting DFS
      auto DFS = InstrDFS[&B->front()];
      // FIXME remove the cycle
      while (DFS < DFSToInstr.size()) {
        auto I = DFSToInstr[DFS++];
        auto OE = ValueToExp[I];
        auto OPE = VExprToPExpr[OE];

        // If Proto of the occurance matches the PE we should insert here
        if (OPE == PE) {
          ShouldInsert = true;
          break;
        }
      }

      // If we do not insert just continue
      if (!ShouldInsert) continue;

      // True if a Factor for this Expression with exactly the same arguments
      // exists. There are two possibilities for arguments equality, there either
      // none which means it wasn't versioned yet, or there are versions(or rather
      // expression definitions) which means they were spawned out of PHIs. We
      // are concern with the first case for now.
      bool FactorExists = false;
      // FIXME remove the cycle
      for (auto F : BlockToFactors[B]) {
        if (!F->getIsLinked() && F->getPExpr() == PE) {
          FactorExists = true;
          break;
        }
      }

      if (!FactorExists) {
        auto F = CreateFactorExpression(*PE, *B);
        BlockToFactors[B].insert({F});
        FactorToBlock[F] = B;
        Substitutions[F] = F;
        FExprs.insert(F);
      }
    }

    // TODO
    // Once operands phi-ud graphs are ready we need to traverse them to insert
    // Factors at each operands' phi definition.
    //
    // NOTE
    // That this step is before Renaming thus operands of the expression inside
    // this phi-ud graph won't have actual versions, though they do have
    // "a version" within LLVM SSA space.
    // if (const auto *BE = dyn_cast<const BasicExpression>(PE)) {
    //   for (auto &O : BE->getOperands()) {
    //     if (const auto *PHI = dyn_cast<const PHINode>(O)) {
    //       auto B = PHI->getParent();
    //       auto F = CreateFactorExpression(*PE, *B);
    //       BlockToFactors[B].insert({F});
    //     }
    //   }
    // }
  }
}

void SSAPRE::
Rename() {
  // We assign SSA versions to each of 3 kinds of expressions:
  //   - Real expression
  //   - Factor expression
  //   - Factor operands, these generally versioned as Bottom

  // The counters are used to number expression versions during DFS walk. Before
  // the renaming phase each instruction(that we do not ignore) is of a proto
  // type(PExpr), after this walk every expression is assign its own version and
  // it becomes a versioned(or instantiated) expression(VExpr).
  DenseMap<const Expression *, int> PExprToCounter;

  // Each PExpr is mapped to a stack of VExpr that grow and shrink during DFS
  // walk. Tops of these stacks are used to name a recent expression occurrence
  // as a Factor operand.
  DenseMap<const Expression *, std::stack<std::pair<unsigned, Expression *>>>
    PExprToVExprStack;

  for (auto &P : PExprToInsts) {
    auto &PE = P.getFirst();
    if (IgnoreExpression(*PE))
      continue;
    PExprToCounter.insert({PE, 0});
    PExprToVExprStack.insert({PE, {}});
  }

  for (auto B : *RPOT) {
    // Since factors live outside basic blocks we set theirs DFS as the first
    // instruction's in the block
    auto FSDFS = InstrSDFS[&B->front()];

    // Set PHI versions first, since factors regarded as occurring at the end
    // of the predecessor blocks
    for (auto &I : *B) {
      if (&I == B->getFirstNonPHI())
        break;
      auto &VE = InstToVExpr[&I];
      auto &PE = VExprToPExpr[VE];
      VE->setVersion(PExprToCounter[PE]++);
    }

    // Then Factors
    for (auto FE : BlockToFactors[B]) {
      // We want to process LFactors specifically after the normal ones so the
      // expressions will assume their versions
      if (FE->getIsLinked()) continue;
      auto PE = FE->getPExpr();
      FE->setVersion(PExprToCounter[PE]++);
      PExprToVExprStack[PE].push({FSDFS, FE});
    }

    // Then LFactors
    for (auto FE : BlockToFactors[B]) {
      if (!FE->getIsLinked()) continue;
      auto PE = FE->getPExpr();
      FE->setVersion(PExprToCounter[PE]++);
      PExprToVExprStack[PE].push({FSDFS, FE});
    }

    // And the rest of the instructions
    for (auto &I : *B) {
      // Skip already passed PHIs
      if (&I != B->getFirstNonPHI())
        continue;

      auto &VE = InstToVExpr[&I];
      auto &PE = VExprToPExpr[VE];
      auto SDFS = InstrSDFS[&I];

      // Backtrace every PExprs' stack if we jumped up the tree
      for (auto &P : PExprToVExprStack) {
        auto &VEStack = P.getSecond();
        while (!VEStack.empty() && VEStack.top().first > SDFS) {
          VEStack.pop();
        }
      }

      // Do nothing for ignored expressions
      if (IgnoreExpression(*VE))
        continue;

      // TODO Any operand definition handling goes here

      // TODO
      // This is a simplified version for operand comparison, normally we
      // would check current operands on their respected stacks with operands
      // for the VExpr on its stack, if they match we assign the same version,
      // otherwise there was a def for VExpr operand and we need to assign a new
      // version. This will be required when operand versioning is implemented.
      //
      // For now this will suffice, the only case we reuse a version if we've
      // seen this expression before, since in SSA there is a singe def for
      // an operand.
      //
      // This limits algorithm effectiveness, because we do not track operands'
      // versions we cannot prove that certain separate expressions are in fact
      // the same expressions of different versions. TBD, anyway.
      //
      // Another thing related to not tracking operand versions, because of that
      // there always will be a single definition of VExpr's operand and the
      // VExpr itself will follow it in the traversal, thus, for now, we do not
      // have to assign ⊥ version to the VExpr whenever we see its operand
      // defined.
      auto &VEStack = PExprToVExprStack[PE];
      auto *VEStackTop = VEStack.empty() ? nullptr : VEStack.top().second;
      auto *VEStackTopF = VEStackTop
                            ? dyn_cast<FactorExpression>(VEStackTop)
                            : nullptr;
      // Empty stack
      if (!VEStackTop) {
        VE->setVersion(PExprToCounter[PE]++);
        VEStack.push({SDFS, VE});
      // Factor
      } else if (VEStackTopF) {
        // If every operands' definition dominates this Factor we are dealing
        // with the same expression and assign Factor's version
        if (OperandsDominate(VE, VEStackTopF)) {
          VE->setVersion(VEStackTop->getVersion());
          Substitutions[VE] = VEStackTop;
        // Otherwise VE's operand(s) is(were) defined in this block and this
        // is indeed a new expression version
        } else {
          VE->setVersion(PExprToCounter[PE]++);
          VEStack.push({SDFS, VE});

          // STEP 3 Init: DownSafe
          // If the top of the stack contains a Factor expression we clear its
          // DownSafe flag because its result is not used and not anticipated
          VEStackTopF->setDownSafe(false);
        }
      // Real occurrence
      } else {
        // We need to campare all operands versions, if they don't match we are
        // dealing with a new expression
        bool SameVersions = true;
        auto VEBE = dyn_cast<BasicExpression>(VE);
        auto VEStackTopBE = dyn_cast<BasicExpression>(VEStackTop);
        for (unsigned i = 0, l = VEBE->getNumOperands(); i < l; ++l) {
          if (ValueToExp[VEBE->getOperand(i)]->getVersion() !=
              ValueToExp[VEStackTopBE->getOperand(i)]->getVersion()) {
            SameVersions = false;
            break;
          }
        }

        if (SameVersions) {
          VE->setVersion(VEStackTop->getVersion());
          Substitutions[VE] = VEStackTop;
        } else {
          VE->setVersion(PExprToCounter[PE]++);
          VEStack.push({SDFS, VE});
        }
      }

      PExprToVersions[PE][VE->getVersion()].insert(VE);

      // if (VEStackTop) {
      //   // TODO Here we campare VEStackTop used operands' versions against
      //   // current operands' versions on their respected stacks, if they match
      //   // we use its version, otherwise assign a new one. If it happens that
      //   // a Factor is on top of the stack we check whether its operands'
      //   // definitions dominate this Factor, otherwise assign a new version
      //   //
      //   // For now every Expression after its Factor assumes its version since
      //   // its operands are defined before this Factor.
      //
      //   // If the top of stack contains take its version
      //   VE->setVersion(VEStackTop->getVersion());
      //   // Set top of the stack as a defining occurrance for the current
      //   // Expression
      //   Substitutions[VE] = VEStackTop;
      // } else {
      //   // Otherwise assign new version
      //   VE->setVersion(PExprToCounter[PE]++);
      //
      //   // STEP 3 Init: DownSafe
      //   // If the top of the stack contains a Factor expression we clear its
      //   // DownSafe flag because its result is not used and not anticipated
      //   // if (VEStackTop && FactorExpression::classof(VEStackTop)) {
      //   //   auto *F = dyn_cast<FactorExpression>(VEStackTop);
      //   //   F->setDownSafe(false);
      //   // }
      //
      //   VEStack.push({SDFS, VE});
      // }
    }

    // For a terminator we need to visit every cfg successor of this block
    // to update its Factor expressions
    auto *T = B->getTerminator();
    for (auto S : T->successors()) {
      for (auto F : BlockToFactors[S]) {
        // Linked Factor's operands are already versioned and set
        if (F->getIsLinked()) continue;
        auto PE = F->getPExpr();
        auto &VEStack = PExprToVExprStack[PE];
        size_t PI = F->getPredIndex(B);
        assert(PI != -1UL && "Should not be the case");
        auto VEStackTop = VEStack.empty() ? nullptr : VEStack.top().second;
        auto VE = VEStack.empty() ? GetBottom() : VEStackTop;

        F->setVExpr(PI, VE);

        // STEP 3 Init: HasRealUse
        bool HasRealUse = false;
        if (VEStackTop) {
          // To check Factor's usage we need to check usage of the Expressions
          // of the same version
          if (FactorExpression::classof(VEStackTop)) {
            HasRealUse = FactorHasRealUse((FactorExpression *)VEStackTop);
          // If it is a real expression we check the usage directly
          } else if (BasicExpression::classof(VEStackTop)) {
            auto Inst = VExprToInst[VEStackTop];
            int NumUses = Inst->getNumUses();
            if (NumUses == 0) {
              HasRealUse = false;
            // If the Value is used by a PHI it is not counted as a use
            } else {
              for (auto U : Inst->users()) {
                if (!PHINode::classof(U)) {
                  HasRealUse = true;
                  break;
                }
              }
            }
          }
        }

        F->setHasRealUse(PI, HasRealUse);
      }
    }

    // STEP 3 Init: DownSafe
    // We set Factor's DownSafe to False if it is the last Expression's
    // occurence before program exit.
    if (T->getNumSuccessors() == 0) {
      for (auto &P : PExprToVExprStack) {
        auto &VEStack = P.getSecond();
        if (VEStack.empty()) continue;
        if (auto *F = dyn_cast<FactorExpression>(VEStack.top().second)) {
          if (!FactorHasRealUse(F))
            F->setDownSafe(false);
        }
      }
    }
  }

  // TODO clean this up
  // TODO Redundant Factors
  // TODO 1. we need to spot redundant Factors that join differently versioned
  // TODO expressions which have the same operands
  // TODO 2. Newly versioned factors that are the same as linked factors

  SmallPtrSet<FactorExpression *, 32> FactorKillList;
  for (auto P : FactorToPHI) {
    auto LF = (FactorExpression *)P.getFirst();
    for (auto F : FExprs) {
      if (F->getIsLinked() || LF == F) continue;
      bool Same = true;
      for (unsigned i = 0, l = LF->getVExprNum(); i < l; ++i) {
        auto LFVE = LF->getVExpr(i);
        auto FVE = F->getVExpr(i);
        // NOTE
        // Kinda a special case, while assigning versioned expressions to a Factor
        // we cannot infer that a variable or a constant is coming from the
        // predecessor and we assign it to ⊥, but a Linked Factor will know
        // for sure whether a constant/variable is involved.
        if (VariableOrConstant(*LFVE) && FVE == GetBottom()) continue;

        // NOTE
        // Yet another special case, since we do not add same version on the
        // stack it is possible to have a Factor as an operand of itself, this
        // happens for cycles only. We treat such an operand as a bottom and
        // ignore it.
        if (FVE == LF || FVE == F) continue;

        if (LFVE != FVE) {
          Same = false;
          break;
        }
      }
      if (Same) {
        FactorKillList.insert(F);
      }
    }
  }

  // Remove all stuff related
  for (auto F : FactorKillList) {
    if (auto P = F->getProto()) {
      P->dropAllReferences();
    }
    FExprs.erase(F);
    auto B = FactorToBlock[F];
    BlockToFactors[B].erase(F);
    FactorToBlock.erase(F);
  }

  // Determine cyclic Factors of whats left
  for (auto F : FExprs) {
    for (auto VE : F->getVExprs()) {
      // This happens if the Factor is contained inside a cycle and the is
      // not change in the expression's operands along this cycle
      if (F->getVersion() == VE->getVersion())
        F->setIsCycle(true);
    }
  }
}

void SSAPRE::
ResetDownSafety(FactorExpression &FE, unsigned ON) {
  auto E = FE.getVExpr(ON);
  if (FE.getHasRealUse(ON) || !FactorExpression::classof(E)) {
    return;
  }

  auto *F = dyn_cast<FactorExpression>(E);
  if (!F->getDownSafe())
    return;

  F->setDownSafe(false);
  for (size_t i = 0, l = F->getVExprNum(); i < l; ++i) {
    ResetDownSafety(*F, i);
  }
}

void SSAPRE::
DownSafety() {
  // Here we propagate DownSafety flag initialized during Step 2 up the Factor
  // graph for each expression
  for (auto F : FExprs) {
    if (F->getDownSafe())
      continue;
    for (size_t i = 0, l = F->getVExprNum(); i < l; ++i) {
      ResetDownSafety(*F, i);
    }
  }
}

void SSAPRE::
ComputeCanBeAvail() {
  for (auto F : FExprs) {
    for (auto V : F->getVExprs()) {
      if (IsBottom(*V)) {
        if (!F->getDownSafe() && F->getCanBeAvail()) {
          ResetCanBeAvail(*F);
        }
        break;
      }
    }
  }
}

void SSAPRE::
ResetCanBeAvail(FactorExpression &G) {
  G.setCanBeAvail(false);
  for (auto F : FExprs) {
    auto I = F->getVExprIndex(G);
    if (I == -1UL) continue;
    if (!F->getHasRealUse(I)) {
      F->setVExpr(I, GetBottom());
      if (!F->getDownSafe() && F->getCanBeAvail()) {
        ResetCanBeAvail(*F);
      }
    }
  }
}

void SSAPRE::
ComputeLater() {
  for (auto F : FExprs) {
    F->setLater(F->getCanBeAvail());
  }
  for (auto F : FExprs) {
    if (F->getLater()) {
      for (size_t i = 0, l = F->getVExprNum(); i < l; ++i) {
        if (F->getHasRealUse(i) && !IsBottom(*F->getVExpr(i))) {
          ResetLater(*F);
          break;
        }
      }
    }
  }
}

void SSAPRE::
ResetLater(FactorExpression &G) {
  G.setLater(false);
  for (auto F : FExprs) {
    if (F->hasVExpr(G) && F->getLater())
      ResetLater(*F);
  }
}

void SSAPRE::
WillBeAvail() {
  ComputeCanBeAvail();
  ComputeLater();
}

void SSAPRE::
FinalizeVisit(BasicBlock &B) {
  for (auto &P : PExprToInsts) {
    AvailDef.insert({P.getFirst(), DenseMap<int,Expression *>()});
  }

  for (auto F : BlockToFactors[&B]) {
    F->clrSave();
    F->setReload(false);
    auto V = F->getVersion();
    // Potentially available or already available Factors we add to the table
    if (F->getWillBeAvail() || F->getIsAvail()) {
      auto PE = F->getPExpr();
      AvailDef[PE][V] = F;
    }
  }

  for (auto &I : B) {
    auto &VE = InstToVExpr[&I];
    auto &PE = VExprToPExpr[VE];

    VE->clrSave();
    VE->setReload(false);

    // Linked PHI nodes are ignored, their Factors are processed instead
    if (PHINode::classof(&I) && PHIToFactor[(PHINode *) &I])
      continue;

    // Traverse operands and add Save count to theirs definitions
    for (auto &O : I.operands()) {
      if (auto &E = ValueToExp[O]) {
        E->addSave();
      }
    }

    // We ignore these definitions
    if (IgnoreExpression(*VE)) {
      VE->setSave(INT32_MAX / 2); // so it won't overflow upon addSave
      continue;
    }

    auto V = VE->getVersion();

    auto &ADPE = AvailDef[PE];
    // If there was no expression occurrence before
    // or it was an expression's operand definition
    // or the previous expression does not strictly dominate the current occurrence
    if (!ADPE.count(V) || IsBottom(*ADPE[V]) ||
        !NotStrictlyDominates(ADPE[V], VE)) {
      ADPE[V] = VE;
    // Or it was a Factor that for sure will be materialized, if not alredy
    // NOTE This wasn't in the paper, only real occurances force reload
    } else if (auto F = dyn_cast<FactorExpression>(ADPE[V])) {
      if (F->getWillBeAvail() || F->getIsAvail()) {
        VE->setReload(true);
      }
    // Or it was a Real occurrence
    } else if (BasicExpression::classof(ADPE[V])) {
      ADPE[V]->addSave();
      VE->setReload(true);
    }
  }

  for (auto S : B.getTerminator()->successors()) {
    for (auto F : BlockToFactors[S]) {
      // ??? Not sure whether Avail ought go through here
      if (F->getWillBeAvail() || F->getIsAvail()) {
        auto PE = F->getPExpr();
        auto PI = F->getPredIndex(&B);
        auto O = F->getVExpr(PI);
        // Satisfies insert if either:
        //   - Version(O) is ⊥
        //   - HRU(O) is False and O is Factor and WBA(O) is False
        if (IsBottom(*O)||
            (!F->getHasRealUse(PI) &&
             FactorExpression::classof(O) &&
             !dyn_cast<FactorExpression>(O)->getWillBeAvail())) {
          // NOTE
          // At this point we just create insertion lists, update F graph,
          // the actual insertion and rewiring will be done at Code Motion step.
          auto I = PE->getProto()->clone();
          auto VE = CreateExpression(*I);
          VE->setSave(1);
          PExprToInsts[PE].insert(I);
          VExprToInst[VE] = I;
          VExprToPExpr[VE] = PE;
          InstToVExpr[I] = VE;
          InstrSDFS[I] = InstrSDFS[B.getTerminator()];
          InstrDFS[I] = InstrDFS[B.getTerminator()];
          F->setVExpr(PI, VE);
          F->setHasRealUse(PI, true);
          if (!BlockToInserts.count(&B)) {
            BlockToInserts.insert({&B, {I}});
          } else {
            BlockToInserts[&B].insert(I);
          }
        } else {
          auto V = O->getVersion();
          auto &ADPE = AvailDef[PE];
          if (BasicExpression::classof(ADPE[V])) {
            ADPE[V]->addSave();
          }
        }
      }
    }
  }
}

void SSAPRE::
Finalize() {
  // Finalize step performs the following tasks:
  //   - Decides for each Real expression whether it should be computed on the
  //     spot or reloaded from the temporary. For each one that is computed, it
  //     also decides whether the result should be saved to the temp. There are
  //     two flags that control all that: Save and Reload.
  //   - For Factors where will_be_avail is true, insertions are performed at
  //     the incoming edges that correspond to Factor operands at which the
  //     expression is not available
  //   - Expression Factors whose will_be_avail predicate is true may become PHI
  //     for the temp. Factors that are not will_be_avail will not be part of the
  //     SSA form of the temp, and links from will_be_avail Factors that
  //     reference them are fixed up to other(real or inserted) expressions.
  for (auto B : *RPOT) {
    FinalizeVisit(*B);
  }
}

bool SSAPRE::
CodeMotion() {
  bool Changed = false;

  DenseMap<const Expression *, std::stack<std::pair<unsigned, Expression *>>>
    PExprToVExprStack;

  for (auto &P : PExprToInsts) {
    auto &PE = P.getFirst();
    if (IgnoreExpression(*PE))
      continue;
    PExprToVExprStack.insert({PE, {}});
  }

  for (auto B : *RPOT) {
    // Since factors live outside basic blocks we set theirs DFS as the first
    // instruction's in the block
    auto FSDFS = InstrSDFS[&B->front()];

    for (auto FE : BlockToFactors[B]) {
      auto PE = FE->getPExpr();
      if (FE->getIsLinked()) {
        auto PHI = (PHINode *)FactorToPHI[FE];
        if (FE->getWillBeAvail()) continue;
        // If Factor is Linked and Available we need to replace linked PHI
        // instruction with a real calculation since we cannot wait longer delay
        // the computation, unless it is a cycle
        if (FE->getIsAvail() && FE->getLater()) {
          // auto I = PE->getProto()->clone();
          //
          // auto VE = CreateExpression(*I);
          // VE->setSave(PHI->getNumUses());
          // PExprToInsts[PE].insert(I);
          // VExprToInst[VE] = I;
          // VExprToPExpr[VE] = PE;
          // InstToVExpr[I] = VE;
          // InstrSDFS[I] = InstrSDFS[&B->front()];
          // InstrDFS[I] = InstrDFS[&B->front()];

          // FIXME this only works if there are TWO incomming values
          assert(FE->getVExprNum() == 2 && "Well, shit...");
          // Push computation to the init block
          Expression * VE;
          // Find non-cycled expression
          for (auto V : FE->getVExprs()) {
            if (V->getVersion() != FE->getVersion()){
              VE = V;
              break;
            }
          }

          // Now any PHI reference will go to VE
          Substitutions[FE] = VE;

          VE->addSave(PHI->getNumUses());
          PHI->replaceAllUsesWith(VExprToInst[VE]);

          Changed = true;
        }
        assert(PHI->getNumUses() == 0 && "Must not have any uses");
        KillList.push_back((Instruction *)PHI);
      } else {
        // Nothing here?
      }

      // Push every Factor to the Stack regardless of the linkage, since further
      // instructions may refer them
      PExprToVExprStack[PE].push({FSDFS, FE});
    }

    // Insert Instructions
    for (auto I : BlockToInserts[B]) {
      IRBuilder<> Builder((Instruction *)B->getTerminator());
      Builder.Insert(I);
      Changed = true;
    }

    for (auto &I : *B) {
      auto &VE = InstToVExpr[&I];
      auto &PE = VExprToPExpr[VE];

      // Only looking at the real expression occurrances
      if (VE->getExpressionType() != ET_Basic)
        continue;

      auto SDFS = InstrSDFS[&I];
      auto &VEStack = PExprToVExprStack[PE];

      // Backtrace every PExprs' stack if we jumped up the tree
      for (auto &P : PExprToVExprStack) {
        auto &VEStack = P.getSecond();
        while (!VEStack.empty() && VEStack.top().first > SDFS) {
          VEStack.pop();
        }
      }

      if (VE->getReload()) {
        auto *VEStackTop = VEStack.empty() ? nullptr : VEStack.top().second;
        assert(VEStackTop && "This must not be null");

        // Replace usage
        Substitutions[VE] = VEStackTop;
        ReloadList.insert(&I);
        // auto &RI = VExprToInst[VEStackTop];
        // I.replaceAllUsesWith(RI);

        // Update Factors
        for (auto F : FExprs) {
          auto VEI = F->getVExprIndex(*VE);
          if (VEI != -1UL)
            F->setVExpr(VEI, VEStackTop);
        }
      } else if (VE->getSave()) {
        // Leave it be
        VEStack.push({SDFS, VE});
      } else {
        for (auto U : I.users()) {
          assert(PHINode::classof(U) &&
              "This instructin must not have any uses except PHIs");
        }
        KillList.push_back(&I);
      }
    }

    // For each terminator we need to visit every cfg successor of this block
    // to update its Factor expressions
    for (auto S : B->getTerminator()->successors()) {
      for (auto F : BlockToFactors[S]) {
        auto &VEStack = PExprToVExprStack[F->getPExpr()];
        size_t PI = F->getPredIndex(B);
        assert(PI != -1UL && "Should not be the case");
        F->setVExpr(PI, VEStack.empty() ? GetBottom() : VEStack.top().second);
      }
    }
  }

  // Insert PHIs for each available
  for (auto &P : BlockToFactors) {
    auto &B = P.getFirst();

    // Check parameters of potential PHIs, they are either:
    //  - Factor
    //  - Saved Expression
    for (auto F : P.getSecond()) {
      // No PHI insertion for NotAwailable or Linked Factors
      if (!F->getWillBeAvail() || F->getIsLinked()) continue;
      bool hasFactors = false;
      bool hasSaved = false;
      bool hasDeleted = false;
      for (auto &O : F->getVExprs()) {
        if (FactorExpression::classof(O)) {
          hasFactors = true;
        } else {
          hasSaved   |= O->getSave();
          hasDeleted |= !O->getSave();
        }
      }

      // Insert a PHI only if its operands are live
      if (hasFactors || hasSaved) {
        assert(!hasDeleted && "Must not be the case");
        IRBuilder<> Builder((Instruction *)B->getFirstNonPHI());
        auto BE = dyn_cast<BasicExpression>(F->getPExpr());
        auto PHI = Builder.CreatePHI(BE->getType(), F->getVExprNum());
        for (auto &VE : F->getVExprs()) {
          auto I = VExprToInst[VE];
          PHI->addIncoming(I, I->getParent());
        }
        // Make Factor Expression point to a real PHI
        VExprToInst[F] = PHI;
        Changed = true;
      }
    }
  }

  // Reload marked instructions
  for (auto I : ReloadList) {
    if (I->hasNUsesOrMore(1)) {
      auto *Def = Substitutions[InstToVExpr[I]];
      while (Def != Substitutions[Def]) {
        Def = Substitutions[Def];
      }

      assert(I != VExprToInst[Def] && "Something went wrong");

      // Increase Def's Save count by the number of uses of the instruction
      Def->addSave(I->getNumUses());

      I->replaceAllUsesWith(VExprToInst[Def]);
    }

    InstToVExpr[I]->setSave(0);
    KillList.push_back(I);
    Changed = true;
  }

  // Kill'em all
  // Before return we want to calculate effects of instruction deletion on the
  // other instructions. For example if we delete the last user of a value and
  // the instruction that produces this value does not have any side effects we
  // can delete it, and so on.
  for (unsigned i = 0, l = KillList.size(); i < l; ++i) {
    auto I = KillList[i];

    for (auto U : I->users()) {
      assert(PHINode::classof(U) &&
          "This instructin must not have any uses except PHIs");
    }

    // Decrease usage count of the instruction's operands
    for (auto &O : I->operands()) {
      if (auto &OE = ValueToExp[O]) {
        if (IgnoreExpression(*OE)) continue;
        bool AlreadyInKill = !OE->getSave();
        OE->remSave();
        if (!AlreadyInKill && !OE->getSave()) {
          KillList.push_back(VExprToInst[OE]);
          l++;
        }
      }
    }

    // Just drop the references for now
    I->dropAllReferences();
  }

  // Clear Protos
  for (auto &P : PExprToInsts) {
    auto *Proto = P.getFirst()->getProto();
    if (Proto)
      Proto->dropAllReferences();
  }


  // Remove instructions completely
  while (!KillList.empty()) {
    KillList.pop_back_val()->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

PreservedAnalyses SSAPRE::
runImpl(Function &F,
        AssumptionCache &_AC,
        TargetLibraryInfo &_TLI, DominatorTree &_DT) {
  DEBUG(dbgs() << "SSAPRE(" << this << ") running on " << F.getName());

  bool Changed = false;

  TLI = &_TLI;
  DL = &F.getParent()->getDataLayout();
  AC = &_AC;
  DT = &_DT;

  NumFuncArgs = F.arg_size();

  RPOT = new ReversePostOrderTraversal<Function *>(&F);

  Init(F);

  FactorInsertion();
  DEBUG(PrintDebug("STEP 1: F-Insertion"));

  Rename();
  DEBUG(PrintDebug("STEP 2: Renaming"));

  DownSafety();
  DEBUG(PrintDebug("STEP 3: DownSafety"));

  WillBeAvail();
  DEBUG(PrintDebug("STEP 4: WillBeAvail"));

  // TEST remove after
  // Fini();
  // return PreservedAnalyses::all();

  Finalize();
  DEBUG(PrintDebug("STEP 5: Finalize"));

  Changed = CodeMotion();
  DEBUG(PrintDebug("STEP 6: CodeMotion"));

  Fini();

  if (!Changed)
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

PreservedAnalyses SSAPRE::run(Function &F, AnalysisManager<Function> &AM) {
  return runImpl(F,
      AM.getResult<AssumptionAnalysis>(F),
      AM.getResult<TargetLibraryAnalysis>(F),
      AM.getResult<DominatorTreeAnalysis>(F));
}


//===----------------------------------------------------------------------===//
// Pass Legacy
//
// Do I need to keep it?
//===----------------------------------------------------------------------===//

class llvm::ssapre::SSAPRELegacy : public FunctionPass {
  DominatorTree *DT;

public:
  static char ID; // Pass identification, replacement for typeid.
  SSAPRE Impl;
  SSAPRELegacy() : FunctionPass(ID) {
    initializeSSAPRELegacyPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "SSAPRE"; }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto PA = Impl.runImpl(F, AC, TLI, DT);
    return !PA.areAllPreserved();
  }

private:
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
  }
};

char SSAPRELegacy::ID = 0;

// createSSAPREPass - The public interface to this file.
FunctionPass *llvm::createSSAPREPass() { return new SSAPRELegacy(); }

INITIALIZE_PASS_BEGIN(SSAPRELegacy,
                      "ssapre",
                      "SSA Partial Redundancy Elimination",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(BreakCriticalEdges)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(SSAPRELegacy,
                    "ssapre",
                    "SSA Partial Redundancy Elimination",
                    false, false)
