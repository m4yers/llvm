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
Expression BExpr(ET_Buttom, ~2U, -10000);

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

// TODO remove Stacks if not used
bool SSAPRE::
OperandsDominate(const PExprToVExprStack_t &S, const Expression *Exp,
                 const FactorExpression *Factor) {

  for (auto &O : VExprToInst[Exp]->operands()) {
    auto E = ValueToExp[O];

    // Variables or Constants occurs indefinitely before any expression
    if (VariableOrConstant(*E)) continue;

    // We want to use the earliest occurrence of the operand, it will be either
    // a Factor, another definition or the same definition if it defines a new
    // version.
    while (E != Substitutions[E])
      E = Substitutions[E];

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
  // We check every Expression of the same version as the Factor we check, since
  // by definition those will come after the Factor
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

bool SSAPRE::
FactorHasRealUseBefore(const FactorExpression *F, const BBVector_t &P,
                       const Expression *E) {
  auto EDFS = InstrDFS[VExprToInst[E]];

  // If Factor is linked with a PHI we need to check its users.
  if (auto PHI = FactorToPHI[F]) {
    for (auto U : PHI->users()) {
      // Ignore PHIs that are linked with Factors, since those bonds solved
      // through the main algorithm
      if (PHINode::classof(U) && PHIToFactor[(PHINode*)U]) continue;
      auto UB = ((Instruction *)U)->getParent();
      for (auto PB : P) {
        // Block is on the Path and the User's DFS less or equal to Expression
        if (UB == PB && InstrDFS[(Instruction *)U] <= EDFS)
          return true;
      }
    }
  }

  // We check every Expression of the same version as the Factor we check,
  // since by definition those will come after the Factor
  auto &Versions = PExprToVersions[F->getPExpr()];
  for (auto V : Versions[F->getVersion()]) {
    for (auto U : VExprToInst[V]->users()) {
      if (PHINode::classof(U) && PHIToFactor[(PHINode*)U]) continue;
      auto UB = ((Instruction *)U)->getParent();
      for (auto PB : P) {
        // Block is on the Path and the User's DFS less or equal to Expression
        if (UB == PB && InstrDFS[(Instruction *)U] <= EDFS)
          return true;
      }
    }
  }

  return false;
}

bool SSAPRE::
HasRealUseBefore(const Expression *S, const BBVector_t &P,
                 const Expression *E) {
  auto EDFS = InstrDFS[VExprToInst[E]];
  auto &Versions = PExprToVersions[VExprToPExpr[S]];
  for (auto V : Versions[S->getVersion()]) {
    for (auto U : VExprToInst[V]->users()) {
      if (PHINode::classof(U) && PHIToFactor[(PHINode*)U]) continue;
      auto UB = ((Instruction *)U)->getParent();
      for (auto PB : P) {
        // Block is on the Path and the User's DFS less or equal to Expression
        if (UB == PB && InstrDFS[(Instruction *)U] <= EDFS)
          return true;
      }
    }
  }
  return false;
}

void SSAPRE::
KillFactor(FactorExpression * F, bool UpdateOperands) {
  FExprs.erase(F);
  auto &B = FactorToBlock[F];
  auto &V = BlockToFactors[B];
  for (auto VS = V.begin(), VV = V.end(); VS != VV; ++VS) {
    if (*VS == F)
      V.erase(VS);
  }

  FactorToBlock.erase(F);

  if (!UpdateOperands) return;
  return;

  // Every time we delete a Factor we propagate DS False to all its Factor
  // operands that has no real use for the rest of Factors that use it
  for (auto O : F->getVExprs()) {
    O = GetSubstitution(O);
    if (!O) continue;
    if (auto FO = dyn_cast<FactorExpression>(O)) {
      bool HRU = false;
      // FIXME remove cycle, this can be resolved by traking the uses/users
      for (auto FE : FExprs) {
        if (FE->hasVExpr(*FO) && FE->getHasRealUse(*FO)) {
          HRU = true;
          break;
        }
      }

      if (HRU) break;

      // This factor is not anticipated anywhere
      FO->setDownSafe(false);

      // Recalculate CBA
      if (FO->getCanBeAvail()) {
        for (auto FOO : FO->getVExprs()) {
          if (auto FFOO = dyn_cast<FactorExpression>(FOO)) {
            // ??? Should it be recursive?
            if (IsBottom(*FFOO)) {
              FFOO->setCanBeAvail(false);
              FFOO->setLater(false);
              // ResetCanBeAvail(*FFOO);
              break;
            }
          }
        }
      }
    }
  }
}

void SSAPRE::
SetOrderBefore(Instruction *I, Instruction *B) {
  InstrSDFS[I] = InstrSDFS[B]; InstrSDFS[B]++;
  InstrDFS[I]  = InstrDFS[B];  InstrDFS[B]++;
}

void SSAPRE::
AddVExpr(Expression *PE, Expression *VE, Instruction *I, BasicBlock *B,
         bool ToBeInserted) {
  assert(PE && VE && I && B);

  AddSubstitution(VE, VE);

  ExpToValue[VE] = I;
  ValueToExp[I] = VE;

  InstToVExpr[I] = VE;
  VExprToInst[VE] = I;
  VExprToPExpr[VE] = PE;

  if (!PExprToVExprs.count(PE)) {
    PExprToVExprs.insert({PE, {VE}});
  } else {
    PExprToVExprs[PE].insert(VE);
  }

  if (!PExprToInsts.count(PE)) {
    PExprToInsts.insert({PE, {I}});
  } else {
    PExprToInsts[PE].insert(I);
  }

  if (ToBeInserted) {
    if (!BlockToInserts.count(B)) {
      BlockToInserts.insert({B, {I}});
    } else {
      BlockToInserts[B].push_back(I);
    }
  } else {
    if (!PExprToBlocks.count(PE)) {
      PExprToBlocks.insert({PE, {B}});
    } else {
      PExprToBlocks[PE].insert(B);
    }
  }
}

void SSAPRE::
MaterializeFactor(FactorExpression *FE, PHINode *PHI) {
  assert(FE && PHI);

  auto PHIE = InstToVExpr[PHI];
  auto PPHI = VExprToPExpr[PHIE];

  Substitutions.erase(PHIE);

  // Erase all memory of it
  ExpToValue.erase(PHIE);
  VExprToInst.erase(PHIE);
  VExprToPExpr.erase(PHIE);

  // We need to remove anything related to this PHIs original prototype,
  // because before we verified that this PHI is actually a Factor it was based
  // on its own PHI proto instance.
  PExprToVExprs.erase(PPHI);
  PExprToInsts.erase(PPHI);
  PExprToBlocks.erase(PPHI);
  PExprToVersions.erase(PPHI);

  // Wire FE to PHI
  FactorToPHI[FE] = PHI;
  PHIToFactor[PHI] = FE;

  InstToVExpr[PHI] = FE;
  VExprToInst[FE] = PHI;
  VExprToPExpr[FE] = FE;

  ExpToValue[FE] = PHI;
  ValueToExp[PHI] = FE;

  FE->setIsMaterialized(true);

  // FIXME proper memeroy clean up
  PPHI->getProto()->dropAllReferences();
  delete PPHI;
  delete PHIE;
}

void SSAPRE::
AddSubstitution(Expression * E, Expression * S) {
  assert(E != nullptr && "Really?");
  assert(S != nullptr && "Substitute must not be nullptr, use Top or Bottom instead");
  E->clrSave();
  Substitutions[E] = S;
}

Expression * SSAPRE::
GetSubstitution(Expression * E) {
  while (E != Substitutions[E])
    E = Substitutions[E];
  return E;
}

Value * SSAPRE::
GetValue(Expression *E) {
  if (auto F = dyn_cast<FactorExpression>(E)) {
    if (F->getIsMaterialized())
      return (Value *)FactorToPHI[F];
    else
      llvm_unreachable("nonononon");
  }
  return (Value *)ExpToValue[GetSubstitution(E)];
}

void SSAPRE::
ReplaceMatFactorWExpression(FactorExpression * FE, Expression * VE) {
  VE->setVersion(FE->getVersion());
  // Replace all Factor uses
  for (auto F : FExprs) {
    if (F == FE) continue;
    for (unsigned i = 0, l = F->getVExprNum(); i < l; ++i) {
      if (F->getVExpr(i) == FE) {
        F->setVExpr(i, VE);
        // If we assign the same version we create a cycle
        if (F->getVersion() == VE->getVersion())
          F->setIsCycle(true);
      }
    }
  }

  // Add save for every real use of this PHI
  auto PHI = (PHINode *)FactorToPHI[FE];
  for (auto U : PHI->users()) {
    auto UI = (Instruction *)U;
    auto UE = InstToVExpr[UI];
    // Do not count Factors as usual
    if (FactorExpression::classof(UE)) continue;
    // Skip istructions without parents, unless they are to be inserted
    if (!UI->getParent() && !IsToBeAdded(UI)) continue;
    VE->addSave();
  }

  KillFactor(FE);
  KillList.push_back(PHI);

  // Any Expression of the same type and version follows this Factor occurrence
  // by definition, since we replace the factor with another expression we can
  // remove all other expressions of the same version and replace their usage
  // with this new expression
  auto PE = FE->getPExpr();
  auto &Versions = PExprToVersions[PE][FE->getVersion()];
  for (auto V : Versions) {
    AddSubstitution(V, VE);
    VE->addSave();
  }

  // Replace all PHI uses
  auto V = GetValue(VE);
  PHI->replaceAllUsesWith(V);

  AddSubstitution(FE, VE);
}

void SSAPRE::
ReplaceFactorWExpression(FactorExpression * FE, Expression * VE) {
  // Replace all PHI uses
  VE = GetSubstitution(VE);
  VE->setVersion(FE->getVersion());

  // Replace all Factor uses
  for (auto F : FExprs) {
    if (F == FE) continue;
    for (unsigned i = 0, l = F->getVExprNum(); i < l; ++i) {
      if (F->getVExpr(i) == FE) {
        // If we assign the same version we create a cycle
        if (F->getVersion() == VE->getVersion())
          F->setIsCycle(true);
        F->setVExpr(i, VE);
      }
    }
  }

  KillFactor(FE);

  // Any Expression of the same type and version follows this Factor occurrence
  // by definition, since we replace the factor with another expression we can
  // remove all other expressions of the same version and replace their usage
  // with the this new expression
  auto PE = VExprToPExpr[VE];
  auto &Versions = PExprToVersions[PE][FE->getVersion()];
  for (auto V : Versions) {
    AddSubstitution(V, VE);
    VE->addSave();
  }

  AddSubstitution(FE, VE);
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
CreateFactorExpression(const Expression &PE, const BasicBlock &B) {
  auto FE = new FactorExpression(B);
  size_t C = 0;
  for (auto S = pred_begin(&B), EE = pred_end(&B); S != EE; ++S) {
    FE->addPred(*S, C++);
  }
  FE->setPExpr(&PE);

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

bool SSAPRE::
IsToBeKilled(Expression &E) {
  auto V = ExpToValue[&E];
  for (auto K : KillList) {
    if (V == K)
      return true;
  }

  return false;
}

bool SSAPRE::
IsToBeAdded(Instruction *I) {
  for (auto P : BlockToInserts) {
    for (auto II : BlockToInserts[P.getFirst()]) {
      if (I == II)
        return true;
    }
  }
  return false;
}

bool SSAPRE::
AreAllUsersKilled(const Instruction *I) {
  assert(I);
  for (auto U : I->users()) {
    auto UI = (Instruction *)U;
    if (UI->getParent()) {
      bool Killed = false;
      for (auto K : KillList) {
        if (K == UI) {
          Killed = true;
          break;
        }
      }

      if (!Killed)
        return false;
    }
  }
  return true;
}

void SSAPRE::
PrintDebug(const std::string &Caption) {
  dbgs() << "\n\n---------------------------------------------";
  dbgs() << Caption << "\n";

  dbgs() << "\nProgram";
  dbgs() << "\n(dfs) (instruction)";
  dbgs() << "\n---------------------------";
  for (auto &B : *RPOT) {
    for (auto &I : *B) {
      dbgs() << "\n" << InstrDFS[&I];
      dbgs() << "\t" << I;
    }
  }

  dbgs() << "\n\nExpressions";
  dbgs() << "\n(l/d) (dfs) (expression)";
  dbgs() << "\n---------------------------\n";
  for (auto &P : PExprToInsts) {
    auto &PE = P.getFirst();
    dbgs() << ExpressionTypeToString(PE->getExpressionType());
    for (auto VE : PExprToVExprs[PE]) {
      auto I = VExprToInst[VE];
      dbgs() << "\n\t";
      dbgs() << (I->getParent() ? "(l)" : "(d)");
      dbgs() << " (" << InstrDFS[I]<< ") ";
      VE->printInternal(dbgs());
    }
    dbgs() << "\n";
  }

  dbgs() << "\nBlockToFactors";
  dbgs() << "\n---------------------------\n";
  for (auto &B : *RPOT) {
    auto BTF = BlockToFactors[B];
    if (!BTF.size()) continue;
    dbgs() << "(" << BTF.size() << ") ";
    B->printAsOperand(dbgs(), false);
    dbgs() << ":";
    for (const auto &F : BTF) {
      dbgs() << "\n";
      F->printInternal(dbgs());
    }
    dbgs() << "\n";
  }

  dbgs() << "\nBlockToInserts";
  dbgs() << "\n---------------------------\n";
  for (auto &B : *RPOT) {
    auto BTI = BlockToInserts[B];
    if (!BTI.size()) continue;
    dbgs() << "(" << BTI.size() << ") ";
    B->printAsOperand(dbgs(), false);
    dbgs() << ":";
    for (const auto &I : BTI) {
      auto VE = InstToVExpr[I];
      dbgs() << "\n";
      VE->printInternal(dbgs());
    }
    dbgs() << "\n";
  }

  dbgs() << "\nSubstitutions";
  dbgs() << "\n---------------------------\n";
  for (auto &P : Substitutions) {
    auto VE = P.getFirst();
    auto VI = VExprToInst[VE];
    auto SE = P.getSecond();
    auto SI = VExprToInst[SE];

    if (!VE) continue;
    if (IgnoreExpression(*VE)) continue;
    if (VI && !VI->getParent()) continue;

    if (VI) {
      VI->print(dbgs());
    } else if (auto FE = dyn_cast<FactorExpression>(VE)) {
      dbgs() << "  Factor V: " << FE->getVersion() << ", PE: " << FE->getPExpr();
    } else if (VE == GetBottom()) {
      continue;
    } else {
      llvm_unreachable("Must not be the case");
    }

    dbgs() << " -> ";
    if (VE == SE) {
      dbgs() << "-";
    } else if (auto FE = dyn_cast<FactorExpression>(SE)) {
      if (FE->getIsMaterialized() && FactorToPHI[FE]->getParent()) {
        FactorToPHI[FE]->print(dbgs());
      } else {
      dbgs() << "Factor V: " << FE->getVersion() << ", PE: " << FE->getPExpr();
      }
    } else if (SE == GetBottom()) {
      dbgs() << "⊥";
    } else if (!SI->getParent()) {
      dbgs() << "(deleted)";
    } else {
      SI->print(dbgs());
    }
    dbgs() << "\n";
  }

  dbgs() << "\nKillList";
  dbgs() << "\n---------------------------\n";
  for (auto &K : KillList) {
    if (K->getParent()) {
      K->print(dbgs());
    } else {
      dbgs() << "\n(removed)";
    }
    dbgs() << "\n";
  }

  dbgs() << "\n---------------------------------------------\n";
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

  AddSubstitution(GetBottom(), GetBottom());

  // Each block starts its count from N millions, this will allow us add
  // instructions within wide DFS/SDFS range
  unsigned ICountGrowth = 100000;
  unsigned ICount = ICountGrowth;
  // DFSToInstr.emplace_back(nullptr);

  DenseMap<const DomTreeNode *, unsigned> RPOOrdering;
  unsigned Counter = 0;
  for (auto &B : *RPOT) {
    if (!B->getSinglePredecessor()) {
      JoinBlocks.push_back(B);
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

      AddVExpr(PE, VE, &I, B);

      if (!PExprToVersions.count(PE)) {
        PExprToVersions.insert({PE, DenseMap<unsigned,ExprVector_t>()});
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
    auto BlockRange = AssignDFSNumbers(B, ICount, &InstrDFS, nullptr);
    ICount += BlockRange.second - BlockRange.first + ICountGrowth;
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
  // So this SDFSO which maps our RPOish DFS(DT) onto SDFS order gives us
  // points where we must backtrace our context(stack or whatever we keep
  // updated).  These are the places where the next SDFSO is less than the
  // previous one.
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
  ICount = ICountGrowth;
  DFI = df_begin(DT->getRootNode());
  for (auto DFE = df_end(DT->getRootNode()); DFI != DFE; ++DFI) {
    auto B = DFI->getBlock();
    auto BlockRange = AssignDFSNumbers(B, ICount, &InstrSDFS, nullptr);
    ICount += BlockRange.second - BlockRange.first + ICountGrowth;
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
  KillList.clear();
}

// PHI operands Prototype solver
// NOTE The Solver works on assumption that there is only two incomming values.
// NOTE I'm sure there is a pass that converts n-phi nodes to 2-phi nodes, or
// NOTE it should be created anyway.
namespace llvm {
namespace ssapre {
  typedef const Expression * Token_t;

  // This structure encode an assumption that a SRC PHI is an operand of DST
  // PHI, the second operand of which is TOK.
  struct PropDst_t {
    Token_t TOK;
    const PHINode * DST;

    PropDst_t() = delete;
    PropDst_t(Token_t TOK, const PHINode *DST)
      : TOK(TOK), DST(DST) {}
  };

  Token_t GetTop() { return (Expression *)0x704; }
  Token_t GetBot() { return (Expression *)0x807; }
  bool IsTop(Token_t T) { return T == GetTop(); }
  bool IsBot(Token_t T) { return T == GetBot(); }
  bool IsTopOrBottom(Token_t T) { return IsTop(T) || IsBot(T); }

  // Rules:
  //   T    ^ T    = T      Exp  ^ T    = Exp
  //   Exp  ^ Exp  = Exp    ExpX ^ ExpY = F
  //   Exp  ^ F    = F      F    ^ T    = F
  //   F    ^ F    = F
  Token_t
  CalculateToken(Token_t A, Token_t B) {
    // T    ^ T    = T
    // Exp  ^ Exp  = Exp
    // F    ^ F    = F
    if (A == B) {
      return A;
    }

    // Exp  ^ T    = Exp
    if (IsTop(A) && !IsTopOrBottom(B)) {
      return B;
    } else if (!IsTopOrBottom(A) && IsTop(B)) {
      return A;
    }

    // Exp  ^ F    = F
    if (IsBot(A) && !IsTopOrBottom(B)) {
      return GetBot();
    } else if (!IsTopOrBottom(A) && IsBot(B)) {
      return GetBot();
    }

    // ExpX ^ ExpY = F
    // F    ^ T    = F
    return GetBot();
  }

  typedef DenseMap<const PHINode *, const FactorExpression *> PHIFactorMap_t;
  typedef DenseMap<const PHINode *, Token_t> PHITokenMap_t;
  typedef SmallVector<PropDst_t, 8> PropDstVector_t;
  typedef DenseMap<const PHINode *, PropDstVector_t> SrcPropMap_t;
  typedef DenseMap<const PHINode *, bool> SrcKillMap_t;
  typedef SmallVector<const PHINode *, 8> PHIVector_t;

  class TokenPropagationSolver {
    SSAPRE &O;
    PHIFactorMap_t PHIFactorMap;
    PHITokenMap_t PHITokenMap;
    SrcPropMap_t SrcPropMap;
    SrcKillMap_t SrcKillMap;

  public:
    TokenPropagationSolver() = delete;
    TokenPropagationSolver(SSAPRE &O) : O(O) {}

    void
    CreateFactor(const PHINode *PHI, Token_t PE) {
      auto E = PHIFactorMap[PHI];
      assert(!E && "FE already exist");
      E = O.CreateFactorExpression(*PE, *PHI->getParent());
      PHIFactorMap[PHI] = E;
      SrcKillMap[PHI] = false;
    }

    bool
    HasTokenFor(const PHINode *PHI) {
      return PHITokenMap.count(PHI) != 0;
    }

    Token_t
    GetTokenFor(const PHINode *PHI) {
      assert(HasTokenFor(PHI) && "Well...");
      return PHITokenMap[PHI];
    }

    bool
    HasFactorFor(const PHINode *PHI) {
      return PHIFactorMap.count(PHI) != 0;
    }

    const FactorExpression *
    GetFactorFor(const PHINode *PHI) {
      assert(HasFactorFor(PHI));
      return PHIFactorMap[PHI];
    }

    PHIFactorMap_t
    GetLiveFactors() {
      // Erase all killed Factors before returning the Map
      for (auto P : SrcKillMap) {
        if (P.getSecond()) {
          // FIXME proper FE deletion here
          PHIFactorMap.erase(P.getFirst());
        }
      }
      return PHIFactorMap;
    }

    void
    AddPropagations(Token_t T, const PHINode *S, PHIVector_t DL) {
      for (auto D : DL) { AddPropagation(T, S, D); }
    }

    void
    AddPropagation(Token_t T, const PHINode *S, const PHINode *D) {
      if (!HasFactorFor(S)) CreateFactor(S, T);
      if (!HasFactorFor(D)) CreateFactor(D, T);

      if (!SrcPropMap.count(S)) {
        SrcPropMap.insert({S, {{T, D}}});
      } else {
        SrcPropMap[S].push_back({T, D});
      }
    }

    void
    FinishPropagation(Token_t T, const PHINode *PHI) {
      assert(!SrcKillMap[PHI] && "The Factor is already killed");

      PHITokenMap.insert({PHI, T});

      // Either Top or Bottom results in deletion of the Factor
      if (IsTopOrBottom(T)) {
        SrcKillMap[PHI] = true;
      }

      // Recursively finish every propagation
      for (auto &PD : SrcPropMap[PHI]) {
        auto R = CalculateToken(T, PD.TOK);
        FinishPropagation(R, PD.DST);
      }
    }
  };
} // namespace ssapre
} // namespace llvm

void SSAPRE::
FactorInsertion() {
  TokenPropagationSolver TokSolver(*this);
  // We examine in each join block first to find already "materialized" Factors
  for (auto B : JoinBlocks) {
    for (auto &I : *B) {

      // When we reach first non-phi instruction we stop
      if (&I == B->getFirstNonPHI()) break;

      auto PHI = dyn_cast<PHINode>(&I);
      if (!PHI) continue;

      // Token is a meet of all the PHI's operands. We optimistically set it
      // initially to Top
      Token_t TOK = GetTop();

      // Back Branch source
      const PHINode * BackBranch = nullptr;

      for (unsigned i = 0, l = PHI->getNumOperands(); i < l; ++i) {
        auto Op = PHI->getOperand(i);
        auto OVE = ValueToExp[Op];

        // A variable or a constant regarded as Top value
        if (VariableOrConstant(*OVE)) {
          TOK = CalculateToken(TOK, GetTop());
          continue;
        }

        if (auto OPHI = dyn_cast<PHINode>(Op)) {
          if (InstrDFS[Op] > InstrDFS[PHI]) {
            // FIXME doc, it is not just a cycle, any back branch
            // This is a cycle and the operand is not yet processed by this
            // loop.  We will use a rolling Token that will provide us with a
            // current value that we propagate upwards. Once we reached the top
            // we will verify whether our assumption was correct. If it was,
            // all the PHIs we have visited and are using the same Token will
            // assume this Token as a PE of its operand. If at the the end, or
            // along the way we get a Bottom(F) value we won't be able to
            // connect these PHI with the same PE. If the Bottom was
            // encountered in the middle of traversal we still can get some
            // joined PHIs if we start the process from this Bottom.  The
            // propagation process stops if we encounter that the rest of the
            // operands are either Expression with the same PE or Constant or
            // Variable or Nothing in this case it is a success; or we
            // encounter Expression with different PE, this is a failure case.
            TOK = CalculateToken(TOK, GetTop());
            assert(!BackBranch && "Must not be a second Back Branch");
            BackBranch = OPHI;
            continue;
          }

          // If the User is a PHI and it is linked to a Factor already, this
          // means this PHI/Factor joins expressions of the same type
          if (auto FOVE = PHIToFactor[OPHI]) {
            TOK = CalculateToken(TOK, FOVE->getPExpr());

          // Another back-branched PHI
          } else if (TokSolver.HasFactorFor(OPHI)) {
            // If we already know the Token for this PHI, use it, otherwise it
            // is Bottom
            Token_t T = TokSolver.HasTokenFor(OPHI)
                          ? TokSolver.GetTokenFor(OPHI)
                          : GetTop();
            TOK = CalculateToken(TOK, T);

          // Otherwise it is Bottom
          } else {
            TOK = CalculateToken(TOK, GetBottom());
          }
          continue;
        // Otherwise we use whatever this VE is prototyped by
        } else {
          TOK = CalculateToken(TOK, VExprToPExpr[OVE]);
          continue;
        }
      }

      // This PHI has back branches and we are still not sure whether it is a
      // materialized Factor.
      if (BackBranch) {

        // It is not a materialized Factor for sure
        if (IsBot(TOK)) break;

        // Now we have either an Expression or Top value to propagate
        // upwards. We get/create Factors for current PHI and its cycle PHI
        // operands and link them appropriately.
        TokSolver.AddPropagation(TOK, BackBranch, PHI);

      // Otherwise we have a certain result for this PHI
      } else {
        // If there is a dependency on this PHI, finish it and wait till all
        // the propagations are done
        if (TokSolver.HasFactorFor(PHI)) {
          TokSolver.FinishPropagation(TOK, PHI);

        // Or if the result is an Expression we just create a new Factor
        } else if (!IsTopOrBottom(TOK)) {
          auto F = CreateFactorExpression(*TOK, *B);

          F->setPExpr(TOK);
          MaterializeFactor(F, (PHINode *)PHI);

          // Set already know expression versions
          for (unsigned i = 0, l = PHI->getNumOperands(); i < l; ++i) {
            auto B = PHI->getIncomingBlock(i);
            auto J = F->getPredIndex(B);
            auto UVE = ValueToExp[PHI->getOperand(i)];
            F->setVExpr(J, UVE);
          }

          BlockToFactors[B].push_back({F});
          FactorToBlock[F] = B;
          AddSubstitution(F, F);
          FExprs.insert(F);
        }
      }
    }
  }

  // Process proven-to-be materialized Factor/PHIs
  for (auto &P : TokSolver.GetLiveFactors()) {
    auto PHI = P.getFirst();
    auto B = PHI->getParent();
    auto F = (FactorExpression *)P.getSecond();
    auto T = TokSolver.GetTokenFor(PHI);

    F->setPExpr(T);
    MaterializeFactor(F, (PHINode *)PHI);

    // Set already know expression versions
    for (unsigned i = 0, l = PHI->getNumOperands(); i < l; ++i) {
      auto B = PHI->getIncomingBlock(i);
      auto J = F->getPredIndex(B);
      auto O = PHI->getOperand(i);

      if (auto OPHI = dyn_cast<PHINode>(O)) {

        // If the PHI is a back-branched Factor
        if (TokSolver.HasFactorFor(OPHI)) {
          F->setVExpr(J, (Expression *)TokSolver.GetFactorFor(OPHI));

        // Or maybe this PHI was already processed
        } else if (auto FE = PHIToFactor[OPHI]){
          F->setVExpr(J, (Expression *)FE);

        // If none above we just use PHIExpression
        } else {
          F->setVExpr(J, ValueToExp[O]);
        }

      } else {
        F->setVExpr(J, ValueToExp[O]);
      }
    }

    BlockToFactors[B].push_back({F});
    FactorToBlock[F] = B;
    AddSubstitution(F, F);
    FExprs.insert(F);
  }

  // Factors are inserted in two cases:
  //   - for each block in expressions IDF
  //   - for each phi of expression operand, which indicates expression
  //     alteration
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
      // We only need to insert a Factor at a merge point when it reaches a
      // later occurance(a DEF at this point) of the expression. A later
      // occurance will have a bigger DFS number;
      bool ShouldInsert = false;
      // FIXME improve this, no cycles...

      SmallVector<BasicBlock *, 32> Queue;
      DenseMap<BasicBlock *, bool> Visited;
      Queue.push_back(B);
      while (!Queue.empty()) {
        auto BB = Queue.pop_back_val();

        if (Visited.count(BB)) continue;

        for (auto &I : *BB) {
          auto OE = ValueToExp[&I];
          auto OPE = VExprToPExpr[OE];

          // If Proto of the occurance matches the PE we should insert here
          if (OPE == PE) {
            ShouldInsert = true;
            break;
          }
        }

        if (ShouldInsert) break;

        Visited[BB] = true;

        // Continue with the successors if none found
        for (auto S : BB->getTerminator()->successors()) {
          Queue.push_back(S);
        }
      }

      // If we do not insert just continue
      if (!ShouldInsert) continue;

      // True if a Factor for this Expression with exactly the same arguments
      // exists. There are two possibilities for arguments equality, there
      // either none which means it wasn't versioned yet, or there are
      // versions(or rather expression definitions) which means they were
      // spawned out of PHIs. We are concern with the first case for now.
      bool FactorExists = false;
      // FIXME remove the cycle
      for (auto F : BlockToFactors[B]) {
        if (!F->getIsMaterialized() && F->getPExpr() == PE) {
          FactorExists = true;
          break;
        }
      }

      if (!FactorExists) {
        auto F = CreateFactorExpression(*PE, *B);
        BlockToFactors[B].push_back({F});
        FactorToBlock[F] = B;
        AddSubstitution(F, F);
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

  // The counters are used to number expression versions during DFS walk.
  // Before the renaming phase each instruction(that we do not ignore) is of a
  // proto type(PExpr), after this walk every expression is assign its own
  // version and it becomes a versioned(or instantiated) expression(VExpr).
  DenseMap<const Expression *, int> PExprToCounter;

  // Each PExpr is mapped to a stack of VExpr that grow and shrink during DFS
  // walk.
  PExprToVExprStack_t PExprToVExprStack;

  BBVector_t Path;

  for (auto &P : PExprToInsts) {
    auto &PE = P.getFirst();
    if (IgnoreExpression(*PE)) continue;
    PExprToCounter.insert({PE, 0});
    PExprToVExprStack.insert({PE, {}});
  }

  for (auto B : *RPOT) {
    // Since factors live outside basic blocks we set theirs DFS as the first
    // instruction's in the block
    auto FSDFS = InstrSDFS[&B->front()];

    // Backtrack path if necessary
    while (!Path.empty() && InstrSDFS[&Path.back()->front()] > FSDFS)
      Path.pop_back();

    Path.push_back(B);

    // Set PHI versions first, since factors regarded as occurring at the end
    // of the predecessor blocks
    for (auto &I : *B) {
      if (&I == B->getFirstNonPHI()) break;
      auto &VE = InstToVExpr[&I];
      auto &PE = VExprToPExpr[VE];
      VE->setVersion(PExprToCounter[PE]++);
    }

    // Then Factors
    for (auto FE : BlockToFactors[B]) {
      // We want to process MFactors specifically after the normal ones so the
      // expressions will assume their versions
      if (FE->getIsMaterialized()) continue;
      auto PE = FE->getPExpr();
      FE->setVersion(PExprToCounter[PE]++);
      PExprToVExprStack[PE].push({FSDFS, FE});
    }

    // Then MFactors
    for (auto FE : BlockToFactors[B]) {
      if (!FE->getIsMaterialized()) continue;
      auto PE = FE->getPExpr();
      FE->setVersion(PExprToCounter[PE]++);
      PExprToVExprStack[PE].push({FSDFS, FE});
    }

    // And the rest of the instructions
    for (auto &I : *B) {
      // Skip already passed PHIs
      if (PHINode::classof(&I)) continue;

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
      // This is a simplified version for operand comparison, normally we would
      // check current operands on their respected stacks with operands for the
      // VExpr on its stack, if they match we assign the same version,
      // otherwise there was a def for VExpr operand and we need to assign a
      // new version. This will be required when operand versioning is
      // implemented.
      //
      // For now this will suffice, the only case we reuse a version if we've
      // seen this expression before, since in SSA there is a singe def for an
      // operand.
      //
      // This limits algorithm effectiveness, because we do not track operands'
      // versions we cannot prove that certain separate expressions are in fact
      // the same expressions of different versions. TBD, anyway.
      //
      // Another thing related to not tracking operand versions, because of
      // that there always will be a single definition of VExpr's operand and
      // the VExpr itself will follow it in the traversal, thus, for now, we do
      // not have to assign ⊥ version to the VExpr whenever we see its operand
      // defined.
      auto &VEStack = PExprToVExprStack[PE];
      auto *VEStackTop = VEStack.empty() ? nullptr : VEStack.top().second;
      auto *VEStackTopF = VEStackTop
                            ? dyn_cast<FactorExpression>(VEStackTop)
                            : nullptr;

      // NOTE
      // We do not push on the stack neither already seen versions nor operand
      // definitions, since ... TODO finish this

      // Stack is empty
      if (!VEStackTop) {
        VE->setVersion(PExprToCounter[PE]++);
        VEStack.push({SDFS, VE});
      // Factor
      } else if (VEStackTopF) {
        // If every operands' definition dominates this Factor we are dealing
        // with the same expression and assign Factor's version
        if (OperandsDominate(PExprToVExprStack, VE, VEStackTopF)) {
          VE->setVersion(VEStackTop->getVersion());
          AddSubstitution(VE, VEStackTop);
        // Otherwise VE's operand(s) is(were) defined in this block and this
        // is indeed a new expression version
        } else {
          VE->setVersion(PExprToCounter[PE]++);
          VEStack.push({SDFS, VE});

          // STEP 3 Init: DownSafe
          // If the top of the stack contains a Factor expression and its
          // version is not used along this path we clear its DownSafe flag
          // because its result is not anticipated by any other expression:
          // ---------   ---------
          //       \       /
          //  ------------------
          //   %V = Factor(...)
          //
          //        ...
          //  ( N defs of %V)
          //  ( M uses of %V)
          //        ...
          //
          //   def an opd
          //   new V
          //  ------------------
          //  If M == 0 we clear the %V's DownSafe flag
          if (!FactorHasRealUseBefore(VEStackTopF, Path, VE)) {
            VEStackTopF->setDownSafe(false);
          }
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
          AddSubstitution(VE, VEStackTop);
        } else {
          VE->setVersion(PExprToCounter[PE]++);
          VEStack.push({SDFS, VE});
        }
      }

      PExprToVersions[PE][VE->getVersion()].push_back(VE);
    }

    // For a terminator we need to visit every cfg successor of this block
    // to update its Factor expressions
    auto *T = B->getTerminator();
    for (auto S : T->successors()) {
      for (auto F : BlockToFactors[S]) {
        auto PE = F->getPExpr();
        auto PI = F->getPredIndex(B);
        assert(PI != -1UL && "Should not be the case");

        auto &VEStack = PExprToVExprStack[PE];
        auto VEStackTop = VEStack.empty() ? nullptr : VEStack.top().second;
        auto VE = VEStack.empty() ? GetBottom() : VEStackTop;

        // Linked Factor's operands are already versioned and set
        if (F->getIsMaterialized()) {
          VE = F->getVExpr(PI);
        } else {
          F->setVExpr(PI, VE);
        }

        if (IsBottom(*VE)) continue;

        // STEP 3 Init: HasRealUse
        bool HasRealUse = false;
        if (VEStackTop) {
          // To check Factor's usage we need to check usage of the Expressions
          // of the same version
          if (FactorExpression::classof(VEStackTop)) {
            HasRealUse = FactorHasRealUseBefore(
                           (FactorExpression *)VEStackTop,
                           Path,
                           InstToVExpr[T]);
          // If it is a real expression we check the usage directly
          } else if (BasicExpression::classof(VEStackTop)) {
            HasRealUse = HasRealUseBefore(VEStackTop, Path, InstToVExpr[T]);
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
          if (!FactorHasRealUseBefore(F, Path, InstToVExpr[T]))
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

      if (F->getBB() != LF->getBB()) continue;
      if (F->getPExpr() != LF->getPExpr()) continue;
      if (F->getIsMaterialized() || LF == F) continue;

      bool Same = true;
      for (unsigned i = 0, l = LF->getVExprNum(); i < l; ++i) {
        auto LFVE = LF->getVExpr(i);
        auto FVE = F->getVExpr(i);
        // NOTE
        // Kinda a special case, while assigning versioned expressions to a
        // Factor we cannot infer that a variable or a constant is coming from
        // the predecessor and we assign it to ⊥, but a Linked Factor will know
        // for sure whether a constant/variable is involved.
        if (VariableOrConstant(*LFVE) && FVE == GetBottom()) continue;

        // NOTE
        // Yet another special case, since we do not add same version on the
        // stack it is possible to have a Factor as an operand of itself, this
        // happens for cycles only. We treat such an operand as a bottom and
        // ignore it.
        if (FVE == LF || FVE == F) continue;

        // The actual instances is of no use, since MFactor can cointain real
        // expression the the other Factor may contain the MFactor as operand
        // if those expressions never used
        if (LFVE->getVersion() != FVE->getVersion()) {
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
    KillFactor(F, false);
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
    if (F->getDownSafe()) continue;
    for (size_t i = 0, l = F->getVExprNum(); i < l; ++i) {
      ResetDownSafety(*F, i);
    }
  }
}

void SSAPRE::
ComputeCanBeAvail() {
  for (auto F : FExprs) {
    if (!F->getDownSafe() && F->getCanBeAvail()) {
      for (auto V : F->getVExprs()) {
        if (IsBottom(*V)) {
          ResetCanBeAvail(*F);
          break;
        }
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
    F->clrReload();
    auto V = F->getVersion();
    if (F->getWillBeAvail()) {
      auto PE = F->getPExpr();
      AvailDef[PE][V] = F;
    }
  }

  // NOTE Save property is only applicable to the real instructions, PHI is not
  // NOTE considered a real instruction
  for (auto &I : B) {
    auto &VE = InstToVExpr[&I];
    auto &PE = VExprToPExpr[VE];

    VE->clrSave();
    VE->clrReload();

    // Linked PHI nodes are ignored, their Factors are processed separately
    if (PHINode::classof(&I) && PHIToFactor[(PHINode *) &I])
      continue;

    // Traverse operands and add Save count to theirs definitions
    for (auto &O : I.operands()) {
      if (auto &E = ValueToExp[O]) {
        if (BasicExpression::classof(E))
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
    auto DEF = AvailDef[PE][V];
    // If there was no expression occurrence before
    // or it was an expression's operand definition
    // or the previous expression does not strictly dominate the current occurrence
    if (!DEF || IsBottom(*DEF) ||
        !NotStrictlyDominates(DEF, VE)) {
      ADPE[V] = VE;
    // Or it was a Real occurrence
    } else if (BasicExpression::classof(DEF)) {
      DEF->addSave();
      AddSubstitution(VE, DEF);
      // VE->setReload(true);
    }
  }

  for (auto S : B.getTerminator()->successors()) {
    for (auto F : BlockToFactors[S]) {
      if (F->getWillBeAvail() && !F->getIsCycle() && !F->getIsMaterialized()) {
        auto PE = (Expression *)F->getPExpr();
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
          // VE->setSave(1);
          F->setVExpr(PI, VE);
          // F->setHasRealUse(PI, true);
          AddVExpr(PE, VE, I, &B, /* not yet inserted */ true);
          SetOrderBefore(I, B.getTerminator());
        } else {
          auto V = O->getVersion();
          auto &DEF = AvailDef[PE][V];
          if (BasicExpression::classof(DEF)) {
            // DEF->addSave();
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

  // Insert Instructions
  for (auto P : BlockToInserts) {
    auto B = P.getFirst();
    auto T = (Instruction *)B->getTerminator();
    for (auto I : BlockToInserts[B]) {
      I->insertBefore(T);
      Changed = true;
    }
  }

  PrintDebug("CodeMotion after Insertion");

  for (auto BS = JoinBlocks.rbegin(), BE = JoinBlocks.rend(); BS != BE; ++BS) {
    auto B = (BasicBlock *)*BS;
    for (auto FE : BlockToFactors[B]) {
      auto PE = (Expression *)FE->getPExpr();

      if (FE->getIsCycle()) {
        if (FE->getVExprNum() != 2)
          llvm_unreachable("well, shit..");

        // Cycled Expression
        Expression * CE = nullptr;
        unsigned CI;
        // Non-Cycled incomming Expression
        Expression * VE = nullptr;
        // The source of the incomming Expression
        BasicBlock * PB = nullptr;
        for (unsigned i = 0, l = FE->getVExprNum(); i < l; ++i) {
          auto V = FE->getVExpr(i);
          if (V->getVersion() == FE->getVersion()) {
            CE = V;
            CI = i;
          } else {
            PB = (BasicBlock *)FE->getPred(i);
            VE = V;
          }
        }

        // Cycled side is never used
        if (!FE->getHasRealUse(CI) && !FE->getDownSafe()) {
          ReplaceFactorWExpression(FE, GetBottom());
          if (auto PHI = (PHINode *)FactorToPHI[FE]) {
            assert(AreAllUsersKilled(PHI) && "Should not be like that");
            KillList.push_back(PHI);
          }
          continue;
        }

        // Regardless of whether the Factor is materialized its non-cycled
        // expression may be a constant which we regard as bottom value. In any
        // case if the incomming value is bottom we need to create one.
        if (IsBottom(*VE)) {
          auto I = PE->getProto()->clone();
          VE = CreateExpression(*I);
          AddVExpr(PE, VE, I, PB);
          auto T = PB->getTerminator();
          SetOrderBefore(I, T);
          I->insertBefore(T);
        }

        if (FE->getIsMaterialized()) {
          ReplaceMatFactorWExpression(FE, VE);
        } else {
          ReplaceFactorWExpression(FE, VE);
        }
      } else {
        // If Mat and Later this Factor is useless and we replace it with a real
        // computation
        if (FE->getIsMaterialized() && FE->getLater()) {
          auto I = PE->getProto()->clone();

          auto VE = CreateExpression(*I);
          AddVExpr(PE, VE, I, B);
          auto T = B->getFirstNonPHI();
          SetOrderBefore(I, T);
          I->insertBefore((Instruction *)T);

          ReplaceMatFactorWExpression(FE, VE);

        // This PHI/Factor stays after all so we need to save all its
        // operands
        } else if (FE->getIsMaterialized()) {
            for (auto VE : FE->getVExprs()) {
              if (BasicExpression::classof(VE)) {
                VE->addSave();
              }
            }
        // The others are yet to be materialized
        } else {
        }
      }
    }
  }


  PrintDebug("CodeMotion after JoinBlocks");

  // for (auto B : *RPOT) {
  //
  //   for (auto &I : *B) {
  //     auto &VE = InstToVExpr[&I];
  //     // auto &PE = VExprToPExpr[VE];
  //
  //     // Only looking at the real expression occurrances
  //     if (VE->getExpressionType() != ET_Basic)
  //       continue;
  //
  //     if (!(IsToBeKilled(*VE) || VE->getReload() || VE->getSave())) {
  //       for (auto U : I.users()) {
  //         assert(PHINode::classof(U) &&
  //             "This instructin must not have any uses except PHIs");
  //       }
  //       KillList.push_back(&I);
  //     }
  //   }
  // }
  //
  // PrintDebug("CodeMotion after RPOT pass");

  // Insert PHIs for each available
  for (auto &P : BlockToFactors) {
    auto &B = P.getFirst();

    // Check parameters of potential PHIs, they are either:
    //  - Factor
    //  - Saved Expression
    for (auto F : P.getSecond()) {
      // No PHI insertion for NotAwailable or Linked Factors
      if (!F->getWillBeAvail() || F->getIsMaterialized()) continue;
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
          auto SE = GetSubstitution(VE);
          auto I = VExprToInst[SE];
          PHI->addIncoming(I, I->getParent());

          // Add Save for each operand, since this Factor is live now
          if (BasicExpression::classof(SE)) {
            SE->addSave();
          }
        }
        // Make Factor Expression point to a real PHI
        MaterializeFactor(F, PHI);
        Changed = true;
      }
    }
  }

  PrintDebug("CodeMotion after PHI insertion");

  // Apply Substitutions
  for (auto P : VExprToInst) {
    auto VE = (Expression *)P.getFirst();

    if (!VE) continue;
    if (IgnoreExpression(*VE)) continue;
    if (IsToBeKilled(*VE)) continue;

    Instruction * VI = nullptr;
    if (auto FE = dyn_cast<FactorExpression>(VE)) {
      if (auto PHI = FactorToPHI[FE]) {
        VI = (Instruction *)PHI;
      } else {
        // assert(FE == Substitutions[FE] && "I don't think so");
        // This is just jump record
        continue;
      }
    } else {
      VI = VExprToInst[VE];
    }
    auto SE = GetSubstitution(VE);
    assert(!IsToBeKilled(*SE));

    if (VE == SE) continue;

    // ??? Not sure about this
    // ??? We could use two values Top and Bottom to distinguish between values
    // ??? to remain and to be removed
    if (auto FE = dyn_cast<FactorExpression>(SE)) {
      if (!FE->getIsMaterialized()) continue;
    }

    // The value is not used at all
    if (SE == GetBottom()) {
      assert(AreAllUsersKilled(VI));
      KillList.push_back(VI);
      continue;
    }

    // If the Substitution is Bottom and it is not in the kill list add it there
    if (SE == GetBottom() && !IsToBeKilled(*VE)) {
      // assert(VI->getNumUses() == 0);
      KillList.push_back(VI);
      continue;
    }

    auto SI = (Instruction*)GetValue(SE);
    assert(VI != SI && "Something went wrong");

    VE->clrSave();

    // Check if this instruction is used at all
    bool HRU = false;
    for (auto U : VI->users()) {
      auto UI = (Instruction *)U;
      if (UI->getParent()) {
        HRU = true;
      }
    }

    // If this instruction does not have real use we subtract one Save from its
    // sub
    if (!HRU) {
      SE->remSave();
      if (!SE->getSave() && !IsToBeKilled(*SE)) {
        KillList.push_back(SI);
        continue;
      }
    }

    VI->replaceAllUsesWith(SI);
    Changed = true;

    KillList.push_back(VI);
  }

  PrintDebug("CodeMotion after Substitutions");

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
    auto K = KillList.pop_back_val();
    if (!K->getParent()) continue;
    K->eraseFromParent();
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
  Func = &F;

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

  Finalize();
  DEBUG(PrintDebug("STEP 5: Finalize"));

  Changed = CodeMotion();
  DEBUG(PrintDebug("STEP 6: CodeMotion"));

  Fini();

  DEBUG(F.dump());

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
