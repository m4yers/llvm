//===---- SSAPRELegacy.cpp - SSA PARTIAL REDUNDANCY ELIMINATION--------*- C++ -*-===//
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
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
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

using namespace llvm;
using namespace llvm::ssapre;

#define DEBUG_TYPE "ssapre"

STATISTIC(SSAPREInstrSaved,    "Number of instructions saved");
STATISTIC(SSAPREInstrReloaded, "Number of instructions reloaded");
STATISTIC(SSAPREInstrInserted, "Number of instructions inserted");
STATISTIC(SSAPREInstrDeleted,  "Number of instructions deleted");
STATISTIC(SSAPREBlocksAdded,   "Number of blocks deleted");


//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

PreservedAnalyses SSAPRE::runImpl(Function &F, DominatorTree &DT) {
  bool Changed = false;

  errs() << "Hello: ";
  errs().write_escaped(F.getName()) << '\n';

  if (!Changed)
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

PreservedAnalyses SSAPRE::run(Function &F, AnalysisManager<Function> &AM) {
  return runImpl(F, AM.getResult<DominatorTreeAnalysis>(F));
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

    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto PA = Impl.runImpl(F, DT);
    return !PA.areAllPreserved();
  }

private:
  void getAnalysisUsage(AnalysisUsage &AU) const override {
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
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(SSAPRELegacy,
                    "ssapre",
                    "SSA Partial Redundancy Elimination",
                    false, false)
