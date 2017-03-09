//===-------- SSAPRE.h - SSA PARTIAL REDUNDANCY ELIMINATION -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the interface for LLVM's SSA Partial Redundancy Elimination
// pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_SSAPRE_H
#define LLVM_TRANSFORMS_SCALAR_SSAPRE_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/Dominators.h"

namespace llvm {

/// A private "module" namespace for types and utilities used by SSAPRE. These
/// are implementation details and should not be used by clients.
namespace ssapre LLVM_LIBRARY_VISIBILITY {

class SSAPRELegacy;

} // end namespace ssapre

/// Performs SSA PRE pass.
class SSAPRE : public PassInfoMixin<SSAPRE> {
public:
  PreservedAnalyses run(Function &F, AnalysisManager<Function> &AM);

private:
  friend ssapre::SSAPRELegacy;

  PreservedAnalyses runImpl(Function &F, DominatorTree &DT);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_SSAPRE_H
