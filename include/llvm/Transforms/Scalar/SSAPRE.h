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

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Dominators.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/ArrayRecycler.h"
#include <stack>

namespace llvm {

namespace ssapre LLVM_LIBRARY_VISIBILITY {

class SSAPRELegacy;


//===----------------------------------------------------------------------===//
// Pass Expressions
//===----------------------------------------------------------------------===//

enum ExpressionType {
  ET_Base,
  ET_Buttom,
  ET_Ignored,
  ET_Unknown,
  ET_BasicStart,
  ET_Basic,
  ET_Phi,
  ET_Factor, // Phi for expressions, Φ in paper
  // TODO later
  // ET_Call,
  // ET_AggregateValue,
  // ET_Load,
  // ET_Store,
  ET_BasicEnd

};

inline std::string ExpressionTypeToString(ExpressionType ET) {
  switch (ET) {
  case ET_Ignored: return "ExpressionTypeIgnored";
  case ET_Unknown: return "ExpressionTypeUnknown";
  case ET_Basic:   return "ExpressionTypeBasic";
  case ET_Phi:     return "ExpressionTypePhi";
  case ET_Factor:  return "ExpressionTypeFactor";
  default:         return "ExpressionType???";
  }
}

class Expression {
private:
  static unsigned LastID;
  unsigned ID;
  ExpressionType EType;
  unsigned Opcode;
  int Version;

  bool Save;
  bool Reload;

public:
  Expression(ExpressionType ET = ET_Base, unsigned O = ~2U, bool S = true)
      : ID(LastID++), EType(ET), Opcode(O), Version(-1),
        Save(S), Reload(false) {}
  // Expression(const Expression &) = delete;
  // Expression &operator=(const Expression &) = delete;
  virtual ~Expression();

  unsigned getID() const { return ID; }
  unsigned getOpcode() const { return Opcode; }
  void setOpcode(unsigned opcode) { Opcode = opcode; }
  ExpressionType getExpressionType() const { return EType; }

  int getVersion() const { return Version; }
  void setVersion(int V) { Version = V; }

  int getSave() const { return Save; }
  void setSave(int S) { Save = S; }

  int getReload() const { return Reload; }
  void setReload(int R) { Reload = R; }

  static unsigned getEmptyKey() { return ~0U; }
  static unsigned getTombstoneKey() { return ~1U; }

  bool operator==(const Expression &Other) const {
    if (getOpcode() != Other.getOpcode())
      return false;
    if (getOpcode() == getEmptyKey() || getOpcode() == getTombstoneKey())
      return true;
    // Compare the expression type for anything but load and store.
    // For load and store we set the opcode to zero.
    // This is needed for load coercion.
    // TODO figure out the reason for this
    // if (getExpressionType() != ET_Load && getExpressionType() != ET_Store &&
    //     getExpressionType() != Other.getExpressionType())
    //   return false;

    return equals(Other);
  }

  virtual bool equals(const Expression &Other) const { return true; }

  virtual hash_code getHashValue() const {
    return hash_combine(getExpressionType(), getOpcode());
  }

  //
  // Debugging support
  //
  virtual void printInternal(raw_ostream &OS) const {
    OS << ExpressionTypeToString(getExpressionType());
    OS << ", V: " << Version;
    OS << ", S: " << (Save ? "T" : "F");
    OS << ", R: " << (Reload ? "T" : "F");
    OS << ", OPC: " << getOpcode() << ", ";
  }

  void print(raw_ostream &OS) const {
    OS << "{ ";
    printInternal(OS);
    OS << "}";
  }

  void dump() const { print(dbgs()); }
}; // class Expression

class IgnoredExpression : public Expression {
protected:
  Instruction *Inst;

public:
  IgnoredExpression(Instruction *I) : IgnoredExpression(I, ET_Ignored) {}
  IgnoredExpression(Instruction *I, ExpressionType ET) : Expression(ET), Inst(I) {}
  IgnoredExpression() = delete;
  IgnoredExpression(IgnoredExpression &) = delete;
  IgnoredExpression &operator=(const IgnoredExpression &) = delete;
  ~IgnoredExpression() override;

  static bool classof(const Expression *EB) {
    return EB->getExpressionType() == ET_Ignored;
  }

  Instruction *getInstruction() const { return Inst; }
  void setInstruction(Instruction *I) { Inst = I; }

  bool equals(const Expression &Other) const override {
    const auto &OU = cast<IgnoredExpression>(Other);
    return Expression::equals(Other) && Inst == OU.Inst;
  }

  hash_code getHashValue() const override {
    return hash_combine(getExpressionType(), Inst);
  }

  //
  // Debugging support
  //
  void printInternal(raw_ostream &OS) const override {
    this->Expression::printInternal(OS);
    OS << "I = " << *Inst;
  }
}; // class IgnoredExpression

class UnknownExpression final : public IgnoredExpression {
public:
  UnknownExpression(Instruction *I) : IgnoredExpression(I, ET_Unknown) {}
  UnknownExpression() = delete;
  UnknownExpression(UnknownExpression &) = delete;
  UnknownExpression &operator=(const UnknownExpression &) = delete;
  ~UnknownExpression() override;

  static bool classof(const Expression *EB) {
    return EB->getExpressionType() == ET_Unknown;
  }
};

class BasicExpression : public Expression {
private:
  typedef ArrayRecycler<Value *> RecyclerType;
  typedef RecyclerType::Capacity RecyclerCapacity;
  SmallVector<Value *, 2> Operands;
  Type *ValueType;

public:
  BasicExpression()
      : BasicExpression({}, ET_Basic) {}
  BasicExpression(SmallVector<Value *, 2> O)
      : BasicExpression(O, ET_Basic) {}
  BasicExpression(SmallVector<Value *, 2> O, ExpressionType ET)
      : Expression(ET), Operands(O) {}
  // BasicExpression(const BasicExpression &) = delete;
  // BasicExpression &operator=(const BasicExpression &) = delete;
  ~BasicExpression() override;

  static bool classof(const Expression *EB) {
    ExpressionType ET = EB->getExpressionType();
    return ET > ET_BasicStart && ET < ET_BasicEnd;
  }

  void swapOperands(unsigned First, unsigned Second) {
    std::swap(Operands[First], Operands[Second]);
  }

  Value *getOperand(unsigned N) const {
    return Operands[N];
  }

  const SmallVector<Value *, 2>& getOperands() const {
    return Operands;
  }

  void addOperand(Value *V) {
    Operands.push_back(V);
  }

  void setOperand(unsigned N, Value *V) {
    assert(N < Operands.size() && "Operand out of range");
    Operands[N] = V;
  }

  unsigned getNumOperands() const { return Operands.size(); }

  void setType(Type *T) { ValueType = T; }
  Type *getType() const { return ValueType; }

  bool equals(const Expression &Other) const override {
    if (getOpcode() != Other.getOpcode())
      return false;

    const auto &OE = cast<BasicExpression>(Other);
    return getType() == OE.getType() && Operands == OE.Operands;
  }

  hash_code getHashValue() const override {
    return hash_combine(getExpressionType(), getOpcode(), ValueType,
        hash_combine_range(Operands.begin(), Operands.end()));
  }

  //
  // Debugging support
  //
  void printInternal(raw_ostream &OS) const override {
    this->Expression::printInternal(OS);
    OS << "OPS: { ";
    for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
      OS << "[" << i << "] = ";
      Operands[i]->printAsOperand(OS);
      if (i + 1 != e)
        OS << ", ";
    }
    OS << " }";
  }
}; // class BasicExpression

class PHIExpression final : public BasicExpression {
private:
  BasicBlock *BB;

public:
  PHIExpression(SmallVector<Value *, 2> O, BasicBlock *BB)
      : BasicExpression(O, ET_Phi), BB(BB) {}
  PHIExpression() = default;
  // PHIExpression(const PHIExpression &) = delete;
  // PHIExpression &operator=(const PHIExpression &) = delete;
  ~PHIExpression() override;

  static bool classof(const Expression *EB) {
    return EB->getExpressionType() == ET_Phi;
  }

  bool equals(const Expression &Other) const override {
    if (!this->BasicExpression::equals(Other))
      return false;
    const PHIExpression &OE = cast<PHIExpression>(Other);
    return BB == OE.BB;
  }

  hash_code getHashValue() const override {
    return hash_combine(this->BasicExpression::getHashValue(), BB);
  }

  //
  // Debugging support
  //
  void printInternal(raw_ostream &OS) const override {
    this->BasicExpression::printInternal(OS);
    OS << "BB = " << BB->getValueName();
  }
}; // class PHIExpression

class FactorExpression final : public Expression {
private:
  const Expression &PE;
  const BasicBlock &BB;
  SmallVector<const BasicBlock *, 8> Pred;
  SmallVector<Expression *, 8> Versions;

  // If True expression is Anticipated on every path leading from this Factor
  bool DownSafe;
  // True if an Operand is a Real expressin and not Factor or Expression Operand
  // definition(⊥)
  SmallVector<bool, 8> HasRealUse;

  bool CanBeAvail;
  bool Later;

public:
  FactorExpression(const Expression &PE, const BasicBlock &BB,
                   SmallVector<const BasicBlock *, 8> P)
      : Expression(ET_Factor), PE(PE), BB(BB), Pred(P),
                   Versions(P.size(), nullptr),
                   DownSafe(true), HasRealUse(P.size(), false),
                   CanBeAvail(true), Later(true) { }
  FactorExpression() = delete;
  FactorExpression(const FactorExpression &) = delete;
  FactorExpression &operator=(const FactorExpression &) = delete;
  ~FactorExpression() override;

  const Expression& getPExpr() const { return PE; }

  size_t getPredIndex(BasicBlock * B) const {
    for (size_t i = 0; i < Pred.size(); ++i) {
      if (Pred[i] == B) {
        return i;
      }
    }
    return -1;
  }

  size_t getVExprNum() { return Versions.size(); }
  void setVExpr(unsigned P, Expression * V) { Versions[P] = V; }
  Expression * getVExpr(unsigned P) { return Versions[P]; }
  size_t getVExprIndex(Expression * V) {
    for(size_t i = 0, l = Versions.size(); i < l; ++i) {
      if (Versions[i] == V)
        return i;
    }
    return -1;
  }
  SmallVector<Expression *, 8> getVExprs() { return Versions; };

  bool getDownSafe() const { return DownSafe; }
  void setDownSafe(bool DS) { DownSafe = DS; }

  bool getCanBeAvail() const { return CanBeAvail; }
  void setCanBeAvail(bool CBA) { CanBeAvail = CBA; }

  bool getLater() const { return Later; }
  void setLater(bool L) { Later = L; }

  bool getWillBeAvail() const { return CanBeAvail && !Later; }

  void setHasRealUse(unsigned P, bool HRU) { HasRealUse[P] = HRU; }
  bool getHasRealUse(unsigned P) const { return HasRealUse[P]; }

  static bool classof(const Expression *EB) {
    return EB->getExpressionType() == ET_Factor;
  }

  bool equals(const Expression &Other) const override {
    if (!this->Expression::equals(Other))
      return false;
    if (auto OE = dyn_cast<FactorExpression>(&Other)) {
      return &BB == &OE->BB;
    }
    return false;
  }

  hash_code getHashValue() const override {
    return hash_combine(this->Expression::getHashValue(), &PE, &BB,
        hash_combine_range(Versions.begin(), Versions.end()));
  }

  //
  // Debugging support
  //
  void printInternal(raw_ostream &OS) const override {
    this->Expression::printInternal(OS);
    OS << "BB = ";
    BB.printAsOperand(OS, false);
    OS << ", E = " << PE.getID()
       << ", V = <";
    for (unsigned i = 0, l = Versions.size(); i < l; ++i) {
      if (Versions[i]) {
        OS << Versions[i]->getVersion();
      } else {
        OS << "⊥";
      }
      if (i + 1 != l) OS << ",";
    }
    OS << ">";
    OS << ", DS: " << (DownSafe ? "T" : "F");
    OS << ", HRU: <";
    for (unsigned i = 0, l = HasRealUse.size(); i < l; ++i) {
      OS << (HasRealUse[i] ? "T" : "F");
      if (i + 1 != l) OS << ",";
    }
    OS << ">";
    OS << ", CBA: " << (CanBeAvail ? "T" : "F");
    OS << ", L: " << (Later ? "T" : "F");
    OS << ", WBA: " << (getWillBeAvail() ? "T" : "F");
  }
}; // class FactorExpression

class FactorRenamingContext {
public:
  unsigned Counter;
  std::stack<int> Stack;
};

} // end namespace ssapre

using namespace ssapre;

template <> struct DenseMapInfo<const Expression *> {
  static const Expression *getEmptyKey() {
    auto Val = static_cast<uintptr_t>(-1);
    Val <<= PointerLikeTypeTraits<const Expression *>::NumLowBitsAvailable;
    return reinterpret_cast<const Expression *>(Val);
  }
  static const Expression *getTombstoneKey() {
    auto Val = static_cast<uintptr_t>(~1U);
    Val <<= PointerLikeTypeTraits<const Expression *>::NumLowBitsAvailable;
    return reinterpret_cast<const Expression *>(Val);
  }
  static unsigned getHashValue(const Expression *V) {
    return static_cast<unsigned>(V->getHashValue());
  }
  static bool isEqual(const Expression *LHS, const Expression *RHS) {
    if (LHS == RHS)
      return true;
    if (LHS == getTombstoneKey() || RHS == getTombstoneKey() ||
        LHS == getEmptyKey() || RHS == getEmptyKey())
      return false;
    return LHS->equals(*RHS);
  }
};

/// Performs SSA PRE pass.
class SSAPRE : public PassInfoMixin<SSAPRE> {
  const DataLayout *DL;
  const TargetLibraryInfo *TLI;
  AssumptionCache *AC;
  DominatorTree *DT;
  ReversePostOrderTraversal<Function *> *RPOT;

  // Number of function arguments, used by ranking
  unsigned int NumFuncArgs;

  // DFS info.
  // This contains a mapping from Instructions to DFS numbers.
  // The numbering starts at 1. An instruction with DFS number zero
  // means that the instruction is dead.
  typedef DenseMap<const Value *, unsigned> InstrToOrderType;
  InstrToOrderType InstrDFS;
  InstrToOrderType InstrSDFS;

  // This contains the mapping DFS numbers to instructions.
  typedef SmallVector<const Value *, 32> OrderedInstrType;
  OrderedInstrType DFSToInstr;

  // Instruction-to-Expression map
  DenseMap<const Instruction *, Expression *> InstToVExpr;
  DenseMap<Expression *, Instruction *> VExprToInst;

  // ProtoExpression-to-Instructions map
  DenseMap<const Expression *, SmallPtrSet<const Instruction *, 5>> PExprToInsts;

  // ProtoExpression-to-BasicBlock map
  DenseMap<const Expression *, SmallPtrSet<BasicBlock *, 5>> PExprToBlocks;

  // BasicBlock-to-FactorList map
  DenseMap<const BasicBlock *, SmallPtrSet<FactorExpression *, 5>> BlockToFactors;

  // ProtoExpression-to-VersionedExpressions
  DenseMap<const Expression *, SmallPtrSet<Expression *, 5>> PExprToVExprs;

  // VersionedExpression-to-ProtoVersioned
  DenseMap<Expression *, const Expression *> VExprToPExpr;

  SmallPtrSet<FactorExpression *, 32> FExprs;

  DenseMap<const Expression *, DenseMap<int, Expression *>> AvailDef;

  DenseMap<const BasicBlock *, SmallPtrSet<const Expression *, 5>> BlockToInserts;

  SmallPtrSet<Instruction *, 8> KillList;

public:
  PreservedAnalyses run(Function &F, AnalysisManager<Function> &AM);

private:
  friend ssapre::SSAPRELegacy;

  std::pair<unsigned, unsigned> AssignDFSNumbers(BasicBlock *B, unsigned Start,
                                                 InstrToOrderType *M,
                                                 OrderedInstrType *V);

  // This function provides global ranking of operations so that we can place them
  // in a canonical order.  Note that rank alone is not necessarily enough for a
  // complete ordering, as constants all have the same rank.  However, generally,
  // we will simplify an operation with all constants so that it doesn't matter
  // what order they appear in.
  unsigned int GetRank(const Value *V) const;

  // This is a function that says whether two commutative operations should
  // have their order swapped when canonicalizing.
  bool ShouldSwapOperands(const Value *A, const Value *B) const;

  bool FillInBasicExpressionInfo(Instruction &I, BasicExpression *E);

  // Take a Value returned by simplification of Expression E/Instruction
  // I, and see if it resulted in a simpler expression. If so, return
  // that expression.
  // TODO: Once finished, this should not take an Instruction, we only
  // use it for printing.
  Expression * CheckSimplificationResults(Expression *E,
                                          Instruction &I,
                                          Value *V);
  Expression * CreateIgnoredExpression(Instruction &I);
  Expression * CreateUnknownExpression(Instruction &I);
  Expression * CreateBasicExpression(Instruction &I);
  Expression * CreatePHIExpression(Instruction &I);
  FactorExpression * CreateFactorExpression(const Expression &E,
                                            const BasicBlock &B);
  Expression * CreateExpression(Instruction &I);

  bool IgnoreExpression(const Expression &E);

  void PrintDebug(const std::string &Caption);

  void Init();

  void FactorInsertion();

  void Rename();

  void ResetDownSafety(FactorExpression &F, unsigned O);
  void DownSafety();

  void ComputeCanBeAvail();
  void ResetCanBeAvail(FactorExpression &F);
  void ComputeLater();
  void ResetLater(FactorExpression &F);
  void WillBeAvail();

  void FinalizeVisit(BasicBlock &B);
  void Finalize();

  bool CodeMotion();

  PreservedAnalyses runImpl(Function &F, AssumptionCache &_AC,
                            TargetLibraryInfo &_TLI, DominatorTree &_DT);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_SSAPRE_H
