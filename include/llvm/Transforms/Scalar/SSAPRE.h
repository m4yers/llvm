//===-------- SSAPRE.h - SSA PARTIAL REDUNDANCY ELIMINATION -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the interface for LLVM's SSA Partial Redundancy
// Elimination pass.
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
#include "llvm/Support/Allocator.h"
#include <stack>

namespace llvm {

namespace ssapre LLVM_LIBRARY_VISIBILITY {

class SSAPRELegacy;

namespace phi_factoring {
class TokenPropagationSolver;
}


//===----------------------------------------------------------------------===//
// Pass Expressions
//===----------------------------------------------------------------------===//

enum ExpressionType {
  ET_Base,
  ET_Bottom,
  ET_Ignored,
  ET_Unknown,
  ET_Constant,
  ET_Variable,
  ET_Factor, // Phi for expressions, Φ in paper
  ET_BasicStart,
  ET_Basic,
  ET_Phi,
  // TODO later
  // ET_Call,
  // ET_AggregateValue,
  // ET_Load,
  // ET_Store,
  ET_BasicEnd

};

inline std::string ExpressionTypeToString(ExpressionType ET) {
  switch (ET) {
  case ET_Ignored:   return "ExpressionTypeIgnored";
  case ET_Unknown:   return "ExpressionTypeUnknown";
  case ET_Basic:     return "ExpressionTypeBasic";
  case ET_Phi:       return "ExpressionTypePhi";
  case ET_Factor:    return "ExpressionTypeFactor";
  case ET_Variable:  return "ExpressionTypeVariable";
  case ET_Constant:  return "ExpressionTypeConstant";
  default:           return "ExpressionType???";
  }
}

// During debug it is useful to have distinct versions for stuff
typedef int ExpVersion_t;
enum VersionRanges : ExpVersion_t {
  VR_Unset = -1,
  VR_Bottom = -2,
  VR_VariableLo = -1000,
  VR_VariableHi = -1999,
  VR_ConstantLo = -2000,
  VR_ConstantHi = -3999,
  VR_IgnoredLo  = -4000,
  VR_IgnoredHi  = -9999,
};

class Expression {
private:
  ExpressionType EType;
  unsigned Opcode;
  ExpVersion_t Version;

  Instruction *Proto;

  int Saved;

public:
  Expression(ExpressionType ET = ET_Base, unsigned O = ~2U,
             ExpVersion_t V = VR_Unset)
      : EType(ET), Opcode(O), Version(V),
        Proto(nullptr),
        Saved(0) {}
  Expression(const Expression &) = delete;
  Expression &operator=(const Expression &) = delete;
  virtual ~Expression();

  unsigned getOpcode() const { return Opcode; }
  void setOpcode(unsigned opcode) { Opcode = opcode; }
  ExpressionType getExpressionType() const { return EType; }

  ExpVersion_t getVersion() const { return Version; }
  void setVersion(ExpVersion_t V) { Version = V; }

  Instruction * getProto() const { return Proto; }
  void setProto(Instruction *I) { Proto = I; }

  bool getSave() const { return Saved > 0; }
  void setSave(int S) { Saved = S; }
  void clrSave() { Saved = 0; }
  void remSave() { Saved--; }
  void addSave() { Saved++; }
  void addSave(int S) { Saved += S; }

  static unsigned getEmptyKey() { return ~0U; }
  static unsigned getTombstoneKey() { return ~1U; }

  bool operator==(const Expression &O) const {
    if (getOpcode() != O.getOpcode())
      return false;
    if (getOpcode() == getEmptyKey() || getOpcode() == getTombstoneKey())
      return true;
    // Compare the expression type for anything but load and store.
    // For load and store we set the opcode to zero.
    // This is needed for load coercion.
    // TODO figure out the reason for this
    // if (getExpressionType() != ET_Load && getExpressionType() != ET_Store &&
    //     getExpressionType() != O.getExpressionType())
    //   return false;

    return equals(O);
  }

  virtual bool equals(const Expression &O) const {
    if (EType == O.EType && Opcode == O.Opcode && Version == O.Version) {
      assert(Saved == O.Saved &&
          "Expressions are not fully equal");
      return true;
    }
    return false;
  }

  virtual void printInternal(raw_ostream &OS) const {
    OS << ExpressionTypeToString(getExpressionType());
    OS << ", V: " << Version;
    OS << ", S: " << Saved;
    // OS << ", OPC: " << getOpcode() << ", ";
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

  bool equals(const Expression &O) const override {
    if (auto OU = dyn_cast<IgnoredExpression>(&O))
      return Expression::equals(O) && Inst == OU->Inst;
    return false;
  }

  void printInternal(raw_ostream &OS) const override {
    this->Expression::printInternal(OS);
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

class VariableExpression final : public Expression {
private:
  Value &VariableValue;

public:
  VariableExpression(Value &V)
    : Expression(ET_Variable), VariableValue(V) {}
  VariableExpression() = delete;
  VariableExpression(const VariableExpression &) = delete;
  VariableExpression &operator=(const VariableExpression &) = delete;

  Value &getValue() { return VariableValue; }

  static bool classof(const Expression *EB) {
    return EB->getExpressionType() == ET_Variable;
  }

  bool equals(const Expression &Other) const override {
    if (auto OC = dyn_cast<VariableExpression>(&Other)) {
      return &VariableValue == &OC->VariableValue;
    }
    return false;
  }

  void printInternal(raw_ostream &OS) const override {
    this->Expression::printInternal(OS);
    OS << ", V: " << VariableValue;
  }
};

class ConstantExpression final : public Expression {
private:
  Constant &ConstantValue;

public:
  ConstantExpression(Constant &C)
      : Expression(ET_Constant), ConstantValue(C) {}
  ConstantExpression() = delete;
  ConstantExpression(const ConstantExpression &) = delete;
  ConstantExpression &operator=(const ConstantExpression &) = delete;

  Constant &getConstant() { return ConstantValue; }

  static bool classof(const Expression *EB) {
    return EB->getExpressionType() == ET_Constant;
  }

  bool equals(const Expression &Other) const override {
    if (auto OC = cast<ConstantExpression>(&Other)) {
      return &ConstantValue == &OC->ConstantValue;
    }
    return false;
  }

  void printInternal(raw_ostream &OS) const override {
    this->Expression::printInternal(OS);
    OS << ", C:" << ConstantValue;
  }
};

class BasicExpression : public Expression {
private:
  SmallVector<Value *, 2> Operands; // TODO use Expressions here
  Type *ValueType;

public:
  BasicExpression(ExpressionType ET = ET_Basic)
      : Expression(ET), ValueType(nullptr) {}
  BasicExpression(const BasicExpression &) = delete;
  BasicExpression &operator=(const BasicExpression &) = delete;
  ~BasicExpression() override;

  static bool classof(const Expression *EB) {
    ExpressionType ET = EB->getExpressionType();
    return ET > ET_BasicStart && ET < ET_BasicEnd;
  }

  void addOperand(Value *V) {
    Operands.push_back(V);
  }
  Value *getOperand(unsigned N) const {
    return Operands[N];
  }
  void setOperand(unsigned N, Value *V) {
    assert(N < Operands.size() && "Operand out of range");
    Operands[N] = V;
  }
  void swapOperands(unsigned First, unsigned Second) {
    std::swap(Operands[First], Operands[Second]);
  }
  const SmallVector<Value *, 2>& getOperands() const {
    return Operands;
  }

  unsigned getNumOperands() const { return Operands.size(); }

  void setType(Type *T) { ValueType = T; }
  Type *getType() const { return ValueType; }

  bool equals(const Expression &O) const override {
    if (!Expression::equals(O))
      return false;

    if (auto OE = dyn_cast<BasicExpression>(&O)) {
      return getType() == OE->getType() && Operands == OE->Operands;
    }
    return false;
  }

  void printInternal(raw_ostream &OS) const override {
    this->Expression::printInternal(OS);
    OS << ", OPS: " << getNumOperands();
  }
}; // class BasicExpression

class PHIExpression final : public BasicExpression {
private:
  Expression *PE; // common PE of the expressions it functions on
  BasicBlock *BB;

public:
  PHIExpression(BasicBlock *BB)
    : BasicExpression(ET_Phi), PE(PHIExpression::getPExprNotSet()), BB(BB) {}
  PHIExpression() = delete;
  PHIExpression(const PHIExpression &) = delete;
  PHIExpression &operator=(const PHIExpression &) = delete;
  ~PHIExpression() override;

  bool isCommonPExprSet() const {
    return PE != PHIExpression::getPExprNotSet();
  }
  bool hasCommonPExpr() const {
    return isCommonPExprSet() && PE != PHIExpression::getPExprMismatch();
  }
  void setCommonPExpr(Expression *E) { assert(E); PE = E; }
  Expression * getCommonPExpr() const { return PE; }

  static Expression * getPExprNotSet()   { return (Expression *)~0; }
  static Expression * getPExprMismatch() { return (Expression *)~1; }

  static bool classof(const Expression *EB) {
    return EB->getExpressionType() == ET_Phi;
  }

  bool equals(const Expression &O) const override {
    if (!this->BasicExpression::equals(O))
      return false;
    if (auto OE = dyn_cast<PHIExpression>(&O)) {
      return BB == OE->BB;
    }
    return false;
  }

  void printInternal(raw_ostream &OS) const override {
    this->BasicExpression::printInternal(OS);
    OS << ", BB: ";
    BB->printAsOperand(dbgs());
  }
}; // class PHIExpression

class FactorExpression final : public Expression {
private:
  const BasicBlock &BB;

  // This is Proto Expression of the operands if they happen to be of the same
  // Proto
  const Expression *PE;

  // This maps BB to Index inside Versions and HRU
  SmallPtrSet<BasicBlock *, 4> Blocks;
  DenseMap<BasicBlock *, size_t> Pred;
  DenseMap<size_t, BasicBlock *> Indices;

  // The Versioned Expressions that this Factor joins
  SmallVector<Expression *, 8> Versions;

  // If True this Factor is linked to already existing PHI function
  bool Materialized;

  // If True this Factor merges init value and calculated value inside a cycle.
  // These must be treated differently since with all formal predicates
  // calculated this Factor/PHI will be replaced with the actual computation,
  // but instead it should be pushed up to the init block. And if it happens
  // that the Factor is not a Cycle, we still will push calculation to the init
  // block since any constant/variable is regarded as bottom. This of course
  // add register pressure
  //
  // The second course of actions would be to push the computation directly to
  // the first use place, but we need to prove that this place is not inside a
  // cycle, or at least in the same cycle as init.
  bool Cycle;

  // If True expression is Anticipated on every path leading from this Factor
  bool DownSafe;

  // True if an Operand is a Real expression and not Factor or Expression
  // Operand definition(⊥)
  SmallVector<bool, 8> HasRealUse;

  bool CanBeAvail;
  bool Later;

public:
  FactorExpression(const BasicBlock &BB)
      : Expression(ET_Factor), BB(BB),
                   Materialized(false), Cycle(false), DownSafe(true),
                   CanBeAvail(true), Later(true) { }
  FactorExpression() = delete;
  FactorExpression(const FactorExpression &) = delete;
  FactorExpression &operator=(const FactorExpression &) = delete;
  ~FactorExpression() override;

  const BasicBlock * getBB() const { return &BB; }

  void setIsMaterialized(bool L) { Materialized = L; }
  bool getIsMaterialized() const { return Materialized; }

  void setIsCycle(bool C) { Cycle = C; }
  bool getIsCycle() const { return Cycle; }

  void setPExpr(const Expression *E) { PE = E; }
  const Expression* getPExpr() const { return PE; }

  void addPred(BasicBlock *B, size_t I) {
    Pred[B] = I;
    Indices[I] = B;
    Blocks.insert(B);
    Versions.push_back(nullptr);
    HasRealUse.push_back(false);
  }

  const SmallPtrSet<BasicBlock *, 4> &getPreds() { return Blocks; }

  void setVExpr(BasicBlock *B, Expression * V) { Versions[Pred[B]] = V; }

  void replaceVExpr(Expression *E, Expression *V) {
    assert(hasVExpr(E));
    Versions[getVExprIndex(E)] = V;
  }

  bool hasVExpr(const Expression *V) const { return getVExprIndex(V) != -1UL; }
  SmallVector<Expression *, 8> getVExprs() { return Versions; };
  Expression * getVExpr(BasicBlock *B) const { return Versions[Pred.lookup(B)]; }

  size_t getVExprNum() const { return Versions.size(); }
  size_t getVExprIndex(const Expression *V) const  {
    for(size_t i = 0, l = Versions.size(); i < l; ++i) {
      if (Versions[i] == V)
        return i;
    }
    return -1;
  }

  bool getDownSafe() const { return DownSafe; }
  void setDownSafe(bool DS) { DownSafe = DS; }

  bool getCanBeAvail() const { return CanBeAvail; }
  void setCanBeAvail(bool CBA) { CanBeAvail = CBA; }

  // Later marks the latest possible computationl point, or in other words the
  // value does not come from predecessor and computed either here or nowhere
  // at all
  bool getLater() const { return Later; }
  void setLater(bool L) { Later = L; }

  bool getWillBeAvail() const { return CanBeAvail && !Later; }

  // This is only True when this Factor is Materialized and is DownSafe. It is
  // possible to gave False WBA at the same time. The meaning of this is that
  // the Factor is already materialized into a PHI and this PHI is used.  bool
  // getIsAvail() const { return CanBeAvail && Materialized; }

  bool getHasRealUse(Expression *E) const { return HasRealUse[getVExprIndex(E)]; }
  void setHasRealUse(Expression *E, bool HRU) { HasRealUse[getVExprIndex(E)] = HRU; }

  static bool classof(const Expression *EB) {
    return EB->getExpressionType() == ET_Factor;
  }

  bool equals(const Expression &O) const override {
    if (!this->Expression::equals(O))
      return false;
    if (auto OE = dyn_cast<FactorExpression>(&O)) {
      return &BB == &OE->BB;
    }
    return false;
  }

  void printInternal(raw_ostream &OS) const override {
    this->Expression::printInternal(OS);
    OS << ", BB: ";
    BB.printAsOperand(OS, false);
    OS << ", PE: " << (void *)PE;
    OS << ", MAT: " << Materialized;
    OS << ", CYC: " << Cycle;
    OS << ", V: <";
    for (unsigned i = 0, l = Versions.size(); i < l; ++i) {
      if (Versions[i]) {
        OS << Versions[i]->getVersion();
      } else {
        OS << "×";
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
    // OS << ", AV: " << (getIsAvail() ? "T" : "F");
  }
}; // class FactorExpression

} // end namespace ssapre

using namespace ssapre;

typedef SmallVector<BasicBlock *, 32> BBVector_t;
typedef SmallVector<FactorExpression *, 32> FEVector_t;
typedef std::pair<unsigned, Expression *> UIntExpressionPair_t;
typedef std::stack<UIntExpressionPair_t> ExprStack_t;
typedef SmallVector<Expression *, 32> ExpVector_t;
typedef DenseMap<const Expression *, ExprStack_t> PExprToVExprStack_t;

/// Performs SSA PRE pass.
class SSAPRE : public PassInfoMixin<SSAPRE> {
  const DataLayout *DL;
  const TargetLibraryInfo *TLI;
  AssumptionCache *AC;
  DominatorTree *DT;
  Function *Func;
  ReversePostOrderTraversal<Function *> *RPOT;

  BumpPtrAllocator ExpressionAllocator;

  ExpVersion_t LastVariableVersion;
  ExpVersion_t LastConstantVersion;
  ExpVersion_t LastIgnoredVersion;

  SmallVector<const BasicBlock *, 32> JoinBlocks;

  // Values' stuff
  DenseMap<Expression *, const Value *> ExpToValue;
  DenseMap<const Value *, Expression *> ValueToExp;

  // Arguments' stuff
  unsigned int NumFuncArgs;
  DenseMap<VariableExpression *, Value *> VAExpToValue;
  DenseMap<const Value *, VariableExpression *> ValueToVAExp;

  // Constants' stuff
  DenseMap<ConstantExpression *, Value *> COExpToValue;
  DenseMap<const Value *, ConstantExpression *> ValueToCOExp;

  DenseMap<const FactorExpression *, const PHINode *> FactorToPHI;
  DenseMap<const PHINode *, const FactorExpression *> PHIToFactor;

  // DFS info.
  // This contains a mapping from Instructions to DFS numbers.  The numbering
  // starts at 1. An instruction with DFS number zero means that the
  // instruction is dead.
  typedef DenseMap<const Value *, unsigned> InstrToOrderType;
  InstrToOrderType InstrDFS;
  InstrToOrderType InstrSDFS;

  // This contains the mapping DFS numbers to instructions.
  typedef SmallVector<const Value *, 32> OrderedInstrType;
  OrderedInstrType DFSToInstr;

  // Instruction-to-Expression map
  DenseMap<const Instruction *, Expression *> InstToVExpr;
  DenseMap<const Expression *, Instruction *> VExprToInst;

  // ProtoExpression-to-Instructions map
  DenseMap<const Expression *, SmallPtrSet<const Instruction *, 5>> PExprToInsts;

  // ProtoExpression-to-VersionedExpressions
  DenseMap<const Expression *, SmallPtrSet<Expression *, 5>> PExprToVExprs;

  DenseMap<const Expression *, DenseMap<int,ExpVector_t>> PExprToVersions;

  // ProtoExpression-to-BasicBlock map
  DenseMap<const Expression *, SmallPtrSet<BasicBlock *, 5>> PExprToBlocks;

  // BasicBlock-to-FactorList map
  DenseMap<const BasicBlock *, SmallVector<FactorExpression *, 5>> BlockToFactors;
  DenseMap<const FactorExpression *, const BasicBlock *> FactorToBlock;

  // VersionedExpression-to-ProtoVersioned
  DenseMap<const Expression *, const Expression *> VExprToPExpr;

  SmallPtrSet<FactorExpression *, 32> FExprs;

  // This map contains 1-to-1 correspondence between Expression Occurrence and
  // its Definition. Upon initialization Keys will be equal to Values, once
  // an Expression assumes existing Version it must define its Definition, so
  // that during kill time we could replace its use with a proper definition.
  DenseMap<Expression *, Expression *> Substitutions;

  // Store all the PHIs that are considered to be Factors at any point in the
  // pass. Useful during kill time to separate ordinal and factored phis, since
  // former do save their operands, but later do not.
  DenseMap<PHINode *, bool> FactoredPHIs;
  SmallVector<Instruction *, 32> KillList;

public:
  PreservedAnalyses run(Function &F, AnalysisManager<Function> &AM);

private:
  friend ssapre::SSAPRELegacy;
  friend ssapre::phi_factoring::TokenPropagationSolver;

  // Return a reference to the vector containing all Expressions that share
  // the same version with F, by definition those occur after the F
  ExpVector_t & GetFactorVersions(const FactorExpression *F);

  // Return a reference to the vector containing all Expressions that share
  // the same version with E, these versions may occur before or after E
  ExpVector_t & GetExpVersions(const Expression *E);

  // Check whether an Expression is a ⊥ value. It can be not only a real bottom
  // value but a constant or a variable since they do not provide a computation
  bool IsBottom(const Expression *E);

  bool IsVariableOrConstant(const Expression *E);
  bool IsFactoredPHI(Instruction *I);

  // Check whether an Expression is Inductive, in a sense that it uses a
  // phi-result and is an argument to this same phi. Such Inductive Expressions
  // cannot be moved out of the enclosing cycle bounded by this phi.
  bool IsInductionExpression(const FactorExpression *F, const Expression *E);
  // Same as above but restricted to a particular Factor
  bool IsInductionExpression(const Expression *E);

  bool StrictlyDominates(const Expression *Def, const Expression *Use);
  // Not Strictly implies Def == Use -> True
  bool NotStrictlyDominates(const Expression *Def, const Expression *Use);

  // Check whether Expression operands' definitions dominate the Factor
  bool OperandsDominate(const Expression *E, const Expression *F);

  // Find out whether Expression versions are used on a Path before(including)
  // another Expression occurance
  bool HasRealUseBefore(const Expression *S, const BBVector_t &P,
                        const Expression *E);

  // Find out whether Factor(its versions) is used on a Path before(including)
  // another Expression occurance
  bool FactorHasRealUseBefore(const FactorExpression *F, const BBVector_t &P,
                              const Expression *E);

  bool IgnoreExpression(const Expression *E);
  bool IsToBeKilled(Expression *E);
  bool IsToBeKilled(Instruction *I);
  bool AllUsersKilled(const Instruction *I);

  void SetOrderBefore(Instruction *I, Instruction *B);
  void SetAllOperandsSave(Instruction *I);
  void AddSubstitution(Expression * E, Expression * S);

  // Go through all the Substitutions of the Expression and return the most
  // recent one
  Expression * GetSubstitution(Expression * E);

  // Go through all the substitutions of the Expression and return the most
  // recent value available
  Value * GetSubstituteValue(Expression * E);

  void AddConstant(ConstantExpression *CE, Constant *C);
  void AddExpression(Expression *PE, Expression *VE, Instruction *I,
                     BasicBlock *B);

  void AddFactor(FactorExpression *FE, const Expression *PE, const BasicBlock *B);
  void KillFactor(FactorExpression *);
  void MaterializeFactor(FactorExpression *FE, PHINode *PHI);
  void ReplaceFactor(FactorExpression * FE, Expression * E);
  void ReplaceMaterializedFactor(FactorExpression * FE, Expression * E);

  // This function provides global ranking of operations so that we can place
  // them in a canonical order.  Note that rank alone is not necessarily enough
  // for a complete ordering, as constants all have the same rank.  However,
  // generally, we will simplify an operation with all constants so that it
  // doesn't matter what order they appear in.
  unsigned GetRank(const Value *V) const;

  // This is a function that says whether two commutative operations should
  // have their order swapped when canonicalizing.
  bool ShouldSwapOperands(const Value *A, const Value *B) const;

  bool FillInBasicExpressionInfo(Instruction &I, BasicExpression *E);

  std::pair<unsigned, unsigned>
  AssignDFSNumbers(BasicBlock *B, unsigned Start, InstrToOrderType *M,
                   OrderedInstrType *V);

  // Take a Value returned by simplification of Expression E/Instruction I, and
  // see if it resulted in a simpler expression. If so, return that expression.
  Expression *
  CheckSimplificationResults(Expression *E, Instruction &I, Value *V);

  ConstantExpression * CreateConstantExpression(Constant &C);
  VariableExpression * CreateVariableExpression(Value &V);
  Expression * CreateIgnoredExpression(Instruction &I);
  Expression * CreateUnknownExpression(Instruction &I);
  Expression * CreateBasicExpression(Instruction &I);
  Expression * CreatePHIExpression(PHINode &I);

  FactorExpression *
  CreateFactorExpression(const Expression &E, const BasicBlock &B);

  Expression * CreateExpression(Instruction &I);

  void Init(Function &F);
  void Fini();

  void FactorInsertion();

  void Rename();

  void ResetDownSafety(FactorExpression *F, Expression *E);
  void DownSafety();

  void ComputeCanBeAvail();
  void ResetCanBeAvail(FactorExpression *F);
  void ComputeLater();
  void ResetLater(FactorExpression *F);
  void WillBeAvail();

  void Finalize();

  bool FactorGraphWalk();
  bool PHIInsertion();
  bool ApplySubstitutions();
  bool KillEmAll();
  bool CodeMotion();

  void PrintDebug(const std::string &Caption);

  PreservedAnalyses
  runImpl(Function &F, AssumptionCache &_AC, TargetLibraryInfo &_TLI,
          DominatorTree &_DT);
};
} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_SSAPRE_H
