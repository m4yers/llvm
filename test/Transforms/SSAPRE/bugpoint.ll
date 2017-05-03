; RUN: opt < %s -ssapre -S | FileCheck %s
target datalayout = "e-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

declare i32 @memcmp(i8*, i8*, i64)

; CHECK-LABEL: bugpoint_1(
; CHECK:       getelementptr
; CHECK:       br
; CHECK:       phi
; CHECK:       call
; CHECK:       br
; CHECK:       switch
; CHECK:       switch
; CHECK:       ret
define internal i64 @bugpoint_1( i8* %arg) {
  br i1 undef, label %.critedge1, label %.critedge41

.critedge1:
  %add.ptr.c = getelementptr inbounds i8, i8* %arg, i64 undef
  %call34.c = call i32 @memcmp(i8* %add.ptr.c, i8* undef, i64 undef)
  br label %.critedge34

.critedge41:
  %add.ptr = getelementptr inbounds i8, i8* %arg, i64 undef
  %call34 = call i32 @memcmp(i8* %add.ptr, i8* undef, i64 undef)
  br i1 undef, label %.critedge34, label %1

  switch i32 undef, label %.critedge1 [
    i32 0, label %.critedge34
    i32 92, label %.critedge1
    i32 123, label %2
    i32 125, label %2
  ]

  switch i64 undef, label %.critedge34 [
    i64 -1, label %.critedge34
    i64 1, label %.critedge34
  ]

.critedge34:
  ret i64 undef
}

; CHECK-LABEL: bugpoint_2(
; CHECK:       br
; CHECK:       phi
; CHECK:       br
; CHECK:       add
; CHECK:       br
; CHECK:       br
; CHECK:       br
; CHECK:       br
; CHECK-NOT:   phi
; CHECK:       unreachable
define fastcc void @bugpoint_2() unnamed_addr {
entry:
  br label %do.body.i

do.body.i:                                        ; preds = %for.end.i, %entry
  %nstrings.0 = phi i64 [ 0, %entry ], [ %add19.i, %for.end.i ]
  br label %for.end.i

for.end.i:                                        ; preds = %do.body.i
  %add19.i = add i64 %nstrings.0, 1
  br i1 undef, label %do.body.i, label %while.cond.preheader.i

while.cond.preheader.i:                           ; preds = %for.end.i
  %add19.i.lcssa = phi i64 [ %add19.i, %for.end.i ]
  br label %while.cond54.preheader.i

while.cond54.preheader.i:                         ; preds = %while.cond.preheader.i
  br i1 undef, label %while.body57.i.preheader, label %LoadStringArray.exit

while.body57.i.preheader:                         ; preds = %while.cond54.preheader.i
  br label %LoadStringArray.exit

LoadStringArray.exit:                             ; preds = %while.body57.i.preheader, %while.cond54.preheader.i
  %add19.i30 = phi i64 [ %add19.i.lcssa, %while.body57.i.preheader ], [ %add19.i.lcssa, %while.cond54.preheader.i ]
  unreachable
}

; CHECK-LABEL: bugpoint_3(
; CHECK:       br
; CHECK:       phi
; CHECK:       br
; CHECK:       br
; CHECK:       br
; CHECK:       phi
; CHECK:       br
; CHECK:       phi
; CHECK:       br
; CHECK:       br
; CHECK:       br
define fastcc void @bugpoint_3() unnamed_addr {
entry:
  br label %for.cond80

for.cond80:                                       ; preds = %for.inc148, %entry
  %totnumassigns.3 = phi i16 [ %totnumassigns.4, %for.inc148 ], [ 0, %entry ]
  br i1 undef, label %for.cond85, label %do.cond

for.cond85:                                       ; preds = %for.cond80
  br i1 undef, label %if.then115, label %for.inc148

if.then115:                                       ; preds = %for.cond85
  br label %for.inc148

for.inc148:                                       ; preds = %if.then115, %for.cond85
  %totnumassigns.4 = phi i16 [ undef, %if.then115 ], [ %totnumassigns.3, %for.cond85 ]
  br label %for.cond80

do.cond:                                          ; preds = %for.cond80
  br label %for.cond160

for.cond160:                                      ; preds = %for.inc248, %do.cond
  %totnumassigns.5 = phi i16 [ %totnumassigns.6, %for.inc248 ], [ %totnumassigns.3, %do.cond ]
  br i1 undef, label %for.inc248, label %if.then192

if.then192:                                       ; preds = %for.cond160
  %inc197 = add i16 %totnumassigns.5, 1
  br label %for.inc248

for.inc248:                                       ; preds = %if.then192, %for.cond160
  %totnumassigns.6 = phi i16 [ %inc197, %if.then192 ], [ %totnumassigns.5, %for.cond160 ]
  br label %for.cond160
}

; CHECK-LABEL: bugpoint_4(
; CHECK:       br
; CHECK:       getelement
; CHECK:       br
; CHECK:       br
; CHECK:       load
; CHECK:       br
; CHECK:       br
; CHECK:       br
; CHECK:       load
; CHECK:       unreachable
; CHECK:       unreachable
; CHECK:       ret
define fastcc void @bugpoint_4() unnamed_addr {

entry:
  %extra_bits = alloca [4 x i16], align 2
  switch i12 undef, label %sw.epilog [
    i12 6, label %sw.bb20
  ]

sw.bb20:                                          ; preds = %entry, %entry, %entry, %entry
  br label %for.cond56.preheader

for.cond56.preheader:                             ; preds = %sw.bb20
  br label %for.body59

while.cond.preheader:                             ; preds = %if.end80
  %arraydecay92 = getelementptr inbounds [4 x i16], [4 x i16]* %extra_bits, i64 0, i64 0
  br i1 undef, label %while.body.lr.ph, label %while.end

while.body.lr.ph:                                 ; preds = %while.cond.preheader
  %0 = load i16, i16* %arraydecay92, align 2
  br label %while.end

for.body59:                                       ; preds = %if.end80, %for.cond56.preheader
  br label %if.end80

if.end80:                                         ; preds = %for.body59
  %arraydecay83 = getelementptr inbounds [4 x i16], [4 x i16]* %extra_bits, i64 0, i64 0
  br i1 undef, label %for.body59, label %while.cond.preheader

while.end:                                        ; preds = %while.body.lr.ph, %while.cond.preheader
  %arraydecay92.lcssa = phi i16* [ %arraydecay92, %while.body.lr.ph ], [ %arraydecay92, %while.cond.preheader ]
  %1 = load i16, i16* %arraydecay92.lcssa, align 2
  unreachable

sw.bb106:                                         ; preds = %entry
  unreachable

sw.epilog:                                        ; preds = %entry
  ret void
}
