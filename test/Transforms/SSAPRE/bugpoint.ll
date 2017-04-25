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
