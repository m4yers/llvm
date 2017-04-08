; RUN: opt < %s -ssapre -S | FileCheck %s
target datalayout = "e-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

; CHECK-LABEL: @multi_cycle_1(
; CHECK:       add
; CHECK:       br
; CHECK:       br
; CHECK:       br
; CHECK:       br
; CHECK:       br
; CHECK:       br
; CHECK:       br
; CHECK:       br
; CHECK:       br
; CHECK:       ret
define i32 @multi_cycle_1(i32, i8**) #0 {
  br label %3

  %.0 = phi i32 [ 1, %2 ], [ %.1, %11 ]
  br i1 false, label %4, label %12

  br label %5

  %.1 = phi i32 [ %.0, %4 ], [ %.2, %10 ]
  br i1 false, label %6, label %11

  br label %7

  %.2 = phi i32 [ %.1, %6 ], [ %9, %8 ]
  br i1 false, label %8, label %10

  %9 = add nsw i32 %0, 1
  br label %7

  br label %5

  br label %3

  ret i32 %.0
}
