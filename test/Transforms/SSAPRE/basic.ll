; RUN: opt < %s -ssapre -S | FileCheck %s
target datalayout = "e-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

; CHECK-LABEL: @basic_1(
; CHECK-NOT:   add
; CHECK-NOT:   phi
define i32 @basic_1(i32, i8**) #0 {
  %3 = icmp ne i32 %0, 0
  br i1 %3, label %4, label %6

  %5 = add nsw i32 %0, 1
  br label %8

  %7 = add nsw i32 %0, 1
  br label %8

  ret i32 0
}

; CHECK-LABEL: @basic_2(
; CHECK:       add
; CHECK:       add
; CHECK:       phi
define i32 @basic_2(i32, i8**) #0 {
  %3 = icmp ne i32 %0, 0
  br i1 %3, label %4, label %5

  br label %7

  %6 = add nsw i32 %0, 1
  br label %7

  %8 = add nsw i32 %0, 1
  ret i32 %8
}
