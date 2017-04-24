; RUN: opt < %s -ssapre -S | FileCheck %s
target datalayout = "e-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"


; CHECK-LABEL: @complex_1(
; CHECK-NOT:   getelementptr
define void @complex_1(i8** %argv) #0 {
  br label %1

; <label>:1:                                      ; preds = %10, %6, %0
  br i1 undef, label %2, label %11

; <label>:2:                                      ; preds = %1
  br i1 undef, label %3, label %7

; <label>:3:                                      ; preds = %2
  br label %4

; <label>:4:                                      ; preds = %5, %3
  br i1 undef, label %5, label %6

; <label>:5:                                      ; preds = %4
  br label %4

; <label>:6:                                      ; preds = %4
  br label %1

; <label>:7:                                      ; preds = %2
  br label %8

; <label>:8:                                      ; preds = %9, %7
  br i1 undef, label %9, label %10

; <label>:9:                                      ; preds = %8
  %arrayidx35 = getelementptr inbounds i8*, i8** %argv, i64 undef
  br label %8

; <label>:10:                                     ; preds = %8
  br label %1

; <label>:11:                                     ; preds = %1
  ret void
}
