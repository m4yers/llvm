; RUN: opt < %s -ssapre -S | FileCheck %s
target datalayout = "e-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

; -------------  -------------      -------------  -------------
;  %5 = %0 + 1    %7 = %0 + 1
; -------------  -------------      -------------  -------------
;          \       /            \\           \       /
;        -------------          //         -------------
;            ret 0                             ret 0
;        -------------                     -------------
; CHECK-LABEL: @join_1(
; CHECK-NOT:   add
; CHECK-NOT:   add
; CHECK-NOT:   phi
; CHECK-NOT:   add
define i32 @join_1(i32, i8**) #0 {
  %3 = icmp ne i32 %0, 0
  br i1 %3, label %4, label %6

  %5 = add nsw i32 %0, 1
  br label %8

  %7 = add nsw i32 %0, 1
  br label %8

  ret i32 0
}

; -------------  -------------      -------------  -------------
;  %5 = %0 + 1    %7 = %0 + 1        %5 = %0 + 1
;  use  %5       -------------       use  %5       -------------
; -------------     /               -------------     /
;          \       /            \\           \       /
;        -------------          //         -------------
;            ret 0                             ret 0
;        -------------                     -------------
; CHECK-LABEL: @join_2(
; CHECK:       add
; CHECK:       inttoptr
; CHECK:       load
; CHECK:       br
; CHECK-NOT:   add
; CHECK:       br
; CHECK-NOT:   phi
; CHECK-NOT:   add
define i64 @join_2(i64, i8**) #0 {
  %3 = icmp ne i64 %0, 0
  br i1 %3, label %4, label %6

  %5 = add nsw i64 %0, 1
  %ptr = inttoptr i64 %5 to i64*
  %val = load i64, i64* %ptr
  br label %8

  %7 = add nsw i64 %0, 1
  br label %8

  ret i64 0
}

; TODO Here we need to recognize that %5 and %7 calculate the same value though
; they are versioned differently
; -------------  -------------      -------------  -------------
;  %5 = %0 + 1    %7 = %0 + 1        %5 = %0 + 1    %7 = %0 + 1
; -------------  -------------      -------------  -------------
;          \       /            \\           \       /
;      -----------------        //       -----------------
;       %p = phi(%5,%7)                   %p = phi(%5,%7)
;       ret %p                            ret %p
;      -----------------                 -----------------
; CHECK-LABEL: @join_3(
; CHECK:       br
; CHECK-NOT:   add
; CHECK:       br
; CHECK-NOT:   add
; CHECK:       br
; CHECK-NOT:   phi
; CHECK:       add
define i32 @join_3(i32, i8**) #0 {
  %3 = icmp ne i32 %0, 0
  br i1 %3, label %4, label %6

  %5 = add nsw i32 %0, 1
  br label %8

  %7 = add nsw i32 %0, 1
  br label %8

  %p = phi i32 [ %5, %4 ], [ %7, %6 ]
  ret i32 %p
}

; -------------  -------------      -------------  -------------
;                 %6 = %0 + 1
; -------------  -------------      -------------  -------------
;          \       /            \\           \       /
;        -------------          //         -------------
;         %8 = %0 + 1                       %8 = %0 + 1
;         ret %8                            ret %8
;        -------------                     -------------
; CHECK-LABEL: @join_4(
; CHECK:       br
; CHECK-NOT:   add
; CHECK:       br
; CHECK-NOT:   add
; CHECK:       br
; CHECK-NOT:   phi
; CHECK:       add
; CHECK:       ret
define i32 @join_4(i32, i8**) #0 {
  %3 = icmp ne i32 %0, 0
  br i1 %3, label %4, label %5

  br label %7

  %6 = add nsw i32 %0, 1
  br label %7

  %8 = add nsw i32 %0, 1
  ret i32 %8
}

; -------------  -------------      -------------  -------------
;                 %6 = %0 + 1        %5 = %0 + 1    %6 = %0 + 1
; -------------   use %6            -------------   use %6
;         \      -------------              \      -------------
;          \       /            \\           \       /
;        -------------          //       -----------------
;         %8 = %0 + 1                     %p = %phi(%5,%6)
;         ret %8                          ret %p
;        -------------                   -----------------
; CHECK-LABEL: @join_5(
; CHECK:       br
; CHECK:       add
; CHECK:       br
; CHECK:       add
; CHECK:       inttoptr
; CHECK:       load
; CHECK:       br
; CHECK:       phi
; CHECK-NOT:   add
; CHECK:       ret
define i64 @join_5(i64, i8**) #0 {
  %3 = icmp ne i64 %0, 0
  br i1 %3, label %4, label %5

  br label %7

  %6 = add nsw i64 %0, 1
  %ptr = inttoptr i64 %6 to i64*
  %val = load i64, i64* %ptr
  br label %7

  %8 = add nsw i64 %0, 1
  ret i64 %8
}

; -------------  -------------      -------------  -------------
;                 %6 = %0 + 1                       %6 = %0 + 1
; -------------   use %6            -------------   use %6
;         \      -------------              \      -------------
;          \       /            \\           \       /
;        -------------          //       -----------------
;         %8 = %0 + 1                     ret 0
;         ret 0                          -----------------
;        -------------
; CHECK-LABEL: @join_6(
; CHECK:       br
; CHECK-NOT:   add
; CHECK:       br
; CHECK:       add
; CHECK:       inttoptr
; CHECK:       load
; CHECK:       br
; CHECK-NOT:   phi
; CHECK-NOT:   add
; CHECK:       ret
define i64 @join_6(i64, i8**) #0 {
  %3 = icmp ne i64 %0, 0
  br i1 %3, label %4, label %5

  br label %7

  %6 = add nsw i64 %0, 1
  %ptr = inttoptr i64 %6 to i64*
  %val = load i64, i64* %ptr
  br label %7

  %8 = add nsw i64 %0, 1
  ret i64 0
}

; Nothing must change
;
; CHECK-LABEL: @join_7(
; CHECK:       icmp
; CHECK:       br
; CHECK:       icmp
; CHECK:       br
; CHECK:       icmp
; CHECK:       br
; CHECK:       icmp
; CHECK:       br
; CHECK:       phi
; CHECK:       xor
; CHECK:       phi
; CHECK:       zext
; CHECK:       ret
define i32 @join_7(i32) #0 {
  %2 = icmp sge i32 %0, 33
  br i1 %2, label %3, label %12

  %4 = icmp sle i32 %0, 126
  br i1 %4, label %5, label %12

  %6 = icmp sge i32 %0, 48
  br i1 %6, label %7, label %9

  %8 = icmp sle i32 %0, 57
  br label %9

  %10 = phi i1 [ false, %5 ], [ %8, %7 ]
  %11 = xor i1 %10, true
  br label %12

  %13 = phi i1 [ false, %3 ], [ false, %1 ], [ %11, %9 ]
  %14 = zext i1 %13 to i32
  ret i32 %14
}
