; RUN: opt < %s -ssapre -S | FileCheck %s
target datalayout = "e-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

; -------------  -------------      -------------  -------------
;  %5 = %0 + 1    %7 = %0 + 1
; -------------  -------------      -------------  -------------
;          \       /            \\           \       /
;        -------------          //         -------------
;            ret 0                             ret 0
;        -------------                     -------------
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

; -------------  -------------      -------------  -------------
;  %5 = %0 + 1    %7 = %0 + 1        %5 = %0 + 1
;  use  %5       -------------       use  %5       -------------
; -------------     /               -------------     /
;          \       /            \\           \       /
;        -------------          //         -------------
;            ret 0                             ret 0
;        -------------                     -------------
; CHECK-LABEL: @basic_1_used(
; CHECK:       add
; CHECK:       inttoptr
; CHECK:       load
; CHECK:       br
; CHECK-NOT:   add
; CHECK:       br
; CHECK-NOT:   phi
define i64 @basic_1_used(i64, i8**) #0 {
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
; CHECK-LABEL: @basic_2(
; CHECK:       br
; CHECK-NOT:   add
; CHECK:       br
; CHECK-NOT:   add
; CHECK:       br
; CHECK-NOT:   phi
; CHECK:       add
define i32 @basic_2(i32, i8**) #0 {
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
; CHECK-LABEL: @basic_3(
; CHECK:       br
; CHECK-NOT:   add
; CHECK:       br
; CHECK-NOT:   add
; CHECK:       br
; CHECK-NOT:   phi
; CHECK:       add
define i32 @basic_3(i32, i8**) #0 {
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
; CHECK-LABEL: @basic_3_partial_used(
; CHECK:       br
; CHECK:       add
; CHECK:       br
; CHECK:       add
; CHECK:       inttoptr
; CHECK:       load
; CHECK:       br
; CHECK-NOT:   add
; CHECK:       phi
; CHECK:       ret
define i64 @basic_3_partial_used(i64, i8**) #0 {
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
; CHECK-LABEL: @basic_3_partial_not_used(
; CHECK:       br
; CHECK:       br
; CHECK:       add
; CHECK:       inttoptr
; CHECK:       load
; CHECK:       br
; CHECK-NOT:   add
; CHECK-NOT:   phi
; CHECK:       ret
define i64 @basic_3_partial_not_used(i64, i8**) #0 {
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

;           ---------------                        ---------------
;                                                    %1 = %0 + 1
;           ---------------                        ---------------
; .-------------.  |                     .-------------.  |
; |       -------------------            |       -------------------
; |          %p = phi(0,%5)         \\   |
; |       -------------------       //   |       -------------------
; |          /           \               |          /           \
; |  -------------   -------------       |  -------------   -------------
; |   %5 = %0 + 1       ret %p           |                     ret %1
; |  -------------   -------------       |  -------------   -------------
; ._______/                              ._______/
;
; CHECK-LABEL: @cycle_1(
; CHECK:       add
; CHECK:       br
; CHECK:       br
; CHECK:       br
; CHECK:       ret
define i32 @cycle_1(i32, i8**) #0 {
  br label %3

  %.0 = phi i32 [ 0, %2 ], [ %5, %4 ]
  br i1 false, label %4, label %6

  %5 = add nsw i32 %0, 1
  br label %3

  ret i32 %.0
}

;           ---------------                        ---------------
;             %1 = %0 + 1                            %1 = %0 + 1
;           ---------------                        ---------------
; .-------------.  |                     .-------------.  |
; |       -------------------            |       -------------------
; |          %p = phi(0,%5)         \\   |
; |       -------------------       //   |       -------------------
; |          /           \               |          /           \
; |  -------------   -------------       |  -------------   -------------
; |   %5 = %0 + 1       ret %p           |                     ret %1
; |  -------------   -------------       |  -------------   -------------
; ._______/                              ._______/
;
; CHECK-LABEL: @cycle_2(
; CHECK:       add
; CHECK:       br
; CHECK:       br
; CHECK:       br
; CHECK:       ret
define i32 @cycle_2(i32, i8**) #0 {
  %3 = add nsw i32 %0, 1
  br label %4

  %.0 = phi i32 [ %3, %2 ], [ %6, %5 ]
  br i1 false, label %5, label %7

  %6 = add nsw i32 %0, 1
  br label %4

  ret i32 %.0
}
