; RUN: opt < %s -ssapre -S | FileCheck %s
target datalayout = "e-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

; CHECK-LABEL: complex_1(
declare zeroext i8 @c_tolower(i8 zeroext) #1
define i32 @complex_1(i8*, i8*) #0 {

; CHECK:      icmp
; CHECK-NEXT: br
  %3 = icmp eq i8* %0, %1
  br i1 %3, label %4, label %5

; CHECK: br
  br label %25

; CHECK: br
  br label %6

; CHECK: br
  %.02 = phi i8* [ %0, %5 ], [ %15, %17 ]
  %.01 = phi i8* [ %1, %5 ], [ %16, %17 ]
  %7 = load i8, i8* %.02, align 1
  %8 = call zeroext i8 @c_tolower(i8 zeroext %7)
  %9 = load i8, i8* %.01, align 1
  %10 = call zeroext i8 @c_tolower(i8 zeroext %9)
  %11 = zext i8 %8 to i32
  %12 = icmp eq i32 %11, 0
  br i1 %12, label %13, label %14

; CHECK:      zext i8 %10 to i32
; CHECK-NEXT: br
  br label %21

; CHECK:      getelementptr inbounds i8, i8* %.02, i32 1
; CHECK-NEXT: getelementptr inbounds i8, i8* %.01, i32 1
; CHECK-NEXT: br
  %15 = getelementptr inbounds i8, i8* %.02, i32 1
  %16 = getelementptr inbounds i8, i8* %.01, i32 1
  br label %17

; CHECK:      zext i8 %10 to i32
; CHECK-NEXT: icmp
; CHECK-NEXT: br
  %18 = zext i8 %8 to i32
  %19 = zext i8 %10 to i32
  %20 = icmp eq i32 %18, %19
  br i1 %20, label %6, label %21

; CHECK:      phi i32 [ %19, %18 ], [ %14, %13 ]
; CHECK-NEXT: sub nsw i32 %11, %22
; CHECK-NEXT: br
  %22 = zext i8 %8 to i32
  %23 = zext i8 %10 to i32
  %24 = sub nsw i32 %22, %23
  br label %25

; CHECK:      phi i32 [ 0, %4 ], [ %23, %21 ]
; CHECK-NEXT: ret
  %.0 = phi i32 [ 0, %4 ], [ %24, %21 ]
  ret i32 %.0
}
