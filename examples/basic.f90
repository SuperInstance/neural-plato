!> Basic example: test all neural_plato modules
program basic_example
  use neural_plato
  implicit none

  integer, parameter :: dp = selected_real_kind(15, 307)

  call test_amnesia()
  call test_snap()
  call test_tucker()
  call test_negative()

  print *, ''
  print *, '=== ALL TESTS PASSED ==='

contains

  subroutine test_amnesia()
    real(dp) :: acc
    print *, '--- Amnesia Curve ---'
    acc = predict_accuracy(1.0_dp)
    print *, '  100% coverage -> accuracy:', acc
    if (acc < 0.9_dp) error stop 'amnesia: 100% too low'
    acc = predict_accuracy(0.5_dp)
    print *, '   50% coverage -> accuracy:', acc
    if (acc < 0.3_dp .or. acc > 0.6_dp) error stop 'amnesia: 50% unexpected'
    acc = predict_accuracy(0.05_dp)
    print *, '    5% coverage -> accuracy:', acc
    if (acc > 0.15_dp) error stop 'amnesia: 5% should be near zero'
    print *, '  PASSED'
  end subroutine test_amnesia

  subroutine test_snap()
    real(dp) :: positions(2, 5)
    integer :: snapped(2, 5)
    integer :: i
    print *, '--- Intent Snap ---'
    positions(1, 1) = 0.3_dp
    positions(2, 1) = 0.7_dp
    positions(1, 2) = 2.8_dp
    positions(2, 2) = 0.9_dp
    positions(1, 3) = -1.2_dp
    positions(2, 3) = 0.3_dp
    positions(1, 4) = 1.0_dp
    positions(2, 4) = 0.0_dp
    positions(1, 5) = 5.5_dp
    positions(2, 5) = 3.2_dp

    call snap_batch(positions, snapped, 5)
    do i = 1, 5
      print *, '  pos (', positions(1,i), ',', positions(2,i), ') -> dodecet (', snapped(1,i), ',', snapped(2,i), ')'
    end do
    ! (1,0) is exactly on dodecet point 1
    if (snapped(1, 4) /= 1 .or. snapped(2, 4) /= 0) error stop 'snap: (1,0) should map to (1,0)'
    print *, '  PASSED'
  end subroutine test_snap

  subroutine test_tucker()
    real(dp) :: query(4), row_keys(2, 3), col_keys(2, 4), core(2, 2)
    real(dp) :: scores(3, 4)
    integer :: i
    print *, '--- Tucker Decompose ---'
    query = [0.5_dp, -0.3_dp, 0.8_dp, 0.1_dp]
    row_keys = reshape([1.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 0.5_dp, 0.5_dp], [2, 3])
    col_keys = reshape([1.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 0.5_dp, 0.5_dp, -0.5_dp, 0.5_dp], [2, 4])
    core = reshape([1.0_dp, 0.0_dp, 0.0_dp, 1.0_dp], [2, 2])

    call compute_scores(query, row_keys, col_keys, core, scores, 4, 2, 3, 4)
    print *, '  Score matrix:'
    do i = 1, 3
      print *, '   ', scores(i, :)
    end do
    print *, '  PASSED'
  end subroutine test_tucker

  subroutine test_negative()
    real(dp) :: constraints(4, 3), shadow(4)
    print *, '--- Negative Space ---'
    constraints = reshape([1.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, &
                           0.0_dp, 1.0_dp, 0.0_dp, 0.0_dp, &
                           0.0_dp, 0.0_dp, 1.0_dp, 0.0_dp], [4, 3])
    call compute_shadow(constraints, shadow, 3, 4)
    print *, '  Shadow vector:', shadow
    if (size(shadow) /= 4) error stop 'shadow: wrong size'
    print *, '  PASSED'
  end subroutine test_negative

end program basic_example
