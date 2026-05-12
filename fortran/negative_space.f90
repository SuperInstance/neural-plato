!> Shadow reconstruction from negative constraints.
!>
!> Given a set of "negative" facts (things that are known to be false/absent),
!> reconstruct the "shadow" — the implied structure of what must be true.
!>
!> This is the computational dual of direct reconstruction: instead of
!> building from what you have, you carve away what you know isn't there.
module negative_space
  implicit none
  integer, parameter, private :: dp = selected_real_kind(15, 307)

contains

  !> Compute shadow vector from negative constraints.
  !>
  !> Each constraint defines a hyperplane in n_dims-dimensional space.
  !> The shadow is the projection of the origin onto the feasible region
  !> defined by the intersection of half-spaces.
  !>
  !> Uses iterative projection (Dykstra's algorithm simplified).
  subroutine compute_shadow(constraints, shadow, n_constraints, n_dims)
    integer, intent(in)  :: n_constraints, n_dims
    real(dp), intent(in)  :: constraints(n_constraints, n_dims) !! Constraint normals
    real(dp), intent(out) :: shadow(n_dims)                     !! Shadow vector

    real(dp), allocatable :: x(:), correction(:), old_x(:)
    real(dp) :: dot_val, norm_sq, step_size
    integer :: i, iter, max_iter

    max_iter = 100
    step_size = 1.0_dp / real(n_constraints, dp)

    allocate(x(n_dims))
    allocate(correction(n_dims))
    allocate(old_x(n_dims))

    ! Initialize: uniform distribution
    x = 1.0_dp / sqrt(real(n_dims, dp))
    correction = 0.0_dp

    ! Iterative projection onto each constraint's feasible half-space
    do iter = 1, max_iter
      old_x = x

      do i = 1, n_constraints
        ! Project x onto the half-space defined by constraint i
        ! constraint: dot(constraint, x) >= 0  (feasible direction)
        dot_val = dot_product(constraints(i, :), x + correction)
        norm_sq = dot_product(constraints(i, :), constraints(i, :))

        if (norm_sq < 1.0e-12_dp) cycle

        if (dot_val < 0.0_dp) then
          ! x is on the wrong side; project back
          x = x + correction
          correction = -(dot_val / norm_sq) * constraints(i, :)
          x = x + correction
          correction = 0.0_dp
        else
          correction = x + correction - x
        end if
      end do

      ! Normalize shadow to unit length
      norm_sq = dot_product(x, x)
      if (norm_sq > 1.0e-12_dp) then
        x = x / sqrt(norm_sq)
      end if

      ! Check convergence
      if (dot_product(x - old_x, x - old_x) < 1.0e-10_dp) exit
    end do

    shadow = x

    deallocate(x, correction, old_x)

  end subroutine compute_shadow

  !> Predict shadow reconstruction accuracy.
  !>
  !> Accuracy improves with more negative facts (up to a saturation point).
  !> Modeled as: accuracy = 1 - exp(-alpha * density)
  !> where density = n_negative / n_total
  function shadow_accuracy(n_negative_facts, n_total_facts) result(accuracy)
    integer, intent(in) :: n_negative_facts, n_total_facts
    real(dp) :: accuracy

    real(dp) :: density, alpha

    if (n_total_facts <= 0) then
      accuracy = 0.0_dp
      return
    end if

    density = real(n_negative_facts, dp) / real(n_total_facts, dp)

    ! Saturation parameter: 50% negative facts gives ~90% accuracy
    alpha = 4.6_dp  ! ln(100) ≈ 4.6

    accuracy = 1.0_dp - exp(-alpha * density)

    ! Cap at theoretical maximum (~97.5%)
    accuracy = min(accuracy, 0.975_dp)

  end function shadow_accuracy

  !> Compute the information gain from adding one more negative constraint.
  !> Returns marginal accuracy improvement.
  function marginal_gain(n_negative, n_total) result(gain)
    integer, intent(in) :: n_negative, n_total
    real(dp) :: gain

    gain = shadow_accuracy(n_negative + 1, n_total) - &
           shadow_accuracy(n_negative, n_total)

  end function marginal_gain

end module negative_space
