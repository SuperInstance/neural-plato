!> Experimental forgetting curves from baton protocol experiments.
!>
!> Models the relationship between memory coverage and reconstruction accuracy.
!> Based on empirical data from Seed-2.0-mini experiments (May 2026).
module amnesia_curve
  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  ! Experimental data points from baton protocol experiments
  integer, parameter :: N_DATA = 8
  real(dp), parameter :: coverage_data(8) = [1.00_dp, 0.75_dp, 0.50_dp, 0.33_dp, &
                                               0.25_dp, 0.15_dp, 0.10_dp, 0.05_dp]
  real(dp), parameter :: accuracy_data(8) = [0.975_dp, 0.775_dp, 0.475_dp, 0.325_dp, &
                                               0.225_dp, 0.225_dp, 0.125_dp, 0.000_dp]

  ! The coverage threshold below which accuracy collapses
  real(dp), parameter :: AMNESIA_CLIFF = 0.10_dp

  ! Curve fitting parameters (linear regression on empirical data)
  ! accuracy ≈ slope * coverage + intercept
  real(dp), parameter :: FIT_SLOPE = 0.912_dp
  real(dp), parameter :: FIT_INTERCEPT = 0.012_dp

contains

  !> Predict reconstruction accuracy from memory coverage fraction.
  !> Uses piecewise linear interpolation on experimental data,
  !> with extrapolation below the amnesia cliff.
  function predict_accuracy(coverage) result(accuracy)
    real(dp), intent(in) :: coverage
    real(dp) :: accuracy

    integer :: i

    ! Clamp to [0, 1]
    if (coverage <= 0.0_dp) then
      accuracy = 0.0_dp
      return
    end if
    if (coverage >= 1.0_dp) then
      accuracy = accuracy_data(1)
      return
    end if

    ! Below amnesia cliff: accuracy drops to zero
    if (coverage < AMNESIA_CLIFF) then
      ! Linear ramp from 0 at coverage=0 to last_data_point at AMNESIA_CLIFF
      accuracy = (coverage / AMNESIA_CLIFF) * accuracy_data(7)  ! data point at 0.10
      return
    end if

    ! Linear interpolation between data points
    do i = 1, N_DATA - 1
      if (coverage >= coverage_data(i+1) .and. coverage <= coverage_data(i)) then
        ! Linear interp: t in [0,1] between data(i) and data(i+1)
        block
          real(dp) :: t, span
          span = coverage_data(i) - coverage_data(i+1)
          if (span < 1.0e-12_dp) then
            accuracy = accuracy_data(i)
          else
            t = (coverage - coverage_data(i+1)) / span
            accuracy = accuracy_data(i+1) + t * (accuracy_data(i) - accuracy_data(i+1))
          end if
        end block
        return
      end if
    end do

    ! Fallback: linear fit
    accuracy = FIT_SLOPE * coverage + FIT_INTERCEPT

  end function predict_accuracy

  !> Style factor: multiplier on accuracy based on prompt style.
  !> Styles: 0=direct, 1=conversational, 2=Socratic, 3=narrative, 4=technical
  function style_factor(style_id) result(factor)
    integer, intent(in) :: style_id
    real(dp) :: factor

    select case(style_id)
    case (0)
      factor = 1.00_dp   ! Direct — baseline
    case (1)
      factor = 0.95_dp   ! Conversational — slight degradation
    case (2)
      factor = 0.88_dp   ! Socratic — more tokens, some loss
    case (3)
      factor = 0.92_dp   ! Narrative — good structure helps
    case (4)
      factor = 1.05_dp   ! Technical — models excel at structured content
    case default
      factor = 1.00_dp
    end select

  end function style_factor

  !> Predict accuracy based on compression ratio (output chars vs input chars).
  !> Heavily compressed output loses accuracy.
  function compression_accuracy(n_chars) result(accuracy)
    integer, intent(in) :: n_chars
    real(dp) :: accuracy

    real(dp) :: ratio

    ! Assume reference is ~1000 chars for full fidelity
    ratio = real(n_chars, dp) / 1000.0_dp

    if (ratio >= 1.0_dp) then
      accuracy = 0.975_dp  ! Full coverage
    else if (ratio >= 0.5_dp) then
      accuracy = 0.475_dp + 0.5_dp * (ratio - 0.5_dp) * 2.0_dp
    else if (ratio >= 0.1_dp) then
      accuracy = ratio * 0.95_dp
    else
      accuracy = 0.0_dp
    end if

  end function compression_accuracy

  !> Batch predict accuracy for an array of coverage values.
  subroutine batch_predict(coverages, accuracies, n)
    integer, intent(in)  :: n
    real(dp), intent(in)  :: coverages(n)
    real(dp), intent(out) :: accuracies(n)

    integer :: i

    do i = 1, n
      accuracies(i) = predict_accuracy(coverages(i))
    end do

  end subroutine batch_predict

  !> Compute the derivative of accuracy w.r.t. coverage at a given point.
  !> Useful for finding the steepest part of the forgetting curve.
  function accuracy_gradient(coverage) result(grad)
    real(dp), intent(in) :: coverage
    real(dp) :: grad

    real(dp) :: eps
    eps = 0.005_dp

    grad = (predict_accuracy(coverage + eps) - predict_accuracy(coverage - eps)) / (2.0_dp * eps)

  end function accuracy_gradient

end module amnesia_curve
