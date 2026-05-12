!> Vectorized Eisenstein lattice snap for constraint theory.
module intent_snap
  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  ! The 12 Eisenstein integers (dodecet system)
  integer, parameter :: DODECET_Q(12) = [1, 2, 2, 1, -1, -2, -2, -1, 0, 1, 1, 0]
  integer, parameter :: DODECET_R(12) = [0, 1, 2, 2,  2,  1,  0, -1, -1, -1, 0, 0]

  real(dp), parameter :: DODECET_Q_DP(12) = [1.0_dp, 2.0_dp, 2.0_dp, 1.0_dp, &
                                              -1.0_dp, -2.0_dp, -2.0_dp, -1.0_dp, &
                                               0.0_dp, 1.0_dp, 1.0_dp, 0.0_dp]
  real(dp), parameter :: DODECET_R_DP(12) = [0.0_dp, 1.0_dp, 2.0_dp, 2.0_dp, &
                                              2.0_dp, 1.0_dp, 0.0_dp, -1.0_dp, &
                                             -1.0_dp, -1.0_dp, 0.0_dp, 0.0_dp]

contains

  subroutine snap_batch(positions, snapped, n)
    integer, intent(in)  :: n
    real(dp), intent(in)  :: positions(2, n)
    integer, intent(out)  :: snapped(2, n)

    integer :: i, best_idx

    do i = 1, n
      best_idx = snap_single(positions(1, i), positions(2, i))
      snapped(1, i) = DODECET_Q(best_idx)
      snapped(2, i) = DODECET_R(best_idx)
    end do

  end subroutine snap_batch

  subroutine constraint_distance(positions, distances, n)
    integer, intent(in)  :: n
    real(dp), intent(in)  :: positions(2, n)
    real(dp), intent(out) :: distances(n)

    integer :: i, j, best_idx
    real(dp) :: dq, dr, dist_sq, best_dist

    do i = 1, n
      best_dist = huge(1.0_dp)
      best_idx = 1
      do j = 1, 12
        dq = positions(1, i) - DODECET_Q_DP(j)
        dr = positions(2, i) - DODECET_R_DP(j)
        dist_sq = dq*dq + dr*dr + dq*dr
        if (dist_sq < best_dist) then
          best_dist = dist_sq
          best_idx = j
        end if
      end do
      distances(i) = sqrt(best_dist)
    end do

  end subroutine constraint_distance

  function snap_single(q, r) result(snapped_idx)
    real(dp), intent(in) :: q, r
    integer :: snapped_idx

    integer :: j, best_idx
    real(dp) :: dq, dr, dist_sq, best_dist

    best_dist = huge(1.0_dp)
    best_idx = 1

    do j = 1, 12
      dq = q - DODECET_Q_DP(j)
      dr = r - DODECET_R_DP(j)
      dist_sq = dq*dq + dr*dr + dq*dr
      if (dist_sq < best_dist) then
        best_dist = dist_sq
        best_idx = j
      end if
    end do

    snapped_idx = best_idx

  end function snap_single

  subroutine snap_full_lattice(positions, snapped, n)
    integer, intent(in)  :: n
    real(dp), intent(in)  :: positions(2, n)
    integer, intent(out)  :: snapped(2, n)

    integer :: i, j, best_c
    real(dp) :: q, r, dq, dr, dist_sq, best_d
    real(dp) :: candidates_q(4), candidates_r(4)

    do i = 1, n
      q = positions(1, i)
      r = positions(2, i)

      candidates_q(1) = floor(q)
      candidates_q(2) = floor(q)
      candidates_q(3) = ceiling(q)
      candidates_q(4) = ceiling(q)

      candidates_r(1) = floor(r)
      candidates_r(2) = ceiling(r)
      candidates_r(3) = floor(r)
      candidates_r(4) = ceiling(r)

      best_d = huge(1.0_dp)
      best_c = 1

      do j = 1, 4
        dq = q - candidates_q(j)
        dr = r - candidates_r(j)
        dist_sq = dq*dq + dr*dr + dq*dr
        if (dist_sq < best_d) then
          best_d = dist_sq
          best_c = j
        end if
      end do

      snapped(1, i) = nint(candidates_q(best_c))
      snapped(2, i) = nint(candidates_r(best_c))
    end do

  end subroutine snap_full_lattice

end module intent_snap
