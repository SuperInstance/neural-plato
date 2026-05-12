!> Basic example demonstrating neural-plato Fortran modules.
program basic_example
  use neural_plato
  implicit none

  type(MemoryLayer) :: mem
  real(dp), allocatable :: qvec(:,:), ovec(:,:)
  real(dp) :: acc, cov
  real(dp) :: positions(2, 5), snapped_f(2, 5)
  integer :: snapped(2, 5), i
  real(dp) :: constraints(3, 4), shadow_vec(4)
  real(dp) :: covs(6), accs(6)

  print *, '=== neural-plato v' // trim(VERSION) // ' ==='
  print *, ''

  ! --- Sparse Memory ---
  print *, '--- Sparse Memory Layer ---'
  call init_memory(mem, n_rows=8, n_cols=8, hidden_dim=16, n_ranks=4, top_k=4)

  allocate(qvec(16, 1))
  allocate(ovec(16, 1))
  call random_number(qvec)
  qvec = qvec * 2.0_dp - 1.0_dp

  call query(mem, qvec, ovec, 1)
  print *, 'Query output norm:', sqrt(dot_product(ovec(:,1), ovec(:,1)))
  print *, ''

  ! --- Amnesia Curve ---
  print *, '--- Amnesia Curve ---'
  do i = 0, 10
    cov = real(i, dp) / 10.0_dp
    acc = predict_accuracy(cov)
    write(*, '(A,F5.2,A,F6.3)') '  Coverage ', cov, ' -> Accuracy ', acc
  end do
  print *, '  Amnesia cliff at coverage < ', AMNESIA_CLIFF
  print *, ''

  ! --- Intent Snap ---
  print *, '--- Eisenstein Snap ---'
  positions(:,1) = [1.2_dp, 0.3_dp]
  positions(:,2) = [-0.5_dp, 1.8_dp]
  positions(:,3) = [0.0_dp, 0.0_dp]
  positions(:,4) = [2.1_dp, 1.9_dp]
  positions(:,5) = [-1.7_dp, -0.2_dp]

  call snap_batch(positions, snapped, 5)
  do i = 1, 5
    write(*, '(A,F5.2,F6.2,A,I3,I3)') '  Snap ', positions(1,i), positions(2,i), &
                                       ' -> ', snapped(1,i), snapped(2,i)
  end do
  print *, ''

  ! --- Negative Space ---
  print *, '--- Negative Space ---'
  constraints(1,:) = [0.5_dp, -0.3_dp, 0.8_dp, 0.1_dp]
  constraints(2,:) = [-0.2_dp, 0.7_dp, -0.4_dp, 0.6_dp]
  constraints(3,:) = [0.9_dp, 0.1_dp, 0.3_dp, -0.5_dp]
  call compute_shadow(constraints, shadow_vec, 3, 4)
  print *, '  Shadow vector:', shadow_vec
  print *, '  Shadow accuracy (5/20):', shadow_accuracy(5, 20)
  print *, ''

  ! --- Batch predict ---
  print *, '--- Batch Amnesia Prediction ---'
  covs = [1.0_dp, 0.75_dp, 0.50_dp, 0.25_dp, 0.10_dp, 0.05_dp]
  call batch_predict(covs, accs, 6)
  do i = 1, 6
    write(*, '(A,F5.2,A,F6.3)') '  ', covs(i), ' -> ', accs(i)
  end do

  print *, ''
  print *, '=== Done ==='

  deallocate(qvec, ovec)

end program basic_example
