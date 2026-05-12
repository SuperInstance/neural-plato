!> Sparse memory layer inspired by UltraMem (ICLR 2025, ByteDance Seed team).
module sparse_memory
  implicit none
  integer, parameter :: dp = selected_real_kind(15, 307)

  type :: MemoryLayer
    real(dp), allocatable :: values(:,:,:)     ! (n_rows, n_cols, hidden_dim)
    real(dp), allocatable :: row_keys(:,:)     ! (n_ranks, n_rows)
    real(dp), allocatable :: col_keys(:,:)     ! (n_ranks, n_cols)
    real(dp), allocatable :: tucker_core(:,:)  ! (n_ranks, n_ranks)
    real(dp), allocatable :: linear_w(:,:)     ! (hidden_dim, 4) for IVE
    integer :: n_rows, n_cols, hidden_dim, n_ranks, top_k
  end type

contains

  subroutine init_memory(layer, n_rows, n_cols, hidden_dim, n_ranks, top_k)
    type(MemoryLayer), intent(inout) :: layer
    integer, intent(in) :: n_rows, n_cols, hidden_dim, n_ranks, top_k

    real(dp) :: scale

    layer%n_rows = n_rows
    layer%n_cols = n_cols
    layer%hidden_dim = hidden_dim
    layer%n_ranks = n_ranks
    layer%top_k = top_k

    allocate(layer%values(n_rows, n_cols, hidden_dim))
    allocate(layer%row_keys(n_ranks, n_rows))
    allocate(layer%col_keys(n_ranks, n_cols))
    allocate(layer%tucker_core(n_ranks, n_ranks))
    allocate(layer%linear_w(hidden_dim, 4))

    scale = sqrt(2.0_dp / real(n_ranks + hidden_dim, dp))

    call random_number(layer%values)
    layer%values = (layer%values - 0.5_dp) * 2.0_dp * scale

    call random_number(layer%row_keys)
    layer%row_keys = (layer%row_keys - 0.5_dp) * 2.0_dp * scale

    call random_number(layer%col_keys)
    layer%col_keys = (layer%col_keys - 0.5_dp) * 2.0_dp * scale

    call random_number(layer%tucker_core)
    layer%tucker_core = (layer%tucker_core - 0.5_dp) * 2.0_dp * scale

    call random_number(layer%linear_w)
    layer%linear_w = (layer%linear_w - 0.5_dp) * 2.0_dp * scale

  end subroutine init_memory

  subroutine query(layer, query_vec, output_vec, n_query)
    type(MemoryLayer), intent(in) :: layer
    integer, intent(in) :: n_query
    real(dp), intent(in)  :: query_vec(layer%hidden_dim, n_query)
    real(dp), intent(out) :: output_vec(layer%hidden_dim, n_query)

    real(dp), allocatable :: scores(:,:), q_proj(:,:), core_out(:,:)
    real(dp), allocatable :: top_scores(:)
    integer, allocatable  :: top_indices(:)
    integer :: q, k, r, c, idx, total_cells
    real(dp) :: total_score

    total_cells = layer%n_rows * layer%n_cols
    allocate(scores(layer%n_rows, layer%n_cols))
    allocate(q_proj(layer%n_ranks, n_query))
    allocate(core_out(layer%n_ranks, n_query))
    allocate(top_indices(layer%top_k))
    allocate(top_scores(layer%top_k))

    output_vec = 0.0_dp

    do q = 1, n_query
      ! Project query to rank space
      q_proj(:,q) = matmul(layer%row_keys, query_vec(:,q))

      ! Apply Tucker core
      core_out(:,q) = matmul(layer%tucker_core, q_proj(:,q))

      ! Compute full score matrix
      call compute_all_scores(layer, query_vec(1,q), scores)

      ! Find top-k (flatten 2D -> 1D indices)
      call find_top_k_2d(scores, layer%n_rows, layer%n_cols, &
                         top_indices, top_scores, layer%top_k)

      ! Weighted sum
      total_score = sum(abs(top_scores))
      if (total_score < 1.0e-10_dp) total_score = 1.0_dp

      do k = 1, layer%top_k
        idx = top_indices(k)
        r = (idx - 1) / layer%n_cols + 1
        c = mod(idx - 1, layer%n_cols) + 1
        output_vec(:,q) = output_vec(:,q) + &
          (abs(top_scores(k)) / total_score) * layer%values(r, c, :)
      end do
    end do

    deallocate(scores, q_proj, core_out, top_indices, top_scores)

  end subroutine query

  subroutine tucker_scores(layer, query_vec, scores, n_scores)
    type(MemoryLayer), intent(in) :: layer
    integer, intent(in) :: n_scores
    real(dp), intent(in)  :: query_vec(layer%hidden_dim)
    real(dp), intent(out) :: scores(n_scores)

    real(dp) :: q_rank(layer%n_ranks), c_rank(layer%n_ranks)
    real(dp) :: col_proj(layer%n_cols)
    integer :: r, c, idx

    ! Project to rank space
    do r = 1, layer%n_ranks
      q_rank(r) = dot_product(layer%row_keys(r, :), query_vec)
    end do

    ! Apply Tucker core
    c_rank = matmul(layer%tucker_core, q_rank)

    ! Column projections
    do c = 1, layer%n_cols
      col_proj(c) = dot_product(layer%col_keys(:, c), c_rank)
    end do

    ! Assemble flattened scores
    idx = 1
    do r = 1, layer%n_rows
      do c = 1, layer%n_cols
        scores(idx) = q_rank(min(r, layer%n_ranks)) * col_proj(c)
        idx = idx + 1
      end do
    end do

  end subroutine tucker_scores

  subroutine top_k_indices(scores, indices, n_scores, k)
    integer, intent(in) :: n_scores, k
    real(dp), intent(in)  :: scores(n_scores)
    integer, intent(out)  :: indices(k)

    logical :: used(n_scores)
    integer :: i, j, best_j
    real(dp) :: best_score

    used = .false.

    do i = 1, k
      best_score = -huge(1.0_dp)
      best_j = 1
      do j = 1, n_scores
        if (.not. used(j) .and. scores(j) > best_score) then
          best_score = scores(j)
          best_j = j
        end if
      end do
      indices(i) = best_j
      used(best_j) = .true.
    end do

  end subroutine top_k_indices

  subroutine implicit_expand(values_in, linear_weights, values_out, n_v, n_h, n_expand)
    integer, intent(in) :: n_v, n_h, n_expand
    real(dp), intent(in)  :: values_in(n_v, n_h)
    real(dp), intent(in)  :: linear_weights(n_h, n_expand)
    real(dp), intent(out) :: values_out(n_v, n_h * n_expand)

    integer :: e

    ! Original values in first slot
    values_out(:, 1:n_h) = values_in

    ! Expanded: linear projections (broadcast scalar weight across each row)
    do e = 1, n_expand - 1
      block
        integer :: row_i
        do row_i = 1, n_v
          values_out(row_i, e*n_h+1:(e+1)*n_h) = values_in(row_i, :) * sum(linear_weights(:, e)) / n_h
        end do
      end block
    end do

  end subroutine implicit_expand

  ! ---- Internal helpers ----

  subroutine compute_all_scores(layer, query_vec, scores)
    type(MemoryLayer), intent(in) :: layer
    real(dp), intent(in)  :: query_vec(layer%hidden_dim)
    real(dp), intent(out) :: scores(layer%n_rows, layer%n_cols)

    real(dp) :: q_rank(layer%n_ranks), c_rank(layer%n_ranks)
    real(dp) :: col_proj(layer%n_cols)
    real(dp) :: row_proj(layer%n_rows)
    integer :: r, c

    ! Project query to rank space
    do r = 1, layer%n_ranks
      q_rank(r) = dot_product(layer%row_keys(r, :), query_vec)
    end do

    ! Tucker core
    c_rank = matmul(layer%tucker_core, q_rank)

    ! Column projections
    do c = 1, layer%n_cols
      col_proj(c) = dot_product(layer%col_keys(:, c), c_rank)
    end do

    ! Row projections (map query to row space)
    do r = 1, layer%n_rows
      row_proj(r) = dot_product(layer%row_keys(:, r), query_vec)
    end do

    ! Outer product scores
    do r = 1, layer%n_rows
      do c = 1, layer%n_cols
        scores(r, c) = row_proj(r) * col_proj(c)
      end do
    end do

  end subroutine compute_all_scores

  subroutine find_top_k_2d(scores, n_rows, n_cols, indices, top_scores, k)
    integer, intent(in) :: n_rows, n_cols, k
    real(dp), intent(in)  :: scores(n_rows, n_cols)
    integer, intent(out)  :: indices(k)
    real(dp), intent(out) :: top_scores(k)

    real(dp) :: flat(n_rows * n_cols)
    integer :: used(n_rows * n_cols)
    integer :: i, j, total, best_j
    real(dp) :: best_val

    total = n_rows * n_cols

    ! Flatten
    do i = 1, n_rows
      do j = 1, n_cols
        flat((i-1)*n_cols + j) = scores(i, j)
      end do
    end do

    used = 0

    do i = 1, k
      best_val = -huge(1.0_dp)
      best_j = 1
      do j = 1, total
        if (used(j) == 0 .and. flat(j) > best_val) then
          best_val = flat(j)
          best_j = j
        end if
      end do
      indices(i) = best_j
      top_scores(i) = best_val
      used(best_j) = 1
    end do

  end subroutine find_top_k_2d

end module sparse_memory
