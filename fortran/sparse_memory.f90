!> Sparse memory layer inspired by UltraMem (ICLR 2025, ByteDance Seed team).
module sparse_memory
  implicit none
  integer, parameter, private :: dp = selected_real_kind(15, 307)

  type :: MemoryLayer
    real(dp), allocatable :: values(:,:,:)     ! (n_rows, n_cols, hidden_dim)
    real(dp), allocatable :: row_keys(:,:)     ! (n_ranks, hidden_dim) - projection
    real(dp), allocatable :: row_embed(:,:)    ! (n_ranks, n_rows) - row embeddings
    real(dp), allocatable :: col_keys(:,:)     ! (n_ranks, hidden_dim) - projection
    real(dp), allocatable :: col_embed(:,:)    ! (n_ranks, n_cols) - col embeddings
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
    allocate(layer%row_keys(n_ranks, hidden_dim))
    allocate(layer%row_embed(n_ranks, n_rows))
    allocate(layer%col_keys(n_ranks, hidden_dim))
    allocate(layer%col_embed(n_ranks, n_cols))
    allocate(layer%tucker_core(n_ranks, n_ranks))
    allocate(layer%linear_w(hidden_dim, 4))

    scale = sqrt(2.0_dp / real(n_ranks + hidden_dim, dp))

    call random_number(layer%values)
    layer%values = (layer%values - 0.5_dp) * 2.0_dp * scale

    call random_number(layer%row_keys)
    layer%row_keys = (layer%row_keys - 0.5_dp) * 2.0_dp * scale

    call random_number(layer%row_embed)
    layer%row_embed = (layer%row_embed - 0.5_dp) * 2.0_dp * scale

    call random_number(layer%col_keys)
    layer%col_keys = (layer%col_keys - 0.5_dp) * 2.0_dp * scale

    call random_number(layer%col_embed)
    layer%col_embed = (layer%col_embed - 0.5_dp) * 2.0_dp * scale

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

    real(dp), allocatable :: scores(:,:)
    real(dp), allocatable :: top_scores(:)
    integer, allocatable  :: top_indices(:)
    integer :: q, k, r, c, idx
    real(dp) :: total_score

    allocate(scores(layer%n_rows, layer%n_cols))
    allocate(top_indices(layer%top_k))
    allocate(top_scores(layer%top_k))

    output_vec = 0.0_dp

    do q = 1, n_query
      ! Compute full score matrix via TDQKR
      call compute_all_scores(layer, query_vec(:,q), scores)

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

    deallocate(scores, top_indices, top_scores)

  end subroutine query

  subroutine tucker_scores(layer, query_vec, scores, n_scores)
    type(MemoryLayer), intent(in) :: layer
    integer, intent(in) :: n_scores
    real(dp), intent(in)  :: query_vec(layer%hidden_dim)
    real(dp), intent(out) :: scores(n_scores)

    real(dp) :: q_rank(layer%n_ranks), c_rank(layer%n_ranks)
    real(dp) :: row_proj(layer%n_rows), col_proj(layer%n_cols)
    integer :: r, c, idx

    ! Project query to rank space: q_rank = row_keys * query
    q_rank = matmul(layer%row_keys, query_vec)

    ! Apply Tucker core
    c_rank = matmul(layer%tucker_core, q_rank)

    ! Row scores: row_proj(r) = dot(row_embed(:,r), q_rank)
    do r = 1, layer%n_rows
      row_proj(r) = dot_product(layer%row_embed(:, r), q_rank)
    end do

    ! Column scores: col_proj(c) = dot(col_embed(:,c), c_rank)
    do c = 1, layer%n_cols
      col_proj(c) = dot_product(layer%col_embed(:, c), c_rank)
    end do

    ! Assemble flattened scores as outer product
    idx = 1
    do r = 1, layer%n_rows
      do c = 1, layer%n_cols
        scores(idx) = row_proj(r) * col_proj(c)
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

    ! Expanded: scaled linear projections
    do e = 1, n_expand - 1
      block
        real(dp) :: w_sum
        integer :: row_i
        w_sum = sum(linear_weights(:, e)) / real(n_h, dp)
        do row_i = 1, n_v
          values_out(row_i, e*n_h+1:(e+1)*n_h) = values_in(row_i, :) * w_sum
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
    real(dp) :: row_proj(layer%n_rows), col_proj(layer%n_cols)
    integer :: r, c

    ! Step 1: Project query to rank space via row_keys
    ! q_rank = row_keys * query  [n_ranks x 1]
    q_rank = matmul(layer%row_keys, query_vec)

    ! Step 2: Apply Tucker core decomposition
    ! c_rank = tucker_core * q_rank  [n_ranks x 1]
    c_rank = matmul(layer%tucker_core, q_rank)

    ! Step 3: Compute row scores via row embeddings
    do r = 1, layer%n_rows
      row_proj(r) = dot_product(layer%row_embed(:, r), q_rank)
    end do

    ! Step 4: Compute column scores via column embeddings
    do c = 1, layer%n_cols
      col_proj(c) = dot_product(layer%col_embed(:, c), c_rank)
    end do

    ! Step 5: Assemble score matrix via outer product
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
