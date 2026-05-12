!> neural_plato — Main module re-exporting all submodules.
!>
!> Use: use neural_plato → brings in all modules.
module neural_plato
  use sparse_memory
  use amnesia_curve
  use intent_snap
  use tucker_decompose
  use negative_space
  implicit none

  character(len=*), parameter :: VERSION = "0.1.0"

  public :: VERSION

end module neural_plato
