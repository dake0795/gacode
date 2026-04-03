!---------------------------------------------------------------------------
! cgyro_stress.F90
!
! PURPOSE:
!  Compute the per-field nonlinear bracket {h, <chi_i>} as a function of
!  (kx, theta, species, ky, field), where field i = phi, Apar, Bpar.
!
!  After velocity integration this gives the per-field contribution to
!  the zonal stress kernel, analogous to stella's calstress diagnostic.
!
!  Enable with STRESS_PRINT_FLAG=1 (nonlinear runs only).
!  Output written to bin.cgyro.stress_phi, bin.cgyro.stress_apar,
!  bin.cgyro.stress_bpar via cgyro_write_timedata.
!---------------------------------------------------------------------------

subroutine cgyro_stress

  use mpi
  use cgyro_globals
  use cgyro_nl_comm
  use cgyro_nl

  implicit none

  integer :: i_field

  do i_field=1, n_field
     ! Pack h_x into fpackA/B and send first half asynchronously
     call cgyro_nl_fftw_comm1_async_stress

     ! Pack field i_field only into gpack (zeroing other slots), send asynchronously
     call cgyro_nl_fftw_comm2_async_stress(i_field)

     ! Send second half of h_x (fpackB) if velocity space is split
     call cgyro_nl_fftw_comm3_async

     ! Compute nonlinear bracket
     call cgyro_nl_fftw
     call cgyro_nl_fftw_comm_test

     ! Store bracket result into stress(:,:,:,i_field)
     call cgyro_nl_fftw_comm1_r_stress(i_field)
     call cgyro_nl_fftw_comm_test
  end do

#if defined(OMPGPU)
!$omp target update from(stress)
#elif defined(_OPENACC)
!$acc update host(stress)
#endif

end subroutine cgyro_stress
