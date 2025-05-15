!-----------------------------------------------------
! gyro_alloc_profile_exp.f90
!
! PURPOSE:
!  Allocate experimental-grid profile functions 
!  (all _exp and _p) variables IF they are not 
!  already allocated.
!-----------------------------------------------------

subroutine gyro_alloc_profile_exp

  use gyro_globals
  use gyro_profile_exp

  implicit none

  if (.not.allocated(rhogrid_exp)) then

     allocate(rhogrid_exp(n_grid_exp))
     allocate(rmin_exp(n_grid_exp))
     allocate(rmaj_exp(n_grid_exp))
     allocate(q_exp(n_grid_exp))
     allocate(kappa_exp(n_grid_exp))
     allocate(delta_exp(n_grid_exp))
     allocate(zeta_exp(n_grid_exp))

     allocate(tem_exp(n_spec,n_grid_exp))
     allocate(den_exp(n_spec,n_grid_exp))
 
     allocate(w0_exp(n_grid_exp))
     allocate(w0p_exp(n_grid_exp))
     allocate(gamma_e_exp(n_grid_exp))
     allocate(gamma_p_exp(n_grid_exp))
     allocate(mach_exp(n_grid_exp))
     allocate(z_eff_exp(n_grid_exp))
     allocate(zmag_exp(n_grid_exp))
     allocate(ptot_exp(n_grid_exp))

     allocate(r_p(n_grid_exp))
     allocate(b_unit_p(n_grid_exp))
     allocate(rhosda_p(n_grid_exp))
     allocate(csda_p(n_grid_exp))
     allocate(drmaj_p(n_grid_exp))
     allocate(dzmag_p(n_grid_exp))
     allocate(s_delta_p(n_grid_exp))
     allocate(s_zeta_p(n_grid_exp))
     allocate(s_kappa_p(n_grid_exp))
     allocate(q_p(n_grid_exp))
     allocate(shat_p(n_grid_exp))

     allocate(dlnndr_p(n_spec,n_grid_exp))
     allocate(dlntdr_p(n_spec,n_grid_exp))
     allocate(dlnptotdr_p(n_grid_exp))

     allocate(beta_unit_p(n_grid_exp))
     allocate(beta_unit_ptot_p(n_grid_exp))
  
  endif

end subroutine gyro_alloc_profile_exp
