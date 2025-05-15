subroutine vgen_getgeo

  use vgen_globals
  use expro
  use geo

  implicit none

  integer :: i, j
  real :: r_min
  integer :: n_theta=4
  real, dimension(4) :: theta=(/ 0.0, 1.57079632, 3.14159265, -1.57079632/)

  GEO_signb_in    = EXPRO_signb

  open(unit=1,file='out.vgen.geo',status='replace')
  r_min = EXPRO_rmin(EXPRO_n_exp)

  do i=2,EXPRO_n_exp
     
     ! Parameters to be passed to GEO library   
     !
     ! NOTE: dp/dr set to zero without loss of generality.
     ! 
     GEO_rmin_in      = EXPRO_rmin(i)/r_min
     GEO_rmaj_in      = EXPRO_rmaj(i)/r_min
     GEO_drmaj_in     = EXPRO_drmaj(i)
     GEO_zmag_in      = EXPRO_zmag(i)/r_min
     GEO_dzmag_in     = EXPRO_dzmag(i)
     GEO_q_in         = EXPRO_q(i)
     GEO_s_in         = EXPRO_s(i)
     GEO_kappa_in     = EXPRO_kappa(i)
     GEO_s_kappa_in   = EXPRO_skappa(i)
     GEO_delta_in     = EXPRO_delta(i)
     GEO_s_delta_in   = EXPRO_sdelta(i)
     GEO_zeta_in      = EXPRO_zeta(i)
     GEO_s_zeta_in    = EXPRO_szeta(i)
     GEO_shape_sin3_in    = EXPRO_shape_sin3(i)
     GEO_shape_s_sin3_in  = EXPRO_shape_ssin3(i)
     GEO_shape_sin4_in    = EXPRO_shape_sin4(i)
     GEO_shape_s_sin4_in  = EXPRO_shape_ssin4(i)
     GEO_shape_sin5_in    = EXPRO_shape_sin5(i)
     GEO_shape_s_sin5_in  = EXPRO_shape_ssin5(i)
     GEO_shape_sin6_in    = EXPRO_shape_sin6(i)
     GEO_shape_s_sin6_in  = EXPRO_shape_ssin6(i)
     GEO_shape_cos0_in    = EXPRO_shape_cos0(i)
     GEO_shape_s_cos0_in  = EXPRO_shape_scos0(i)
     GEO_shape_cos1_in    = EXPRO_shape_cos1(i)
     GEO_shape_s_cos1_in  = EXPRO_shape_scos1(i)
     GEO_shape_cos2_in    = EXPRO_shape_cos2(i)
     GEO_shape_s_cos2_in  = EXPRO_shape_scos2(i)
     GEO_shape_cos3_in    = EXPRO_shape_cos3(i)
     GEO_shape_s_cos3_in  = EXPRO_shape_scos3(i)
     GEO_shape_cos4_in    = EXPRO_shape_cos4(i)
     GEO_shape_s_cos4_in  = EXPRO_shape_scos4(i)
     GEO_shape_cos5_in    = EXPRO_shape_cos5(i)
     GEO_shape_s_cos5_in  = EXPRO_shape_scos5(i)
     GEO_shape_cos6_in    = EXPRO_shape_cos6(i)
     GEO_shape_s_cos6_in  = EXPRO_shape_scos6(i)
     GEO_beta_star_in = 0.0
     GEO_model_in = 0       ! Call GEO with model shape

     call geo_interp(n_theta,theta,.true.)
     
     write(1,'(e16.8)',advance='no') EXPRO_rho(i)
     do j=1,n_theta
        write(1,'(e16.8)',advance='no') theta(j)
        write(1,'(e16.8)',advance='no') GEO_b(j)  * EXPRO_bunit(i)
        write(1,'(e16.8)',advance='no') GEO_bp(j) * EXPRO_bunit(i)
        write(1,'(e16.8)',advance='no') GEO_bt(j) * EXPRO_bunit(i)
        write(1,'(e16.8)',advance='no') GEO_bigr(j) * r_min
     enddo
     write(1,*)
  enddo

  ! Clean up
  close(1)

end subroutine vgen_getgeo
