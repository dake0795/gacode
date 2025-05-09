!---------------------------------------------------------------------------
! cgyro_flux.f90
!
! PURPOSE:
!  Compute flux and mean-square fluctuation amplitudes as
!  functions of (kx,ky).
!
! NOTES:
!
!  Flux definitions (these are all broken down by ky) 
!
!   gflux(l,ky) = Radial Fourier expansion of flux (l is the Fourier harmonic)
!   gflux(0,ky) = Domain-averaged flux [this is the standard flux-tube flux]
!     cflux(ky) = Interior-averaged flux [this is the "central" flux]
!
!  Global flux logic:
!
!   Retain radial harmonics l = [0,...,N_GLOBAL], where N_GLOBAL=4 by default
!
!   To trigger output of the global flux (viewable by -plot xflux),
!    set GFLUX_PRINT_FLAG=1
!
! NOTE:
!  cflux is only of interest when profile or ExB shear is active
!  cflux(ky) is the average flux in the "positive-shear region"
!  gflux(ky,0)-cflux is the flux in the "negative-shear region"
!---------------------------------------------------------------------------

subroutine cgyro_flux

  use mpi
  use cgyro_globals

  implicit none

  integer :: ie,ix,is,it,ir,i_field,itor
  integer :: l,icl
  real :: dv,cn
  real :: vpar
  complex, dimension(0:n_global,n_field) :: prod1,prod2,prod3
  real :: dvr
  real :: erot
  real :: flux_norm
  complex :: cprod
  real, parameter :: x_fraction=0.2
  real :: u
  
!$omp parallel do private(iv_loc,iv,is,ix,ie,dv,vpar,ic,ir,it,erot,cprod,cn) &
!$omp&            private(prod1,prod2,prod3,l,icl,dvr,u,flux_norm) &
!$omp&            shared(moment_loc,gflux_loc,cflux_loc)
  do itor=nt1,nt2

     !-----------------------------------------------------
     ! 1. Compute kx-ky moments (n,E,v)
     !-----------------------------------------------------

     moment_loc(:,:,:,itor,:) = 0.0

     iv_loc = 0
     do iv=nv1,nv2

        iv_loc = iv_loc+1

        is = is_v(iv)
        ix = ix_v(iv)
        ie = ie_v(iv)

        ! Integration weight
        dv = w_exi(ie,ix)

        ! Parallel velocity
        vpar = vth(is)*vel2(ie)*xi(ix)

        do ic=1,nc

           ir = ir_c(ic)
           it = it_c(ic)

           erot  = (energy(ie)+lambda_rot(it,is))*temp(is)

           if (itp(it) > 0) then
              cprod = cap_h_c(ic,iv_loc,itor)*dvjvec_c(1,ic,iv_loc,itor)/z(is)
              cn    = dv*z(is)*dens2_rot(it,is)/temp(is)

              ! Note addition division by rho below
              
              ! Density moment: (delta n_a)/(n_norm)
              moment_loc(ir,itp(it),is,itor,1) = moment_loc(ir,itp(it),is,itor,1)-(cn*field(1,ic,itor)-cprod)

              ! Energy moment : (delta E_a)/(n_norm T_norm)
              moment_loc(ir,itp(it),is,itor,2) = moment_loc(ir,itp(it),is,itor,2)-(cn*field(1,ic,itor)-cprod)*erot

              ! Velocity moment : (delta v_a)/(n_norm v_norm)
              moment_loc(ir,itp(it),is,itor,3) = moment_loc(ir,itp(it),is,itor,3)-(cn*field(1,ic,itor)-cprod)*vpar
           endif

        enddo
     enddo

     ! Unlike moments, fields are *not* divided by rho
     ! (cgyro_plot will adjust so both have same normalization)
     moment_loc(:,:,:,itor,:) = moment_loc(:,:,:,itor,:)/rho

     !-------------------------------------------------------------
     ! 2. Compute global ky-dependent fluxes (with field breakdown)
     !-------------------------------------------------------------

     gflux_loc(:,:,:,:,itor) = 0.0
     if (triad_print_flag == 1) then
        triad_loc_old(:,:,itor,6) = 0.0
     endif
     iv_loc = 0
     do iv=nv1,nv2

        iv_loc = iv_loc+1

        is = is_v(iv)
        ix = ix_v(iv)
        ie = ie_v(iv)

        ! Integration weight
        dv = w_exi(ie,ix)

        ! Parallel velocity
        vpar = vth(is)*vel2(ie)*xi(ix)

        do ic=1,nc

           ir = ir_c(ic)
           it = it_c(ic)

           ! prod* are local to this loop
           prod1 = 0.0 
           prod2 = 0.0
           prod3 = 0.0

           ! Global flux coefficients (complex coefficients required to compute radial profile)
           do l=0,n_global

              ! H w^* + H^* w

              if (ir-l > 0) then
                 icl = ic_c(ir-l,it)
                 prod1(l,:) = prod1(l,:) &
                      +i_c*cap_h_c(ic,iv_loc,itor)*conjg(jvec_c(:,icl,iv_loc,itor)*field(:,icl,itor))
                 prod2(l,:) = prod2(l,:) &
                      +i_c*cap_h_c(ic,iv_loc,itor)*conjg(i_c*jxvec_c(:,icl,iv_loc,itor)*field(:,icl,itor))
                 prod3(l,:) = prod3(l,:) &
                      -cap_h_c_dot(ic,iv_loc,itor)*conjg(jvec_c(:,icl,iv_loc,itor)*field(:,icl,itor)) &
                      +cap_h_c(ic,iv_loc,itor)*conjg(jvec_c(:,icl,iv_loc,itor)*field_dot(:,icl,itor))
              endif
              if (ir+l <= n_radial) then
                 icl = ic_c(ir+l,it)
                 prod1(l,:) = prod1(l,:) &
                      -i_c*conjg(cap_h_c(ic,iv_loc,itor))*jvec_c(:,icl,iv_loc,itor)*field(:,icl,itor)
                 prod2(l,:) = prod2(l,:) &
                      -i_c*conjg(cap_h_c(ic,iv_loc,itor))*i_c*jxvec_c(:,icl,iv_loc,itor)*field(:,icl,itor)
                 prod3(l,:) = prod3(l,:) &
                      -conjg(cap_h_c_dot(ic,iv_loc,itor))*jvec_c(:,icl,iv_loc,itor)*field(:,icl,itor) &
                      +conjg(cap_h_c(ic,iv_loc,itor))*jvec_c(:,icl,iv_loc,itor)*field_dot(:,icl,itor)
              endif

           enddo

           dvr  = w_theta(it)*dens2_rot(it,is)*dv
           erot = (energy(ie)+lambda_rot(it,is))*temp(is)

           ! 1. Density flux: Gamma_a
           gflux_loc(:,is,1,:,itor) = gflux_loc(:,is,1,:,itor)+prod1(:,:)*dvr

           ! 2. Energy flux : Q_a
           gflux_loc(:,is,2,:,itor) = gflux_loc(:,is,2,:,itor)+prod1(:,:)*dvr*erot

           prod1(:,:) = prod1(:,:)*(mach*bigr(it)/rmaj+btor(it)/bmag(it)*vpar)+prod2(:,:)

           ! 3. Momentum flux: Pi_a
           gflux_loc(:,is,3,:,itor) = gflux_loc(:,is,3,:,itor)+prod1(:,:)*dvr*bigr(it)*mass(is)

           ! 4. Exchange
           gflux_loc(:,is,4,:,itor) = gflux_loc(:,is,4,:,itor)+0.5*prod3(:,:)*dvr*z(is)

           if (triad_print_flag == 1) then
              ! Field potential remaining term. dPsi/dt H^* 
               cprod =+3*sum(jvec_c(:,ic,iv_loc,itor)*field(:,ic,itor)      ) &
                      -4*sum(jvec_c(:,ic,iv_loc,itor)*field_old2(:,ic,itor) ) &
                      +1*sum(jvec_c(:,ic,iv_loc,itor)*field_old3(:,ic,itor) )
               cprod = cprod*conjg(cap_h_c(ic,iv_loc,itor))*(z(is)/temp(is))*dvr 
               !$omp critical
               triad_loc_old(is,ir,itor,6) = triad_loc_old(is,ir,itor,6) + cprod
               !$omp end critical
           endif
        enddo

     enddo

     ! W_E/(n_e T_e) = 1/2 < [k_perp^2 lam_D^2 + sum (e^2 n/T) (1-Gamma_0) ] |e phi/Te|^2 >
     ! W_M/(n_e T_e) = < k_perp^2 rho_D^2 |(cs/c) e A_par/Te|^2 > / beta
     if (triad_print_flag == 1) then
        do ic=1,nc
           ir = ir_c(ic)
           it = it_c(ic)
           triad_loc(:,ir,itor,9) = 0.5*k_perp(ic,itor)**2*lambda_debye**2*dens_ele/temp_ele &
                * w_theta(it) * abs(field(1,ic,itor))**2
           triad_loc(:,ir,itor,10) = 0.5*sum_den_x(ic,itor) &
                * w_theta(it) * abs(field(1,ic,itor))**2
           if(n_field > 1) then
              triad_loc(:,ir,itor,11) = k_perp(ic,itor)**2 * rho**2/betae_unit*dens_ele*temp_ele &
                   * w_theta(it) * abs(field(2,ic,itor))**2
           else
              triad_loc(:,ir,itor,11) = 0.0
           endif
        enddo
     endif

     ! Construct "positive/interior" flux (real quantities)
     cflux_loc(:,:,:,itor) = real(gflux_loc(0,:,:,:,itor))
     do l=1,n_global
        u = 2*pi*l*x_fraction
        cflux_loc(:,:,:,itor) = cflux_loc(:,:,:,itor)+2*sin(u)*real(gflux_loc(l,:,:,:,itor))/u
     enddo


  !-------------------------------------------------------------
  ! 2-1. Compute Triad energy transfer
  !-------------------------------------------------------------

     if (triad_print_flag == 1) then
      do is=1,n_species
        ! Triad energy transfer : T_k
        ! triad_loc(is,:,itor,1)
        ! Nonzonal Triad energy transfer(ky!=0) : T_k^*  , Triad energy transfer to ZF(ky=0) : N
        ! triad_loc(is,:,itor,2)
        ! dEntropy / dt   :   dS_k/dt 
        ! triad_loc(is,:,itor,3)
        ! dEM field /dt   :   Wk_perp/dt  
          triad_loc(is,:,itor,4) = -triad_loc(is,:,itor,4) - triad_loc_old(is,:,itor,6)/(2*delta_t)
        ! Entropy         :   S_k
        ! triad_loc(is,:,itor,5)
        ! Diss. (radial)  :  D_r
        ! triad_loc(is,:,itor,6)
        ! Diss. (theta )  :  D_theta
        ! triad_loc(is,:,itor,7)
        ! Diss. (Coll. )  :  D_coll
        ! triad_loc(is,:,itor,8)
        ! W_E (lam_D)
        ! triad_loc(is,:,itor,9)
        ! W_E (polarization)  
        ! triad_loc(is,:,itor,10)
        ! W_M  
        ! triad_loc(is,:,itor,11)  

          triad_loc(is,:,itor,:) = triad_loc(is,:,itor,:) *temp(is)/dlntdr(is_ele)
      enddo
     endif


     !-----------------------------------------------------
     ! 3. Renormalize fluxes to GB or quasilinear forms
     !~----------------------------------------------------

     if (nonlinear_flag == 0 .and. itor > 0) then

        ! Quasilinear normalization (divide by |phi|^2)
        ! Note: We assume we compute flux_norm once per itor
        flux_norm = 0.0
        do ir=1,n_radial
           flux_norm = flux_norm+sum(abs(field(1,ic_c(ir,:),itor))**2*w_theta(:))
        enddo

        ! Correct for sign of q
        flux_norm = flux_norm*q/abs(q)*2 ! need 2 for regression compatibility

        gflux_loc(:,:,:,:,itor) = gflux_loc(:,:,:,:,itor)/flux_norm 
        cflux_loc (:,:,:,itor)  = cflux_loc(:,:,:,itor)/flux_norm 

     else

        ! Complete definition of fluxes (not exchange)
        gflux_loc(:,:,1:3,:,itor) = gflux_loc(:,:,1:3,:,itor)*(k_theta_base*itor*rho)
        cflux_loc(:,1:3,:,itor)   = cflux_loc(:,1:3,:,itor)*(k_theta_base*itor*rho)

        ! GyroBohm normalizations
        gflux_loc(:,:,:,:,itor) = gflux_loc(:,:,:,:,itor)/rho**2
        cflux_loc(:,:,:,itor) = cflux_loc(:,:,:,itor)/rho**2
        if (triad_print_flag == 1) then
          triad_loc(:,:,itor,:)  = triad_loc(:,:,itor,:)/rho**2
        endif
     endif
  enddo

  ! Reduced complex moment(kx,ky), below, is still distributed over n 

  call MPI_ALLREDUCE(moment_loc(:,:,:,:,:), &
       moment(:,:,:,:,:), &
       size(moment), &
       MPI_DOUBLE_COMPLEX, &
       MPI_SUM, &
       NEW_COMM_1, &
       i_err)

  ! Global complex gflux(l,ky), below, is still distributed over n 

  call MPI_ALLREDUCE(gflux_loc, &
       gflux, &
       size(gflux), &
       MPI_DOUBLE_COMPLEX, &
       MPI_SUM, &
       NEW_COMM_1, &
       i_err)

  ! Reduced real cflux(ky), below, is still distributed over n 

  call MPI_ALLREDUCE(cflux_loc, &
       cflux, &
       size(cflux), &
       MPI_DOUBLE_PRECISION, &
       MPI_SUM, &
       NEW_COMM_1, &
       i_err)

  if (triad_print_flag == 1) then
   ! Reduced complex triad(ns,kx), below, is still distributed over n 
   call MPI_ALLREDUCE(triad_loc(:,:,:,:), &
       triad, &
       size(triad), &
       MPI_DOUBLE_COMPLEX, &
       MPI_SUM, &
       NEW_COMM_1, &
       i_err)
  endif


  tave_step = tave_step + 1
  tave_max  = t_current
  do itor=nt1,nt2
     do i_field=1,n_field
        cflux_tave(:,:) = cflux_tave(:,:) + cflux(:,:,i_field,itor)
     enddo
     do i_field=1,n_field
        gflux_tave(:,:) = gflux_tave(:,:) + real(gflux(0,:,:,i_field,itor))
     enddo
  enddo
     
end subroutine cgyro_flux
