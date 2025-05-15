!------------------------------------------------------------------------------
! cgyro_error_estimate.f90
!
! PURPOSE:
!  Compute two time-integration error estimates based on 
!  (1) difference between computed field and quadratic extrapolation, 
!  (2) 3rd-order estimate of collisionless error based on RK algebra
!------------------------------------------------------------------------------

subroutine cgyro_error_estimate

  use mpi
  use cgyro_globals
  use cgyro_io
  use timer_lib

  implicit none

  integer :: i_f,itor
  real, dimension(2) :: norm_loc,norm
  real, dimension(2) :: pair_loc,pair
  real, dimension(2) :: error_loc

  real :: norm_loc_s,error_loc_s,h_s,r_s

#if (!defined(OMPGPU)) && defined(_OPENACC)
  ! launch Estimate of collisionless error via 3rd-order linear estimate async ahead of time on GPU/ACC
  ! CPU-only and OMPGPU code will work on it later
  ! NOTE: If I have multiple itor, sum them all together
  h_s=0.0
  r_s=0.0
!$acc parallel loop collapse(3) independent gang vector &
!$acc&         present(h_x,rhs(:,:,:,1)) reduction(+:h_s,r_s) async(2)
  do itor=nt1,nt2
   do iv_loc=1,nv_loc
     do ic=1,nc
        h_s = h_s + abs(h_x(ic,iv_loc,itor))
        r_s = r_s + abs(rhs(ic,iv_loc,itor,1))
     enddo
   enddo
  enddo
#endif

  call timer_lib_in('field')

  norm_loc_s = 0.0
  error_loc_s = 0.0

  ! field_olds are always only in system memory... too expensive to keep in GPU memory
  ! assuming field was already synched to system memory
!$omp parallel do collapse(3) reduction(+:norm_loc_s,error_loc_s)
  do itor=nt1,nt2
   do ic=1,nc
     do i_f=1,n_field

        ! 1. Estimate of total (field) error via quadratic interpolation

        field_loc(i_f,ic,itor) = 3*field_old(i_f,ic,itor) - &
                3*field_old2(i_f,ic,itor) + &
                field_old3(i_f,ic,itor)
        field_dot(i_f,ic,itor) = (3*field(i_f,ic,itor) - &
                4*field_old(i_f,ic,itor) + &
                field_old2(i_f,ic,itor) )/(2*delta_t)

        ! Define norm and error for each mode number n
        norm_loc_s  = norm_loc_s  + abs(field(i_f,ic,itor))
        error_loc_s = error_loc_s + abs(field(i_f,ic,itor)-field_loc(i_f,ic,itor))

        ! save old values for next iteration
        field_old3(i_f,ic,itor) = field_old2(i_f,ic,itor)
        field_old2(i_f,ic,itor) = field_old(i_f,ic,itor)
        field_old(i_f,ic,itor)  = field(i_f,ic,itor)
     enddo
   enddo
  enddo

  norm_loc(1)  = norm_loc_s
  error_loc(1) = error_loc_s

#if defined(OMPGPU)
!$omp target teams distribute parallel do simd collapse(3) &
!$omp&   private(iv_loc)
#elif defined(_OPENACC)
!$acc parallel loop collapse(3) gang vector private(iv_loc) &
!$acc&         present(cap_h_c_dot,cap_h_c,cap_h_c_old,cap_h_c_old2) &
!$acc&         present(nt1,nt2,nv1,nv2,nc) copyin(delta_t) default(none)
#else
!$omp parallel do collapse(3) private(iv_loc)
#endif
  do itor=nt1,nt2
   do iv=nv1,nv2
     do ic=1,nc
        iv_loc = iv-nv1+1
        cap_h_c_dot(ic,iv_loc,itor) = (3*cap_h_c(ic,iv_loc,itor) - &
                4*cap_h_c_old(ic,iv_loc,itor) + &
                cap_h_c_old2(ic,iv_loc,itor) )/(2*delta_t)
        cap_h_c_old2(ic,iv_loc,itor) = cap_h_c_old(ic,iv_loc,itor)
        cap_h_c_old(ic,iv_loc,itor) = cap_h_c(ic,iv_loc,itor)
     enddo
   enddo
  enddo

  call timer_lib_out('field')

  ! 2. Estimate of collisionless error via 3rd-order linear estimate

  call timer_lib_in('str')

#if (!defined(OMPGPU)) && defined(_OPENACC)
  ! wait for the async GPU compute to be completed
!$acc wait(2)
#else
  ! NOTE: If I have multiple itor, sum them all together
  h_s=0.0
  r_s=0.0
#if defined(OMPGPU)
  ! no async for OMPG{U for now
!$omp target teams distribute parallel do simd collapse(3) &
!$omp&    firstprivate(nt1,nt2,nv_loc,nc) &
!$omp&    reduction(+:h_s,r_s)
#else
!$omp parallel do collapse(3) reduction(+:h_s,r_s) &
!$omp&    firstprivate(nt1,nt2,nv_loc,nc)
#endif
  do itor=nt1,nt2
   do iv_loc=1,nv_loc
     do ic=1,nc
        h_s = h_s + abs(h_x(ic,iv_loc,itor))
        r_s = r_s + abs(rhs(ic,iv_loc,itor,1))
     enddo
   enddo
  enddo

#endif

  pair_loc(1) = h_s
  pair_loc(2) = r_s

  call timer_lib_out('str')

  call timer_lib_in('str_comm')

  ! sum over velocity space
  call MPI_ALLREDUCE(pair_loc,&
       pair,&
       2,&
       MPI_DOUBLE_PRECISION,&
       MPI_SUM,&
       NEW_COMM_1,&
       i_err)

  norm_loc(2) = pair(1)
  error_loc(2) = pair(2)

  ! Get sum of all errors
  call MPI_ALLREDUCE(error_loc, &
       integration_error, &
       2, &
       MPI_DOUBLE_PRECISION, &
       MPI_SUM, &
       NEW_COMM_2, &
       i_err)

  ! Get sum of all norms
  call MPI_ALLREDUCE(norm_loc, &
       norm, &
       2, &
       MPI_DOUBLE_PRECISION, &
       MPI_SUM, &
       NEW_COMM_2, &
       i_err)

  call timer_lib_out('str_comm')

  ! 1: total (field) error
  ! 2: collisionless error
  integration_error = integration_error/norm

  ! Trigger code shutdown on large collisionless error
  if (integration_error(2) > 5e2*error_tol .and. i_time > 2) then
     call cgyro_error('Integration error exceeded limit.')
  endif

end subroutine cgyro_error_estimate


!------------------------------------------------------------------------------
! TEMPORARY home for triad measurement(diagnostics)

subroutine cgyro_triad_estimate(i_f)
  use cgyro_globals
  implicit none

  !-----------------------------------
  integer, intent(in) :: i_f
  !-----------------------------------

  if (triad_exec_flag == 1) then
    call cgyro_triad_diagnostics
  else
    call cgyro_triad_setup
  endif

  if (i_f == 0) then
    triad_exec_flag = 1
  else
    triad_exec_flag = 0
  endif
end subroutine cgyro_triad_estimate


subroutine cgyro_triad_setup
  use timer_lib
  use cgyro_globals

  implicit none

  integer :: is,ix,ie
  integer :: ir,it,iv_loc_m,ic_loc_m,itor

  real :: dv,dvr
  complex :: cprod,cprod2

#if defined(OMPGPU)
!$omp target teams distribute parallel do simd collapse(3)
#elif defined(_OPENACC)
!$acc parallel loop collapse(3) gang vector independent &
!$acc&         present(triad_loc_old) &
!$acc&         present(nt1,nt2,n_radial,n_species) default(none)
#else
!$omp parallel do private(ir,is)
#endif
  do itor=nt1,nt2
      do ir=1,n_radial
        do is=1,n_species
          triad_loc_old(is,ir,itor,5) = triad_loc_old(is,ir,itor,3)
          triad_loc_old(is,ir,itor,3) = triad_loc_old(is,ir,itor,1)
          triad_loc_old(is,ir,itor,1) = 0.0

          triad_loc_old(is,ir,itor,6) = triad_loc_old(is,ir,itor,4)
          triad_loc_old(is,ir,itor,4) = triad_loc_old(is,ir,itor,2)
          triad_loc_old(is,ir,itor,2) = 0.0
        enddo
      enddo
  enddo


#if defined(OMPGPU)
!$omp target teams distribute parallel do simd collapse(2) &
!$omp&         private(iv_loc,it,ic_loc_m) &
!$omp&         private(is,ix,ie,dv,dvr,cprod,cprod2) &
!$omp&         firstprivate(nt1,nt2,n_radial,nv_loc,n_theta,nv1)
#elif defined(_OPENACC)
!$acc parallel loop gang vector collapse(2) default(present) &
!$acc&         private(iv_loc,it,ic_loc_m) &
!$acc&         private(is,ix,ie,dv,dvr,cprod,cprod2) &
!$acc&         firstprivate(nt1,nt2,n_radial,nv_loc,n_theta,nv1)
#else
!$omp parallel do collapse(2) &
!$omp&         private(iv_loc,it,ic_loc_m) &
!$omp&         private(is,ix,ie,dv,dvr,cprod,cprod2) &
!$omp&         firstprivate(nt1,nt2,n_radial,nv_loc,n_theta,nv1)
#endif
  do itor=nt1,nt2
    do ir=1,n_radial
      ! Avoiding memory race: inner loop must remain sequential (only collapse 2) - J
      do iv_loc_m=1,nv_loc
        do it=1,n_theta
   
           ic_loc_m = ic_c(ir,it)

           is = is_v(iv_loc_m +nv1 -1 )
           ix = ix_v(iv_loc_m +nv1 -1 )
           ie = ie_v(iv_loc_m +nv1 -1 )
           dv = w_exi(ie,ix)
           dvr  = w_theta(it)*dens2_rot(it,is)*dv

           ! Density moment
           cprod = w_theta(it)*cap_h_c(ic_loc_m,iv_loc_m,itor)*dvjvec_c(1,ic_loc_m,iv_loc_m,itor)/z(is)
           cprod = -(dvr*z(is)/temp(is)*field(1,ic_loc_m,itor)-cprod)

           cprod = - field(1,ic_loc_m,itor)*conjg(field(1,ic_loc_m,itor))*(z(is)/temp(is))**2*dvr &
                - 2.0*cprod*conjg(field(1,ic_loc_m,itor))*(z(is)/temp(is))
           cprod2  = cap_h_c(ic_loc_m,iv_loc_m,itor)*conjg(cap_h_c(ic_loc_m,iv_loc_m,itor))*dvr + cprod
           ! . Entropy(df) - Entropy(dH)
           triad_loc_old(is,ir,itor,1) = triad_loc_old(is,ir,itor,1)  + cprod
           ! . Entropy(df)
           triad_loc_old(is,ir,itor,2) = triad_loc_old(is,ir,itor,2)  + cprod2

        enddo
      enddo
    enddo
  enddo

end subroutine cgyro_triad_setup



subroutine cgyro_triad_diagnostics
  use timer_lib
  use cgyro_globals

  implicit none

  integer :: is,ix,ie
  integer :: id,itd,itd_class,jr0(0:2),itorbox,jc
  integer :: ir,it,iv_loc_m,ic_loc_m,itor
  integer :: iexch0,itor0,isplit0,iexch_base

  real :: dv,dvr,rval,rval2
  complex :: cprod,cprod2,thfac

#if defined(OMPGPU)
!$omp target teams distribute parallel do simd collapse(3) &
!$omp&         firstprivate(nt1,nt2,n_radial,n_species)
#elif defined(_OPENACC)
!$acc parallel loop collapse(3) gang vector independent default(present) &
!$acc&         firstprivate(nt1,nt2,n_radial,n_species)
#else
!$omp parallel do private(ir,is)
!$omp&         firstprivate(nt1,nt2,n_radial,n_species)
#endif
  do itor=nt1,nt2
      do ir=1,n_radial
        do is=1,n_species
          triad_loc_old(is,ir,itor,5) = triad_loc_old(is,ir,itor,3)
          triad_loc_old(is,ir,itor,3) = triad_loc_old(is,ir,itor,1)
          triad_loc_old(is,ir,itor,1) = 0.0

          triad_loc_old(is,ir,itor,6) = triad_loc_old(is,ir,itor,4)
          triad_loc_old(is,ir,itor,4) = triad_loc_old(is,ir,itor,2)
          triad_loc_old(is,ir,itor,2) = 0.0
          triad_loc(is,ir,itor,:) = 0.0
        enddo
      enddo
  enddo

! Avoiding memory race: inner loop must remain sequential (no vector collapse) - J
  if (nsplitB > 0) then

#if defined(OMPGPU)
!$omp target teams distribute parallel do simd collapse(2) &
!$omp&         private(iv_loc,it,ic_loc_m) &
!$omp&         private(iexch0,itor0,isplit0,iexch_base,is,ix,ie,dv,dvr,rval,rval2,cprod,cprod2) &
!$omp&         private(id,itorbox,jr0,jc,itd,itd_class,thfac) &
!$omp&         firstprivate(nt1,nt2,n_radial,nv_loc,n_theta,nv1,nsplitA,nsplitB,nsplit,nup_theta)
#elif defined(_OPENACC)
!$acc parallel loop gang vector collapse(2) default(present) &
!$acc&         private(iv_loc,it,ic_loc_m) &
!$acc&         private(iexch0,itor0,isplit0,iexch_base,is,ix,ie,dv,dvr,rval,rval2,cprod,cprod2) &
!$acc&         private(id,itorbox,jr0,jc,itd,itd_class,thfac) &
!$acc&         firstprivate(nt1,nt2,n_radial,nv_loc,n_theta,nv1,nsplitA,nsplitB,nsplit,nup_theta)
#else
!$omp parallel do collapse(2) &
!$omp&         private(iv_loc,it,ic_loc_m) &
!$omp&         private(iexch0,itor0,isplit0,iexch_base,is,ix,ie,dv,dvr,rval,rval2,cprod,cprod2) &
!$omp&         private(id,itorbox,jr0,jc,itd,itd_class,thfac) &
!$omp&         firstprivate(nt1,nt2,n_radial,nv_loc,n_theta,nv1,nsplitA,nsplitB,nsplit,nup_theta)
#endif
  do itor=nt1,nt2
    do ir=1,n_radial
      ! Avoiding memory race: inner loop must remain sequential (only collapse 2) - J
      do iv_loc_m=1,nv_loc
        do it=1,n_theta
           itorbox = itor*box_size*sign_qs
           jr0(0) = n_theta*modulo(ir-itorbox-1,n_radial)
           jr0(1) = n_theta*(ir-1)
           jr0(2) = n_theta*modulo(ir+itorbox-1,n_radial)

           ic_loc_m = ic_c(ir,it)

           is = is_v(iv_loc_m +nv1 -1 )
           ix = ix_v(iv_loc_m +nv1 -1 )
           ie = ie_v(iv_loc_m +nv1 -1 )
           dv = w_exi(ie,ix)
           dvr  = w_theta(it)*dens2_rot(it,is)*dv

           ! Density moment
           cprod = w_theta(it)*cap_h_c(ic_loc_m,iv_loc_m,itor)*dvjvec_c(1,ic_loc_m,iv_loc_m,itor)/z(is)
           cprod = -(dvr*z(is)/temp(is)*field(1,ic_loc_m,itor)-cprod)

           iexch0 = (iv_loc_m-1) + (it-1)*nv_loc
           itor0 = iexch0/nsplit
           isplit0 = modulo(iexch0,nsplit)
           if (isplit0 < nsplitA) then
              iexch_base = 1+itor0*nsplitA

              ! 1. Triad energy transfer (All)
              triad_loc(is,ir,itor,1) = triad_loc(is,ir,itor,1) &
                + fpackA(ir,itor-nt1+1,iexch_base+isplit0)*dvr
              ! 2. Triad energy transfer ( {NZ-NZ} coupling , ky!=0)
              if (itor == 0) then
                ! Direct ZF production N
                triad_loc(is,ir,itor,2) = triad_loc(is,ir,itor,2) &
                  + epackA(ir,itor-nt1+1,iexch_base+isplit0)*dvr
              else
                triad_loc(is,ir,itor,2) = triad_loc(is,ir,itor,2) &
                  + epackA(ir,itor-nt1+1,iexch_base+isplit0)*dvr
              endif
           else
              iexch_base = 1+itor0*nsplitB

              ! 1. Triad energy transfer (all)
              triad_loc(is,ir,itor,1) = triad_loc(is,ir,itor,1) &
                + fpackB(ir,itor-nt1+1,iexch_base+(isplit0-nsplitA))*dvr
              ! 2. Triad energy transfer ( {NZ-NZ} coupling , ky!=0)
              if (itor == 0) then
                ! Direct ZF production N
                triad_loc(is,ir,itor,2) = triad_loc(is,ir,itor,2)  &
                  + epackB(ir,itor-nt1+1,iexch_base+(isplit0-nsplitA))*dvr
              else
                triad_loc(is,ir,itor,2) = triad_loc(is,ir,itor,2)  &
                  + epackB(ir,itor-nt1+1,iexch_base+(isplit0-nsplitA))*dvr
              endif
           endif

           cprod = - field(1,ic_loc_m,itor)*conjg(field(1,ic_loc_m,itor))*(z(is)/temp(is))**2*dvr &
                - 2.0*cprod*conjg(field(1,ic_loc_m,itor))*(z(is)/temp(is))
           cprod2  = cap_h_c(ic_loc_m,iv_loc_m,itor)*conjg(cap_h_c(ic_loc_m,iv_loc_m,itor))*dvr + cprod
           ! . Entropy(df) - Entropy(dH)
           triad_loc_old(is,ir,itor,1) = triad_loc_old(is,ir,itor,1)  + cprod
           ! . Entropy(df)
           triad_loc_old(is,ir,itor,2) = triad_loc_old(is,ir,itor,2)  + cprod2

           ! 6. Diss. (radial)
           triad_loc(is,ir,itor,6) = triad_loc(is,ir,itor,6)  &  
                + diss_r(ic_loc_m,iv_loc_m,itor)*h_x(ic_loc_m,iv_loc_m,itor)*conjg(cap_h_c(ic_loc_m,iv_loc_m,itor))*dvr
           ! 7. Diss. (theta )
           rval = omega_stream(it,is,itor)*vel(ie)*xi(ix)
           rval2 = abs(omega_stream(it,is,itor))
           cprod = 0.0 
           cprod2= 0.0

          !icd_c(ic, id, itor)     = ic_c(jr,modulo(it+id-1,n_theta)+1)
          !jc = icd_c(ic, id, itor)
          !dtheta(ic, id, itor)    := cderiv(id)*thfac
          !dtheta_up(ic, id, itor) := uderiv(id)*thfac*up_theta
          itd = n_theta+it-nup_theta
          itd_class = 0
          jc = jr0(itd_class)+itd
          thfac = 0.0

           do id=-nup_theta,nup_theta
              if (itd > n_theta) then
                ! move to next itd_class of compute
                itd = itd - n_theta
                itd_class = itd_class + 1
                jc = jr0(itd_class)+itd
                thfac = -thfac_itor(itd_class,itor)*(itd_class - 2.0)
              endif

              ! Considering not periodic in theta simply by thfac = 0
              cprod2 = cprod2 - rval* thfac*cderiv(id) *(cap_h_c(jc,iv_loc_m,itor) + cap_h_c_old2(jc,iv_loc_m,itor))*0.5
              cprod = cprod - rval2* uderiv(id)*up_theta *g_x(jc,iv_loc_m,itor)
              itd = itd + 1
              jc = jc + 1
           enddo 

           triad_loc(is,ir,itor,7) = triad_loc(is,ir,itor,7) + cprod*conjg(cap_h_c(ic_loc_m,iv_loc_m,itor))*dvr
           ! 8. Diss. (Coll. = Implicit advance - theta_streaming )
           if (explicit_trap_flag == 1) then
              triad_loc(is,ir,itor,8) = triad_loc(is,ir,itor,8) &
                 + ( cap_h_c_triad(iv_loc_m,itor,ic_loc_m)/delta_t )*dvr
           else 
              triad_loc(is,ir,itor,8) = triad_loc(is,ir,itor,8) &
                 + ( cap_h_c_triad(iv_loc_m,itor,ic_loc_m)/delta_t &
                 + cprod2*conjg((cap_h_c(ic_loc_m,iv_loc_m,itor)+cap_h_c_old2(ic_loc_m,iv_loc_m,itor))*0.5) )*dvr
           endif

        enddo
      enddo
    enddo
  enddo

  else ! nsplitB==0

#if defined(OMPGPU)
!$omp target teams distribute parallel do simd collapse(2) &
!$omp&         private(iv_loc,it,ic_loc_m) &
!$omp&         private(iexch0,itor0,isplit0,iexch_base,is,ix,ie,dv,dvr,rval,rval2,cprod,cprod2) &
!$omp&         private(id,itorbox,jr0,jc,itd,itd_class,thfac) &
!$omp&         firstprivate(nt1,nt2,n_radial,nv_loc,n_theta,nv1,nsplitA,nsplit,nup_theta)
#elif defined(_OPENACC)
!$acc parallel loop gang vector collapse(2) default(present) &
!$acc&         private(iv_loc,it,ic_loc_m) &
!$acc&         private(iexch0,itor0,isplit0,iexch_base,is,ix,ie,dv,dvr,rval,rval2,cprod,cprod2) &
!$acc&         private(id,itorbox,jr0,jc,itd,itd_class,thfac) &
!$acc&         firstprivate(nt1,nt2,n_radial,nv_loc,n_theta,nv1,nsplitA,nsplit,nup_theta)
#else
!$omp parallel do collapse(2) &
!$omp&         private(iv_loc,it,ic_loc_m) &
!$omp&         private(iexch0,itor0,isplit0,iexch_base,is,ix,ie,dv,dvr,rval,rval2,cprod,cprod2) &
!$omp&         private(id,itorbox,jr0,jc,itd,itd_class,thfac) &
!$omp&         firstprivate(nt1,nt2,n_radial,nv_loc,n_theta,nv1,nsplitA,nsplit,nup_theta)
#endif
  do itor=nt1,nt2
    do ir=1,n_radial
      do iv_loc_m=1,nv_loc
        do it=1,n_theta
           itorbox = itor*box_size*sign_qs
           jr0(0) = n_theta*modulo(ir-itorbox-1,n_radial)
           jr0(1) = n_theta*(ir-1)
           jr0(2) = n_theta*modulo(ir+itorbox-1,n_radial)

           ic_loc_m = ic_c(ir,it)

           is = is_v(iv_loc_m +nv1 -1 )
           ix = ix_v(iv_loc_m +nv1 -1 )
           ie = ie_v(iv_loc_m +nv1 -1 )
           dv = w_exi(ie,ix)
           dvr  = w_theta(it)*dens2_rot(it,is)*dv

           ! Density moment
           cprod = w_theta(it)*cap_h_c(ic_loc_m,iv_loc_m,itor)*dvjvec_c(1,ic_loc_m,iv_loc_m,itor)/z(is)
           cprod = -(dvr*z(is)/temp(is)*field(1,ic_loc_m,itor)-cprod)

           iexch0 = (iv_loc_m-1) + (it-1)*nv_loc
           itor0 = iexch0/nsplit
           isplit0 = modulo(iexch0,nsplit)
           iexch_base = 1+itor0*nsplitA     

           ! 1. Triad energy transfer (all)
           triad_loc(is,ir,itor,1) = triad_loc(is,ir,itor,1) &
              + fpackA(ir,itor-nt1+1,iexch_base+isplit0)*dvr
           ! 2. Triad energy transfer ( {NZ-NZ} coupling , ky!=0)
           if (itor == 0) then
             ! Direct ZF production N
             triad_loc(is,ir,itor,2) = triad_loc(is,ir,itor,2) &
               + epackA(ir,itor-nt1+1,iexch_base+isplit0)*dvr
           else
             triad_loc(is,ir,itor,2) = triad_loc(is,ir,itor,2) &
               + epackA(ir,itor-nt1+1,iexch_base+isplit0)*dvr
           endif

           cprod = - field(1,ic_loc_m,itor)*conjg(field(1,ic_loc_m,itor))*(z(is)/temp(is))**2*dvr &
                - 2.0*cprod*conjg(field(1,ic_loc_m,itor))*(z(is)/temp(is))
           cprod2  = cap_h_c(ic_loc_m,iv_loc_m,itor)*conjg(cap_h_c(ic_loc_m,iv_loc_m,itor))*dvr + cprod
           ! . Entropy(df) - Entropy(dH)
           triad_loc_old(is,ir,itor,1) = triad_loc_old(is,ir,itor,1)  + cprod
           ! . Entropy(df)
           triad_loc_old(is,ir,itor,2) = triad_loc_old(is,ir,itor,2)  + cprod2


           ! 6. Diss. (radial)
           triad_loc(is,ir,itor,6) = triad_loc(is,ir,itor,6)  &  
                + diss_r(ic_loc_m,iv_loc_m,itor)*h_x(ic_loc_m,iv_loc_m,itor)*conjg(cap_h_c(ic_loc_m,iv_loc_m,itor))*dvr
           ! 7. Diss. (theta )
           rval = omega_stream(it,is,itor)*vel(ie)*xi(ix)
           rval2 = abs(omega_stream(it,is,itor))
           cprod = 0.0 
           cprod2= 0.0

          !icd_c(ic, id, itor)     = ic_c(jr,modulo(it+id-1,n_theta)+1)
          !jc = icd_c(ic, id, itor)
          !dtheta(ic, id, itor)    := cderiv(id)*thfac
          !dtheta_up(ic, id, itor) := uderiv(id)*thfac*up_theta
          itd = n_theta+it-nup_theta
          itd_class = 0
          jc = jr0(itd_class)+itd
          thfac = 0.0

           do id=-nup_theta,nup_theta
              if (itd > n_theta) then
                ! move to next itd_class of compute
                itd = itd - n_theta
                itd_class = itd_class + 1
                jc = jr0(itd_class)+itd
                thfac = -thfac_itor(itd_class,itor)*(itd_class - 2.0)
              endif

              ! Considering not periodic in theta simply by thfac = 0
              cprod2 = cprod2 - rval* thfac*cderiv(id) *(cap_h_c(jc,iv_loc_m,itor)+cap_h_c_old2(jc,iv_loc_m,itor))*0.5
              cprod = cprod - rval2* uderiv(id)*up_theta *g_x(jc,iv_loc_m,itor)
              itd = itd + 1
              jc = jc + 1
           enddo 

           triad_loc(is,ir,itor,7) = triad_loc(is,ir,itor,7) + cprod*conjg(cap_h_c(ic_loc_m,iv_loc_m,itor))*dvr
           ! 8. Diss. (Coll. = Implicit advance - theta_streaming )
           if (explicit_trap_flag == 1) then
              triad_loc(is,ir,itor,8) = triad_loc(is,ir,itor,8) &
                 + ( cap_h_c_triad(iv_loc_m,itor,ic_loc_m)/delta_t )*dvr
           else 
              triad_loc(is,ir,itor,8) = triad_loc(is,ir,itor,8) &
                 + ( cap_h_c_triad(iv_loc_m,itor,ic_loc_m)/delta_t &
                 + cprod2*conjg((cap_h_c(ic_loc_m,iv_loc_m,itor)+cap_h_c_old2(ic_loc_m,iv_loc_m,itor))*0.5) )*dvr
           endif

        enddo
      enddo
    enddo
  enddo

  endif ! if nsplitB>0


! Compute Time difference

#if defined(OMPGPU)
!$omp target teams distribute parallel do simd collapse(3) &
!$omp&         firstprivate(nt1,nt2,n_radial,n_species)
#elif defined(_OPENACC)
!$acc parallel loop collapse(3) gang vector independent default(present) &
!$acc&         firstprivate(nt1,nt2,n_radial,n_species)
#else
!$omp parallel do private(ir,is)
!$omp&         firstprivate(nt1,nt2,n_radial,n_species)
#endif
  do itor=nt1,nt2
      do ir=1,n_radial
        do is=1,n_species
          ! 3. dEntropy / dt
          triad_loc(is,ir,itor,3) = &
          (triad_loc_old(is,ir,itor,6) - 4*triad_loc_old(is,ir,itor,4) + 3*triad_loc_old(is,ir,itor,2))/(2*delta_t)*0.5
          ! 4. dEM field /dt    ,  remaining term will be added in cgyro_flux 
          triad_loc(is,ir,itor,4) = &
          (triad_loc_old(is,ir,itor,5) - 4*triad_loc_old(is,ir,itor,3) + 3*triad_loc_old(is,ir,itor,1))/(2*delta_t)*0.5
          ! 5. Entropy 
          triad_loc(is,ir,itor,5) = triad_loc_old(is,ir,itor,2)*0.5
        enddo
      enddo
  enddo


end subroutine cgyro_triad_diagnostics
