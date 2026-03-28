!-----------------------------------------------------------------
! cgyro_nl_comm.F90
!
! PURPOSE:
!  Nonlinear communication routines
!-----------------------------------------------------------------

module cgyro_nl_comm

  implicit none

contains

subroutine cgyro_nl_dealias_init

  use cgyro_globals

  implicit none

  integer :: itor,l0,l,p,m,l0_max,nt1_nz
  integer :: iex,ir,it,i,panel,npanel,nex
  real :: phase
  integer :: max_pvec_count
  ! max_l0 = box_size*nt2
  integer :: pvec_count(box_size*nt2) !local, temp copy
  integer :: dealias_iex(-3:3)
  integer, dimension(:), allocatable :: dealias_fex_ir
  integer, dimension(:), allocatable :: dealias_fex_it
  complex, dimension(:), allocatable :: dealias_fex_ph
  integer, dimension(:), allocatable :: dealias_pvec_count
  integer, dimension(:,:), allocatable :: dealias_pvec

  ! Wavenumber M from CGYRO paper
  m = n_radial/2

  ! Phase factor (JC: is q*sign_qs correct?)
  phase = 2.0*pi*(q*sign_qs)/box_size

  if (nt1==0) then
    ! don't need pvec for itor==0
    nt1_nz = 1
  else
    nt1_nz = nt1
  endif

  !
  ! max_pvec_count is global and we are defining it here
  !
  ! We will have arrays that depend on this later on
  ! so, precompute early
  !
  max_pvec_count = 1 ! make it >0, to always have valid arrays

  do itor=nt1_nz,nt2
      ! Total number of ballooning angles for finite-n ballooning mode
      l0 = box_size*itor

      pvec_count(1:l0) = 0

      ! Sort p indices by ballooning angle index l
      do p=-m,m-1
         l = mod(mod(p,l0)+l0,l0)+1
         pvec_count(l) = pvec_count(l)+1
         if (pvec_count(l) > max_pvec_count) then
           max_pvec_count = pvec_count(l)
         endif
      enddo
  enddo

  ! need intermediate buffers for uniform compute
  ! required for GPU kernels, but will keep for CPU as well

  ! Will use the same buffers for both h_x and field transforms
  ! since they are never done in parallel
  ! Size them for the worst case
  if (n_field>nv_loc) then
     allocate(inraw_dealias(n_radial,n_theta,n_field,nt1:nt2))
     allocate(outraw_dealias(n_radial,n_theta,n_field,nt1:nt2))
  else
     allocate(inraw_dealias(n_radial,n_theta,nv_loc,nt1:nt2))
     allocate(outraw_dealias(n_radial,n_theta,nv_loc,nt1:nt2))
  endif
#if defined(OMPGPU)
!$omp target enter data map(alloc:inraw_dealias,outraw_dealias)
#elif defined(_OPENACC)
!$acc enter data create(inraw_dealias,outraw_dealias)
#endif

  allocate(dealias_fex_ir(n_theta*max_pvec_count))
  allocate(dealias_fex_it(n_theta*max_pvec_count))
  allocate(dealias_fex_ph(n_theta*max_pvec_count))

  allocate(dealias_raw_ir(n_radial,n_theta,-3:3,nt1:nt2))
  allocate(dealias_raw_it(n_radial,n_theta,-3:3,nt1:nt2))
  allocate(dealias_raw_ph(n_radial,n_theta,-3:3,nt1:nt2))

  ! pre-compute pvec vectors, since they are re-used often
  l0_max = box_size*nt2
  if (l0_max==0) l0_max=1 ! need a valid array, even if not used

  ! allocate eventual itor==0, too, to make it easier to use
  allocate(dealias_pvec_count(l0_max))
  allocate(dealias_pvec(l0_max,max_pvec_count))

  ! now do the full compute on global buffers
  ! cheap, since done only once, do on CPU single threaded
  do itor=nt1_nz,nt2 ! we can leave eventual itor==0 unitialized, not used
      ! Total number of ballooning angles for finite-n ballooning mode
      l0 = box_size*itor

      dealias_pvec_count(:) = 0

      ! Sort p indices by ballooning angle index l
      do p=-m,m-1
        l = mod(mod(p,l0)+l0,l0)+1
        dealias_pvec_count(l) = dealias_pvec_count(l)+1
        dealias_pvec(l,dealias_pvec_count(l)) = p
      enddo

      ! Construct ballooning modes
      ! all _ex quantities refer to extended angle
      ! nex = total number of points along extended angle
      ! iex = extended angle index
      do l=1,l0
       npanel = dealias_pvec_count(l)
       nex = n_theta*npanel
       do panel=1,npanel
        do it=1,n_theta
           p = dealias_pvec(l,panel)
           if (sign_qs > 0.0) then 
              ir = p+m+1
           else
              ir = n_radial-(p+m)
           endif
           iex = (panel-1)*n_theta+it
           ! save indexes and phase
           dealias_fex_ir(iex) = ir
           dealias_fex_it(iex) = it
           dealias_fex_ph(iex) = exp(-i_c*p*phase)
        enddo
       enddo ! do panel
       do iex=1,nex
       
        do i=-3,-1
          dealias_iex(i) = iex+i
          if (dealias_iex(i) < 1) dealias_iex(i) = dealias_iex(i)+nex
        enddo 
        dealias_iex(0) = iex
        do i=1,3
          dealias_iex(i) = iex+i
          if (dealias_iex(i) > nex) dealias_iex(i) = dealias_iex(i)-nex
        enddo 

        ! iex = (panel-1)*n_theta+it
        ! p = pvec(l,panel,itor-itor_offset)
        ! ir = p+m+1
        it = 1+modulo(iex-1,n_theta)
        panel = 1+(iex-1)/n_theta
        p = dealias_pvec(l,panel)
        if (sign_qs > 0.0) then
           ir = p+m+1
        else
           ir = n_radial-(p+m)
        endif
        ! now save the indexes and phases to global space
        do i=-3,3
          dealias_raw_ir(ir,it,i,itor) = dealias_fex_ir(dealias_iex(i))
          dealias_raw_it(ir,it,i,itor) = dealias_fex_it(dealias_iex(i))
          dealias_raw_ph(ir,it,i,itor) = dealias_fex_ph(dealias_iex(i))
        enddo
       enddo ! do iex
      enddo ! do l
  enddo

  if (nt1==0) then
    ! itor=0 special case
    ! Pure periodicity
    itor = 0
    do it=1,n_theta
      do ir=1,n_radial
        do i=-3,-1
          dealias_raw_it(ir,it,i,itor) = it+i
          if (dealias_raw_it(ir,it,i,itor) < 1) then
            dealias_raw_it(ir,it,i,itor) = dealias_raw_it(ir,it,i,itor)+n_theta
          endif
        enddo 
        dealias_raw_it(ir,it,i,itor) = it
        do i=1,3
          dealias_raw_it(ir,it,i,itor) = it+i
          if (dealias_raw_it(ir,it,i,itor) > n_theta) then
            dealias_raw_it(ir,it,i,itor) = dealias_raw_it(ir,it,i,itor)-n_theta
          endif
        enddo 
        do i=-3,3
          dealias_raw_ir(ir,it,i,itor) = ir
          dealias_raw_ph(ir,it,i,itor) = 1.0
        enddo
      enddo
    enddo
  endif

#if defined(OMPGPU)
!$omp target enter data map(to:dealias_raw_ir,dealias_raw_it,dealias_raw_ph)
#elif defined(_OPENACC)
!$acc enter data copyin(dealias_raw_ir,dealias_raw_it,dealias_raw_ph)
#endif

  deallocate(dealias_fex_ph)
  deallocate(dealias_fex_it)
  deallocate(dealias_fex_ir)

  deallocate(dealias_pvec_count)
  deallocate(dealias_pvec)

end subroutine cgyro_nl_dealias_init


! === Internal ===
! Extended-angle filter for nonlinear dealiasing
! fraw (input unfiltered field) array
! f    (output filtered field) array

subroutine impfilter5_n0(&
                    n_theta,n_radial,n_3d,nt1,nt2,&
                    a0,a1,a2,a3,&
                    fraw,f,itor_offset)

  use cgyro_globals, ONLY: dealias_raw_ir,dealias_raw_it,dealias_raw_ph
  implicit none

  integer, intent(in) :: n_theta,n_radial,n_3d
  integer, intent(in) :: nt1,nt2
  real, intent(in) :: a0,a1,a2,a3
  ! both are (n_radial,n_theta,:,itor_offset:(nt2-itor_offset))
  complex, intent(in) :: fraw(:,:,:,:)
  complex, intent(out):: f(:,:,:,:)
  integer, intent(in) :: itor_offset
  ! -------------
  integer :: itor,i3d
  integer :: ir,it
  complex :: fval,my_ph

#if defined(OMPGPU)
!$omp target teams distribute parallel do collapse(3) &
!$omp&         private(fval,my_ph,i3d)
  do itor=nt1,nt2
   do it=1,n_theta
    do ir=1,n_radial
     do i3d=1,n_3d  ! do sequentially, good for GPUs
#elif defined(_OPENACC)
!$acc parallel loop gang vector collapse(3) &
!$acc&         private(fval,my_ph,i3d) &
!$acc&         present(fraw,f,dealias_raw_ir,dealias_raw_it,dealias_raw_ph)
  do itor=nt1,nt2
   do it=1,n_theta
    do ir=1,n_radial
     do i3d=1,n_3d  ! do sequentially, good for GPUs
#else
!$omp parallel do collapse(2) &
!$omp&         private(fval,my_ph,ir,it)
  do itor=nt1,nt2
   do i3d=1,n_3d
    do it=1,n_theta
     do ir=1,n_radial
#endif
        ! ir == dealias_raw_ir(ir,it,0,itor)
        ! it == dealias_raw_it(ir,it,0,itor)
        my_ph = dealias_raw_ph(ir,it,0,itor)
        fval = &
             a0*(&
                 fraw(ir,it,i3d,itor-itor_offset)*&
                 my_ph&
                ) + &
             a1*(&
                 fraw(dealias_raw_ir(ir,it,-1,itor),dealias_raw_it(ir,it,-1,itor),i3d,itor-itor_offset)*&
                 dealias_raw_ph(ir,it,-1,itor) +&
                 fraw(dealias_raw_ir(ir,it,+1,itor),dealias_raw_it(ir,it,+1,itor),i3d,itor-itor_offset)*&
                 dealias_raw_ph(ir,it,+1,itor) &
                ) + &
             a2*(&
                 fraw(dealias_raw_ir(ir,it,-2,itor),dealias_raw_it(ir,it,-2,itor),i3d,itor-itor_offset)*&
                 dealias_raw_ph(ir,it,-2,itor) +&
                 fraw(dealias_raw_ir(ir,it,+2,itor),dealias_raw_it(ir,it,+2,itor),i3d,itor-itor_offset)*&
                 dealias_raw_ph(ir,it,+2,itor) &
                ) + &
             a3*(&
                 fraw(dealias_raw_ir(ir,it,-3,itor),dealias_raw_it(ir,it,-3,itor),i3d,itor-itor_offset)*&
                 dealias_raw_ph(ir,it,-3,itor) +&
                 fraw(dealias_raw_ir(ir,it,+3,itor),dealias_raw_it(ir,it,+3,itor),i3d,itor-itor_offset)*&
                 dealias_raw_ph(ir,it,+3,itor) &
                )

        ! dephase
        f(ir,it,i3d,itor-itor_offset) = fval*conjg(my_ph)
     enddo
    enddo
   enddo !do i3d=1,n_3d
  enddo !do itor=nt1,nt2
end subroutine impfilter5_n0

subroutine impfilter5(&
                    n_theta,n_radial,n_3d,nt1,nt2,&
                    dealias_order,dealias,&
                    fraw,f)

  implicit none

  integer, intent(in) :: n_theta,n_radial,n_3d,nt1,nt2
  integer, intent(in) :: dealias_order
  real, intent(in) :: dealias
  ! both are (n_radial,n_theta,:,1:(nt2-nt1+1))
  complex, intent(in) :: fraw(:,:,:,:)
  complex, intent(out):: f(:,:,:,:)
  ! -----------
  real :: a0,a1,a2,a3

  ! Maximally flat filters (3, 5, or 7-point):
  !
  ! H(0) = 1
  ! H'(0) = H''(0) = ... = 0
  ! H(pi) = 1-dealias
  !
  ! dealias = 0 (no filter)
  ! dealias = 1 (max filter)
  ! dealias_order = 1, 2, or 3 (for 3,5,7 pt)

  if (dealias_order == 1) then
     ! 3-point filter
     a0 = (2-dealias)/2.0
     a1 = dealias/4.0
     a2 = 0.0
     a3 = 0.0
  elseif (dealias_order == 2) then
     ! 5-point filter
     a0 = (8-3*dealias)/8.0
     a1 = dealias/4.0
     a2 = -dealias/16.0
     a3 = 0.0
  else
     ! 7-point filter (default)
     a0 = (16-5*dealias)/16.0
     a1 = (15/64.0)*dealias
     a2 = -(6/64.0)*dealias
     a3 = (1/64.0)*dealias
  endif

  call impfilter5_n0(n_theta,n_radial,n_3d,nt1,nt2,&
                        a0,a1,a2,a3,&
                        fraw,f,nt1-1)

end subroutine impfilter5

subroutine hx_dealias

  use cgyro_globals

  implicit none
  
  integer :: ir,it,iv_loc_m,itor
  
  if ((dealias_order == 0) .or. (dealias == 0.0)) then

   ! Just copy over
#if defined(OMPGPU)
!$omp target teams distribute parallel do collapse(4) default(shared)
#elif defined(_OPENACC)
!$acc parallel loop gang vector collapse(4) present(outraw_dealias,h_x)
#else
!$omp parallel do collapse(4) default(shared)
#endif
   do itor=nt1,nt2
    do iv_loc_m=1,nv_loc
      do it=1,n_theta
        do ir=1,n_radial
          ! ic_c(ir,it) = (ir-1)*n_theta+it
          outraw_dealias(ir,it,iv_loc_m,itor) = h_x((ir-1)*n_theta+it,iv_loc_m,itor)
        enddo
      enddo
    enddo
   enddo

  else

   ! Construct h_x_dealias array
#if defined(OMPGPU)
!$omp target teams distribute parallel do collapse(4) default(shared)
#elif defined(_OPENACC)
!$acc parallel loop gang vector collapse(4) present(inraw_dealias,h_x)
#else
!$omp parallel do collapse(4) default(shared)
#endif
   do itor=nt1,nt2
    do iv_loc_m=1,nv_loc
      do it=1,n_theta
        do ir=1,n_radial
          ! ic_c(ir,it) = (ir-1)*n_theta+it
          inraw_dealias(ir,it,iv_loc_m,itor) = h_x((ir-1)*n_theta+it,iv_loc_m,itor)
        enddo
      enddo
    enddo
   enddo
   ! Extended-angle dealiasing filter
   call impfilter5(n_theta,n_radial,nv_loc,nt1,nt2,&
                  dealias_order,dealias,&
                  inraw_dealias,outraw_dealias)

  endif
end subroutine hx_dealias

subroutine field_dealias

  use cgyro_globals

  implicit none
  
  integer :: ir,it,itf,itor

  if ((dealias_order == 0) .or. (dealias == 0.0)) then

   ! Just copy over
#if defined(OMPGPU)
!$omp target teams distribute parallel do collapse(4) default(shared)
#elif defined(_OPENACC)
!$acc parallel loop gang vector collapse(4) present(outraw_dealias,field)
#else
!$omp parallel do collapse(4) default(shared)
#endif
   do itor=nt1,nt2
    do itf=1,n_field
      do it=1,n_theta
        do ir=1,n_radial
          ! ic_c(ir,it) = (ir-1)*n_theta+it
          outraw_dealias(ir,it,itf,itor) = field(itf,(ir-1)*n_theta+it,itor)
        enddo
      enddo
    enddo
   enddo

  else

   ! Construct field_dealias array
#if defined(OMPGPU)
!$omp target teams distribute parallel do collapse(4) default(shared)
#elif defined(_OPENACC)
!$acc parallel loop gang vector collapse(4) present(inraw_dealias,field)
#else
!$omp parallel do collapse(4) default(shared)
#endif
   do itor=nt1,nt2
    do itf=1,n_field
      do it=1,n_theta
        do ir=1,n_radial
          ! ic_c(ir,it) = (ir-1)*n_theta+it
          inraw_dealias(ir,it,itf,itor) = field(itf,(ir-1)*n_theta+it,itor)
        enddo
      enddo
    enddo
   enddo
   ! Extended angle dealiasing filter
   call impfilter5(n_theta,n_radial,n_field,nt1,nt2,&
                  dealias_order,dealias,&
                  inraw_dealias,outraw_dealias)

  endif
end subroutine field_dealias

!
! ========= End internal subroutines ============
!

! Note: Calling test propagates the async operations in some MPI implementations
subroutine cgyro_nl_fftw_comm_test
  use parallel_lib
  use cgyro_globals

  implicit none

  if (fA_req_valid) call parallel_slib_test(fA_req)
  if (g_req_valid) call parallel_slib_test(g_req)
  if (fB_req_valid) call parallel_slib_test(fB_req)

end subroutine cgyro_nl_fftw_comm_test

!
! Comm is a transposea
! Reminder: nc ~= n_radial*n_theta
! First half of the transpose is done locally
!  from (theta,radial,nv_loc,nt_loc) -> (radial, nt_loc, theta, nv_loc)
! Then AlltoAll finishes the transpose
!  from (radial, nt_loc, theta, nv_loc_1, nv_loc_2) x toroidal_procs -> (radial, nt_loc, theta, nv_loc_1 , toroidal_procs) x nv_loc_2
! Implies nv_loc_2 == toroidal_procs
!

! NOTE: call cgyro_nl_fftw_comm1/2_async before cgyro_nl_fftw
subroutine cgyro_nl_fftw_comm1_f64_async
  use timer_lib
  use parallel_lib
  use cgyro_globals

  implicit none

  integer :: ir,it,iv_loc_m,itor
  integer :: iexch0,itor0,isplit0,iexch_base
  complex :: h_loc

  call timer_lib_in('nl_mem')

  call hx_dealias

  if (nsplitB > 0) then

#if defined(OMPGPU)
!$omp target teams distribute parallel do collapse(4) &
!$omp&         private(iexch0,itor0,isplit0,iexch_base,h_loc)
#elif defined(_OPENACC)
!$acc parallel loop collapse(4) gang vector independent &
!$acc&         private(iexch0,itor0,isplit0,iexch_base,h_loc) &
!$acc&         present(outraw_dealias,fpackA,fpackB) &
!$acc&         present(n_theta,nv_loc,nt1,nt2,n_radial,nsplit,nsplitA,nsplitB) default(none)
#else
!$omp parallel do collapse(3) &
!$omp&         private(iexch0,itor0,isplit0,iexch_base,h_loc)
#endif
  do iv_loc_m=1,nv_loc
   do itor=nt1,nt2
    do it=1,n_theta
     do ir=1,n_radial
       h_loc = outraw_dealias(ir,it,iv_loc_m,itor)
       iexch0 = (iv_loc_m-1) + (it-1)*nv_loc
       itor0 = iexch0/nsplit
       isplit0 = modulo(iexch0,nsplit)
       if (isplit0 < nsplitA) then
          iexch_base = 1+itor0*nsplitA
          fpackA(ir,itor-nt1+1,iexch_base+isplit0) = h_loc
       else
          iexch_base = 1+itor0*nsplitB
          fpackB(ir,itor-nt1+1,iexch_base+(isplit0-nsplitA)) = h_loc
       endif
     enddo
    enddo
   enddo
  enddo

  if ( (nv_loc*n_theta) < (nsplit*n_toroidal_procs) ) then
#if defined(OMPGPU)
!$omp target teams distribute parallel do simd &
!$omp&         private(iexch0,itor0,isplit0,iexch_base)
#elif defined(_OPENACC)
!$acc parallel loop independent gang vector &
!$acc&         private(iexch0,itor0,isplit0,iexch_base) &
!$acc&         present(fpackA,fpackB,nsplit,nsplitA,nsplitB)
#endif
    do iexch0=nv_loc*n_theta,nsplit*n_toroidal_procs-1
       itor0 = iexch0/nsplit
       isplit0 = modulo(iexch0,nsplit)
       if (isplit0 < nsplitA) then
          iexch_base = 1+itor0*nsplitA
          fpackA(1:n_radial,1:nt_loc,iexch_base+isplit0) = (0.0,0.0)
       else
          iexch_base = 1+itor0*nsplitB
          fpackB(1:n_radial,1:nt_loc,iexch_base+(isplit0-nsplitA)) = (0.0,0.0)
       endif
    enddo
  endif

  else ! nsplitB==0

#if defined(OMPGPU)
!$omp target teams distribute parallel do collapse(4) &
!$omp&         private(iexch0,itor0,isplit0,iexch_base,h_loc)
#elif defined(_OPENACC)
!$acc parallel loop collapse(4) gang vector independent &
!$acc&         private(iexch0,itor0,isplit0,iexch_base,h_loc) &
!$acc&         present(outraw_dealias,fpackA) &
!$acc&         present(n_theta,nv_loc,nt1,nt2,n_radial,nsplit,nsplitA) default(none)
#else
!$omp parallel do collapse(3) &
!$omp&         private(iexch0,itor0,isplit0,iexch_base,h_loc)
#endif
  do iv_loc_m=1,nv_loc
   do itor=nt1,nt2
    do it=1,n_theta
     do ir=1,n_radial
       h_loc = outraw_dealias(ir,it,iv_loc_m,itor)
       iexch0 = (iv_loc_m-1) + (it-1)*nv_loc
       itor0 = iexch0/nsplit
       isplit0 = modulo(iexch0,nsplit)
       iexch_base = 1+itor0*nsplitA
       fpackA(ir,itor-nt1+1,iexch_base+isplit0) = h_loc
     enddo
    enddo
   enddo
  enddo

  if ( (nv_loc*n_theta) < (nsplit*n_toroidal_procs) ) then
#if defined(OMPGPU)
!$omp target teams distribute parallel do simd &
!$omp&         private(iexch0,itor0,isplit0,iexch_base)
#elif defined(_OPENACC)
!$acc parallel loop independent gang vector &
!$acc&         private(iexch0,itor0,isplit0,iexch_base) &
!$acc&         present(fpackA,nsplit,nsplitA)
#endif
    do iexch0=nv_loc*n_theta,nsplit*n_toroidal_procs-1
       itor0 = iexch0/nsplit
       isplit0 = modulo(iexch0,nsplit)
       iexch_base = 1+itor0*nsplitA
       fpackA(1:n_radial,1:nt_loc,iexch_base+isplit0) = (0.0,0.0)
    enddo
  endif

  endif ! if nspliB>0

  call timer_lib_out('nl_mem')

  call timer_lib_in('nl_comm')
  ! split the comm in two, so we can start working on first as soon as it is ready
  call parallel_slib_f_nc_async(nsplitA,fpackA,fA_nl,fA_req)
  fA_req_valid = .TRUE.
  ! send only the first half, use comm3 to send the other half
  call timer_lib_out('nl_comm')

end subroutine cgyro_nl_fftw_comm1_f64_async

subroutine cgyro_nl_fftw_comm1_f32_async
  use timer_lib
  use parallel_lib
  use cgyro_globals

  implicit none

  integer :: ir,it,iv_loc_m,itor
  integer :: iexch0,itor0,isplit0,iexch_base
  complex(KIND=REAL32) :: h_loc

  call timer_lib_in('nl_mem')

  call hx_dealias

  if (nsplitB > 0) then

#if defined(OMPGPU)
!$omp target teams distribute parallel do collapse(4) &
!$omp&         private(iexch0,itor0,isplit0,iexch_base,h_loc)
#elif defined(_OPENACC)
!$acc parallel loop collapse(4) gang vector independent &
!$acc&         private(iexch0,itor0,isplit0,iexch_base,h_loc) &
!$acc&         present(outraw_dealias,fpackA32,fpackB32) &
!$acc&         present(n_theta,nv_loc,nt1,nt2,n_radial,nsplit,nsplitA,nsplitB) default(none)
#else
!$omp parallel do collapse(3) &
!$omp&         private(iexch0,itor0,isplit0,iexch_base,h_loc)
#endif
  do iv_loc_m=1,nv_loc
   do itor=nt1,nt2
    do it=1,n_theta
     do ir=1,n_radial
       h_loc = outraw_dealias(ir,it,iv_loc_m,itor)
       iexch0 = (iv_loc_m-1) + (it-1)*nv_loc
       itor0 = iexch0/nsplit
       isplit0 = modulo(iexch0,nsplit)
       if (isplit0 < nsplitA) then
          iexch_base = 1+itor0*nsplitA
          fpackA32(ir,itor-nt1+1,iexch_base+isplit0) = h_loc
       else
          iexch_base = 1+itor0*nsplitB
          fpackB32(ir,itor-nt1+1,iexch_base+(isplit0-nsplitA)) = h_loc
       endif
     enddo
    enddo
   enddo
  enddo

  if ( (nv_loc*n_theta) < (nsplit*n_toroidal_procs) ) then
#if defined(OMPGPU)
!$omp target teams distribute parallel do simd &
!$omp&         private(iexch0,itor0,isplit0,iexch_base)
#elif defined(_OPENACC)
!$acc parallel loop independent gang vector &
!$acc&         private(iexch0,itor0,isplit0,iexch_base) &
!$acc&         present(fpackA32,fpackB32,nsplit,nsplitA,nsplitB)
#endif
    do iexch0=nv_loc*n_theta,nsplit*n_toroidal_procs-1
       itor0 = iexch0/nsplit
       isplit0 = modulo(iexch0,nsplit)
       if (isplit0 < nsplitA) then
          iexch_base = 1+itor0*nsplitA
          fpackA32(1:n_radial,1:nt_loc,iexch_base+isplit0) = (0.0,0.0)
       else
          iexch_base = 1+itor0*nsplitB
          fpackB32(1:n_radial,1:nt_loc,iexch_base+(isplit0-nsplitA)) = (0.0,0.0)
       endif
    enddo
  endif

  else ! nsplitB==0

#if defined(OMPGPU)
!$omp target teams distribute parallel do collapse(4) &
!$omp&         private(iexch0,itor0,isplit0,iexch_base,h_loc)
#elif defined(_OPENACC)
!$acc parallel loop collapse(4) gang vector independent &
!$acc&         private(iexch0,itor0,isplit0,iexch_base,h_loc) &
!$acc&         present(outraw_dealias,fpackA32) &
!$acc&         present(n_theta,nv_loc,nt1,nt2,n_radial,nsplit,nsplitA) default(none)
#else
!$omp parallel do collapse(3) &
!$omp&         private(iexch0,itor0,isplit0,iexch_base,h_loc)
#endif
  do iv_loc_m=1,nv_loc
   do itor=nt1,nt2
    do it=1,n_theta
     do ir=1,n_radial
       h_loc = outraw_dealias(ir,it,iv_loc_m,itor)
       iexch0 = (iv_loc_m-1) + (it-1)*nv_loc
       itor0 = iexch0/nsplit
       isplit0 = modulo(iexch0,nsplit)
       iexch_base = 1+itor0*nsplitA
       fpackA32(ir,itor-nt1+1,iexch_base+isplit0) = h_loc
     enddo
    enddo
   enddo
  enddo

  if ( (nv_loc*n_theta) < (nsplit*n_toroidal_procs) ) then
#if defined(OMPGPU)
!$omp target teams distribute parallel do simd &
!$omp&         private(iexch0,itor0,isplit0,iexch_base)
#elif defined(_OPENACC)
!$acc parallel loop independent gang vector &
!$acc&         private(iexch0,itor0,isplit0,iexch_base) &
!$acc&         present(fpackA32,nsplit,nsplitA)
#endif
    do iexch0=nv_loc*n_theta,nsplit*n_toroidal_procs-1
       itor0 = iexch0/nsplit
       isplit0 = modulo(iexch0,nsplit)
       iexch_base = 1+itor0*nsplitA
       fpackA32(1:n_radial,1:nt_loc,iexch_base+isplit0) = (0.0,0.0)
    enddo
  endif

  endif ! if nspliB>0

  call timer_lib_out('nl_mem')

  call timer_lib_in('nl_comm')
  ! split the comm in two, so we can start working on first as soon as it is ready
  call parallel_slib_f_nc32_async(nsplitA,fpackA32,fA_nl32,fA_req)
  fA_req_valid = .TRUE.
  ! send only the first half, use comm3 to send the other half
  call timer_lib_out('nl_comm')

end subroutine cgyro_nl_fftw_comm1_f32_async

subroutine cgyro_nl_fftw_comm1_async

  use cgyro_globals

  implicit none
  !-----------------------------------

  if (nl_single_flag > 1) then
    call cgyro_nl_fftw_comm1_f32_async
  else
    call cgyro_nl_fftw_comm1_f64_async
  endif
end subroutine cgyro_nl_fftw_comm1_async

subroutine cgyro_nl_fftw_comm3_async
  use timer_lib
  use parallel_lib
  use cgyro_globals

  implicit none

  if (nsplitB > 0) then
    call timer_lib_in('nl_comm')

    if (nl_single_flag > 1) then
      call parallel_slib_f_nc32_async(nsplitB,fpackB32,fB_nl32,fB_req)
    else
      call parallel_slib_f_nc_async(nsplitB,fpackB,fB_nl,fB_req)
    endif
    fB_req_valid = .TRUE.

    call timer_lib_out('nl_comm')
  endif
end subroutine cgyro_nl_fftw_comm3_async

subroutine cgyro_nl_fftw_comm1_r64(ij)
  use timer_lib
  use parallel_lib
  use cgyro_globals

  implicit none

  !-----------------------------------
  integer, intent(in) :: ij
  !-----------------------------------

  integer :: ir,it,iv_loc_m,ic_loc_m,itor
  integer :: iexch0,itor0,isplit0,iexch_base
  complex :: my_psi
  real :: psi_mul

  call timer_lib_in('nl_comm')
  call parallel_slib_r_nc_wait(nsplitA,fA_nl,fpackA,fA_req)
  fA_req_valid = .FALSE.
  if (nsplitB > 0) then
    ! no major compute to overlap
    call parallel_slib_r_nc_wait(nsplitB,fB_nl,fpackB,fB_req)
    fB_req_valid = .FALSE.
  endif
  call timer_lib_out('nl_comm')

  call timer_lib_in('nl')

  psi_mul = (q*rho/rmin)*(2*pi/length)

  if (nsplitB > 0) then

#if defined(OMPGPU)
!$omp target teams distribute parallel do simd collapse(4) &
!$omp&         private(iexch0,itor0,isplit0,iexch_base) &
!$omp&         private(ic_loc_m,my_psi) firstprivate(px_zero)
#elif defined(_OPENACC)
!$acc parallel loop collapse(4) gang vector independent private(ic_loc_m,my_psi) &
!$acc&         private(iexch0,itor0,isplit0,iexch_base) firstprivate(px_zero) &
!$acc&         present(ic_c,rhs,fpackA,fpackB) copyin(psi_mul,nl_min) &
!$acc&         present(nt1,nt2,nv_loc,n_theta,n_radial,nsplit,nsplitA,nsplitB) copyin(ij) default(none)
#else
!$omp parallel do collapse(2) private(ic_loc_m,my_psi) &
!$omp&         private(iexch0,itor0,isplit0,iexch_base) firstprivate(px_zero)
#endif
  do itor=nt1,nt2
    do iv_loc_m=1,nv_loc
      do it=1,n_theta
        do ir=1,n_radial
           ic_loc_m = ic_c(ir,it)
           if ( (itor == 0) .and.  ((ir == 1) .or. (ir == px_zero)) ) then
              ! filter
              my_psi = (0.0,0.0)
           else
              iexch0 = (iv_loc_m-1) + (it-1)*nv_loc
              itor0 = iexch0/nsplit
              isplit0 = modulo(iexch0,nsplit)
              if (isplit0 < nsplitA) then
                 iexch_base = 1+itor0*nsplitA
                 my_psi = fpackA(ir,itor-nt1+1,iexch_base+isplit0)
              else
                 iexch_base = 1+itor0*nsplitB
                 my_psi = fpackB(ir,itor-nt1+1,iexch_base+(isplit0-nsplitA))
              endif
           endif           
           if (itor < nl_min) then
              my_psi = (0.0,0.0)
           endif
           
           ! RHS -> -[f,g] = [f,g]_{r,-alpha}
           rhs(ic_loc_m,iv_loc_m,itor,ij) = rhs(ic_loc_m,iv_loc_m,itor,ij)+psi_mul*my_psi
        enddo
      enddo
    enddo
  enddo

  else ! nsplitB==0

#if defined(OMPGPU)
!$omp target teams distribute parallel do simd collapse(4) &
!$omp&         private(iexch0,itor0,isplit0,iexch_base) &
!$omp&         private(ic_loc_m,my_psi) firstprivate(px_zero)
#elif defined(_OPENACC)
!$acc parallel loop collapse(4) gang vector independent private(ic_loc_m,my_psi) &
!$acc&         private(iexch0,itor0,isplit0,iexch_base) firstprivate(px_zero) &
!$acc&         present(ic_c,rhs,fpackA) copyin(psi_mul,nl_min) &
!$acc&         present(nt1,nt2,nv_loc,n_theta,n_radial,nsplit,nsplitA) copyin(ij) default(none)
#else
!$omp parallel do collapse(2) private(ic_loc_m,my_psi) &
!$omp&         private(iexch0,itor0,isplit0,iexch_base) firstprivate(px_zero)
#endif
  do itor=nt1,nt2
    do iv_loc_m=1,nv_loc
      do it=1,n_theta
        do ir=1,n_radial
           ic_loc_m = ic_c(ir,it)
           if ( (itor == 0) .and.  ((ir == 1) .or. (ir == px_zero)) ) then
              ! filter
              my_psi = (0.0,0.0)
           else
              iexch0 = (iv_loc_m-1) + (it-1)*nv_loc
              itor0 = iexch0/nsplit
              isplit0 = modulo(iexch0,nsplit)
              iexch_base = 1+itor0*nsplitA
              my_psi = fpackA(ir,itor-nt1+1,iexch_base+isplit0)
           endif           
           if (itor < nl_min) then
              my_psi = (0.0,0.0)
           endif
           
           ! RHS -> -[f,g] = [f,g]_{r,-alpha}
           rhs(ic_loc_m,iv_loc_m,itor,ij) = rhs(ic_loc_m,iv_loc_m,itor,ij)+psi_mul*my_psi
        enddo
      enddo
    enddo
  enddo

  endif ! if nsplitB>0

  call timer_lib_out('nl')

end subroutine cgyro_nl_fftw_comm1_r64


subroutine cgyro_nl_fftw_comm1_r32(ij)
  use timer_lib
  use parallel_lib
  use cgyro_globals

  implicit none

  !-----------------------------------
  integer, intent(in) :: ij
  !-----------------------------------

  integer :: ir,it,iv_loc_m,ic_loc_m,itor
  integer :: iexch0,itor0,isplit0,iexch_base
  complex(KIND=REAL32) :: my_psi
  real(KIND=REAL32) :: psi_mul

  call timer_lib_in('nl_comm')
  call parallel_slib_r_nc32_wait(nsplitA,fA_nl32,fpackA32,fA_req)
  fA_req_valid = .FALSE.
  if (nsplitB > 0) then
    ! no major compute to overlap
    call parallel_slib_r_nc32_wait(nsplitB,fB_nl32,fpackB32,fB_req)
    fB_req_valid = .FALSE.
  endif
  call timer_lib_out('nl_comm')

  call timer_lib_in('nl')

  psi_mul = (q*rho/rmin)*(2*pi/length)

  if (nsplitB > 0) then

#if defined(OMPGPU)
!$omp target teams distribute parallel do simd collapse(4) &
!$omp&         private(iexch0,itor0,isplit0,iexch_base) &
!$omp&         private(ic_loc_m,my_psi) firstprivate(px_zero)
#elif defined(_OPENACC)
!$acc parallel loop collapse(4) gang vector independent private(ic_loc_m,my_psi) &
!$acc&         private(iexch0,itor0,isplit0,iexch_base) firstprivate(px_zero) &
!$acc&         present(ic_c,rhs,fpackA32,fpackB32) copyin(psi_mul,nl_min) &
!$acc&         present(nt1,nt2,nv_loc,n_theta,n_radial,nsplit,nsplitA,nsplitB) copyin(ij) default(none)
#else
!$omp parallel do collapse(2) private(ic_loc_m,my_psi) &
!$omp&         private(iexch0,itor0,isplit0,iexch_base) firstprivate(px_zero)
#endif
  do itor=nt1,nt2
    do iv_loc_m=1,nv_loc
      do it=1,n_theta
        do ir=1,n_radial
           ic_loc_m = ic_c(ir,it)
           if ( (itor == 0) .and.  ((ir == 1) .or. (ir == px_zero)) ) then
              ! filter
              my_psi = (0.0,0.0)
           else
              iexch0 = (iv_loc_m-1) + (it-1)*nv_loc
              itor0 = iexch0/nsplit
              isplit0 = modulo(iexch0,nsplit)
              if (isplit0 < nsplitA) then
                 iexch_base = 1+itor0*nsplitA
                 my_psi = fpackA32(ir,itor-nt1+1,iexch_base+isplit0)
              else
                 iexch_base = 1+itor0*nsplitB
                 my_psi = fpackB32(ir,itor-nt1+1,iexch_base+(isplit0-nsplitA))
              endif
           endif           
           if (itor < nl_min) then
              my_psi = (0.0,0.0)
           endif
           
           ! RHS -> -[f,g] = [f,g]_{r,-alpha}
           rhs(ic_loc_m,iv_loc_m,itor,ij) = rhs(ic_loc_m,iv_loc_m,itor,ij)+psi_mul*my_psi
        enddo
      enddo
    enddo
  enddo

  else ! nsplitB==0

#if defined(OMPGPU)
!$omp target teams distribute parallel do simd collapse(4) &
!$omp&         private(iexch0,itor0,isplit0,iexch_base) &
!$omp&         private(ic_loc_m,my_psi) firstprivate(px_zero)
#elif defined(_OPENACC)
!$acc parallel loop collapse(4) gang vector independent private(ic_loc_m,my_psi) &
!$acc&         private(iexch0,itor0,isplit0,iexch_base) firstprivate(px_zero) &
!$acc&         present(ic_c,rhs,fpackA32) copyin(psi_mul,nl_min) &
!$acc&         present(nt1,nt2,nv_loc,n_theta,n_radial,nsplit,nsplitA) copyin(ij) default(none)
#else
!$omp parallel do collapse(2) private(ic_loc_m,my_psi) &
!$omp&         private(iexch0,itor0,isplit0,iexch_base) firstprivate(px_zero)
#endif
  do itor=nt1,nt2
    do iv_loc_m=1,nv_loc
      do it=1,n_theta
        do ir=1,n_radial
           ic_loc_m = ic_c(ir,it)
           if ( (itor == 0) .and.  ((ir == 1) .or. (ir == px_zero)) ) then
              ! filter
              my_psi = (0.0,0.0)
           else
              iexch0 = (iv_loc_m-1) + (it-1)*nv_loc
              itor0 = iexch0/nsplit
              isplit0 = modulo(iexch0,nsplit)
              iexch_base = 1+itor0*nsplitA
              my_psi = fpackA32(ir,itor-nt1+1,iexch_base+isplit0)
           endif           
           if (itor < nl_min) then
              my_psi = (0.0,0.0)
           endif
           
           ! RHS -> -[f,g] = [f,g]_{r,-alpha}
           rhs(ic_loc_m,iv_loc_m,itor,ij) = rhs(ic_loc_m,iv_loc_m,itor,ij)+psi_mul*my_psi
        enddo
      enddo
    enddo
  enddo

  endif ! if nsplitB>0

  call timer_lib_out('nl')

end subroutine cgyro_nl_fftw_comm1_r32

subroutine cgyro_nl_fftw_comm1_r(ij)
  use cgyro_globals

  implicit none

  !-----------------------------------
  integer, intent(in) :: ij
  !-----------------------------------

  if (nl_single_flag .EQ. 0) then
    call cgyro_nl_fftw_comm1_r64(ij)
  else
    call cgyro_nl_fftw_comm1_r32(ij)
  endif

end subroutine cgyro_nl_fftw_comm1_r

!
! Comm2 is a transpose
! Reminder: nc ~= n_radial*n_theta
! First half of the transpose is done locally with sub-sampling
!  from (n_field,n_theta,n_radial,nt_loc) -> (n_field,n_radial,n_jtheta,nt_loc,n_toroidal_procs)
! Then AlltoAll finishes the transpose
!  (n_field,n_radial,n_jtheta,nt_loc,n_toroidal_proc)xn_toroidal_proc -> (n_field,n_radial,n_jtheta,nt_loc,n_toroida_procl)xn_toroidal_proc
! 

subroutine cgyro_nl_fftw_comm2_f64_async
  use timer_lib
  use parallel_lib
  use cgyro_globals

  implicit none

  integer :: ir,it,it_loc,itm,itl,itf
  integer :: itor,mytor
  integer :: iltheta_min
  complex :: gval

  call timer_lib_in('nl_mem')

  call field_dealias

#if defined(OMPGPU)
!$omp target teams distribute parallel do collapse(5) &
!$omp&         private(itor,it,iltheta_min,mytor,gval)
#elif defined(_OPENACC)
!$acc parallel loop gang vector collapse(5) independent &
!$acc&         private(itor,it,iltheta_min,mytor,gval) &
!$acc&         present(outraw_dealias,gpack) &
!$acc&         present(n_toroidal_procs,nt_loc,n_jtheta,nv_loc,nt1) &
!$acc&         present(n_theta,n_radial,n_field,nsplit) &
!$acc&         default(none)
#else
!$omp parallel do collapse(3) &
!$omp&         private(it_loc,itor,mytor,it,ir,iltheta_min,gval)
#endif
  do itm=1,n_toroidal_procs
   do itl=1,nt_loc
    do it_loc=1,n_jtheta
     do ir=1,n_radial
      do itf=1,n_field
       iltheta_min = 1+((itm-1)*nsplit)/nv_loc
       it = it_loc+iltheta_min-1
       itor = itl+(itm-1)*nt_loc
       gval = (0.0,0.0)
       if (it <= n_theta) then
         mytor = nt1+itl-1
         gval = outraw_dealias(ir,it,itf,mytor)
       endif
       ! else just padding
       gpack(itf,ir,it_loc,itor) = gval
      enddo
     enddo
    enddo
   enddo
  enddo

  call timer_lib_out('nl_mem')

  call timer_lib_in('nl_comm')
  call parallel_slib_f_fd_async(n_field,n_radial,n_jtheta,gpack,g_nl,g_req)
  g_req_valid = .TRUE.

  call timer_lib_out('nl_comm')

end subroutine cgyro_nl_fftw_comm2_f64_async

subroutine cgyro_nl_fftw_comm2_f32_async
  use timer_lib
  use parallel_lib
  use cgyro_globals

  implicit none

  integer :: ir,it,it_loc,itm,itl,itf
  integer :: itor,mytor
  integer :: iltheta_min
  complex(KIND=REAL32) :: gval

  call timer_lib_in('nl_mem')

  call field_dealias

#if defined(OMPGPU)
!$omp target teams distribute parallel do collapse(5) &
!$omp&         private(itor,it,iltheta_min,mytor,gval)
#elif defined(_OPENACC)
!$acc parallel loop gang vector collapse(5) independent &
!$acc&         private(itor,it,iltheta_min,mytor,gval) &
!$acc&         present(outraw_dealias,gpack32) &
!$acc&         present(n_toroidal_procs,nt_loc,n_jtheta,nv_loc,nt1) &
!$acc&         present(n_theta,n_radial,n_field,nsplit) &
!$acc&         default(none)
#else
!$omp parallel do collapse(3) &
!$omp&         private(it_loc,itor,mytor,it,ir,iltheta_min,gval)
#endif
  do itm=1,n_toroidal_procs
   do itl=1,nt_loc
    do it_loc=1,n_jtheta
     do ir=1,n_radial
      do itf=1,n_field
       iltheta_min = 1+((itm-1)*nsplit)/nv_loc
       it = it_loc+iltheta_min-1
       itor = itl+(itm-1)*nt_loc
       gval = (0.0,0.0)
       if (it <= n_theta) then
         mytor = nt1+itl-1
         gval = outraw_dealias(ir,it,itf,mytor)
       endif
       ! else just padding
       gpack32(itf,ir,it_loc,itor) = gval
      enddo
     enddo
    enddo
   enddo
  enddo

  call timer_lib_out('nl_mem')

  call timer_lib_in('nl_comm')
  call parallel_slib_f_fd32_async(n_field,n_radial,n_jtheta,gpack32,g_nl32,g_req)
  g_req_valid = .TRUE.

  call timer_lib_out('nl_comm')

end subroutine cgyro_nl_fftw_comm2_f32_async

subroutine cgyro_nl_fftw_comm2_async

  use cgyro_globals

  implicit none
  !-----------------------------------

  if (nl_single_flag > 1) then
    call cgyro_nl_fftw_comm2_f32_async
  else
    call cgyro_nl_fftw_comm2_f64_async
  endif
end subroutine cgyro_nl_fftw_comm2_async

end module cgyro_nl_comm

