#include "LAYERS_OPTIONS.h"
#ifdef ALLOW_GMREDI
#include "GMREDI_OPTIONS.h"
#endif

C--  File layers_fluxcalc.F:
C--   Contents
C--   o LAYERS_FLUXCALC
C--   o LAYERS_TRACERFLUXCALC
C--   o LAYERS_DIAPYCNAL
C--   o LAYERS_LOCATE

CBOP 0
C     !ROUTINE: LAYERS_FLUXCALC
C     !INTERFACE:
      SUBROUTINE LAYERS_FLUXCALC(
     I                  uVel,vVel,tracer,iLa,
     O                  UH,VH,Hw,Hs,PIw,PIs,Uw,Vs,
     I                  myThid )

C     !DESCRIPTION: \bv
C     *==========================================================*
C     | SUBROUTINE LAYERS_FLUXCALC
C     | Calculate the transport in isotracer layers, for a chosen
C     | tracer. This is the meat of the LAYERS package.
C     *==========================================================*
C     \ev

C !USES:
      IMPLICIT NONE
C     == Global variables ===
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "LAYERS_SIZE.h"
#include "LAYERS.h"
#ifdef ALLOW_GMREDI
# include "GMREDI.h"
#endif

C !INPUT PARAMETERS:
C     myThid    :: my Thread Id number
C     uVel  :: zonal velocity (m/s, i=1 held at western face)
C     vVel  :: meridional velocity (m/s, j=1 held at southern face)
C     tracer :: potential temperature, salt or potential density prho
C      UH   :: U integrated over layer (m^2/s)
C      VH   :: V integrated over layer (m^2/s)
C      Hw   :: Layer thickness at the U point (m)
C      Hs   :: Layer thickness at the V point (m)
C      PIw  :: 1 if layer exists, 0 otherwise (at U point)
C      PIs  :: 1 if layer exists, 0 otherwise (at V point)
C      Uw   :: average U over layer (m/s)
C      Vs   :: average V over layer (m/s)
C      iLa  :: layer coordinate index
      INTEGER myThid
      _RL uVel   (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,     nSx,nSy)
      _RL vVel   (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,     nSx,nSy)
      _RL tracer (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,     nSx,nSy)
#ifdef LAYERS_UFLUX
      _RL UH     (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers,nSx,nSy)
# ifdef LAYERS_THICKNESS
      _RL Hw     (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers,nSx,nSy)
      _RL PIw    (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers,nSx,nSy)
      _RL Uw     (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers,nSx,nSy)
# else
      _RL Hw(1), PIw(1), Uw(1)
# endif
#else
      _RL UH(1), Hw(1), PIw(1), Uw(1)
#endif
#ifdef LAYERS_VFLUX
      _RL VH     (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers,nSx,nSy)
# ifdef LAYERS_THICKNESS
      _RL Hs     (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers,nSx,nSy)
      _RL PIs    (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers,nSx,nSy)
      _RL Vs     (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers,nSx,nSy)
# else
      _RL Hs(1), PIs(1), Vs(1)
# endif
#else
      _RL VH(1), Hs(1), PIs(1), Vs(1)
#endif
      INTEGER iLa
CEOP

#ifdef ALLOW_LAYERS

C !LOCAL VARIABLES:
C     bi, bj   :: tile indices
C     i,j      :: horizontal indices
C     k        :: vertical index for model grid
C     kci      :: index from CellIndex
C     kg       :: index for looping though layers_bounds
C     kk       :: vertical index for ZZ (fine) grid
C     kgu,kgv  :: vertical index for isopycnal grid
C     kloc     :: local copy of kgu/v to reduce accesses to index arrays
C     mSteps   :: maximum number of steps for bisection method
C     prho     :: pot. density (less 1000) referenced to layers_krho pressure
C     TatU     :: temperature at U point
C     TatV     :: temperature at V point
C     dzfac    :: temporary sublayer thickness
C     Tloc,Tp1 :: horizontally interpolated tracer values

      INTEGER bi, bj
      INTEGER i,j,k,kk,kg,kci,kp1,kloc
      INTEGER mSteps
      INTEGER kgu(sNx+1,sNy+1), kgv(sNx+1,sNy+1)
      _RL TatU(sNx+1,sNy+1), TatV(sNx+1,sNy+1)
      _RL dzfac
#ifdef ALLOW_GMREDI
      INTEGER kcip1
      _RL delPsi, maskp1
#endif
      LOGICAL errorFlag
      CHARACTER*(MAX_LEN_MBUF) msgBuf

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|

C     compute maximum number of steps for bisection method (approx.
C     log2(Nlayers)) as log2(Nlayers) + 1 for safety
      mSteps = int(log10(dble(Nlayers))/log10(2. _d 0))+1

C --- The tile loops
      DO bj=myByLo(myThid),myByHi(myThid)
      DO bi=myBxLo(myThid),myBxHi(myThid)

C     Initialize the search indices
      DO j = 1,sNy+1
        DO i = 1,sNx+1
C       The temperature index (layer_G) goes from cold to warm.
C       The water column goes from warm (k=1) to cold (k=Nr).
C       So initialize the search with the warmest value.
          kgu(i,j) = Nlayers
          kgv(i,j) = Nlayers
        ENDDO
      ENDDO

C     Reset the arrays
      DO kg=1,Nlayers
       DO j = 1-OLy,sNy+OLy
        DO i = 1-OLx,sNx+OLx
#ifdef LAYERS_UFLUX
         UH (i,j,kg,bi,bj) = 0. _d 0
#ifdef LAYERS_THICKNESS
         Hw(i,j,kg,bi,bj) = 0. _d 0
         PIw(i,j,kg,bi,bj) = 0. _d 0
         Uw(i,j,kg,bi,bj) = 0. _d 0
#endif /* LAYERS_THICKNESS */
#endif /* UH */
#ifdef LAYERS_VFLUX
         VH (i,j,kg,bi,bj) = 0. _d 0
#ifdef LAYERS_THICKNESS
         Hs(i,j,kg,bi,bj) = 0. _d 0
         PIs(i,j,kg,bi,bj) = 0. _d 0
         Vs(i,j,kg,bi,bj) = 0. _d 0
#endif /* LAYERS_THICKNESS */
#endif /* VH */
        ENDDO
       ENDDO
      ENDDO

      DO kk=1,NZZ
       k = MapIndex(kk)
       kci = CellIndex(kk)

#ifdef LAYERS_UFLUX
       DO j = 1,sNy+1
        DO i = 1,sNx+1

C ------ Find theta at the U point (west) on the fine Z grid
         kp1=k+1
         IF (maskW(i,j,kp1,bi,bj).EQ.zeroRS) kp1=k
         TatU(i,j) = MapFact(kk) *
     &    0.5 _d 0 * (tracer(i-1,j,k,bi,bj)+tracer(i,j,k,bi,bj)) +
     &    (1. _d 0 -MapFact(kk)) *
     &    0.5 _d 0 * (tracer(i-1,j,kp1,bi,bj)+tracer(i,j,kp1,bi,bj))

        ENDDO
       ENDDO
C ------ Now that we know T everywhere, determine the binning.
C        find the layer indices kgu
       CALL LAYERS_LOCATE(
     I      layers_bounds(1,iLa),Nlayers,mSteps,sNx,sNy,TatU,
     O      kgu,
     I      myThid )
#ifndef TARGET_NEC_SX
C     check for failures
       IF ( debugLevel .GE. debLevC ) THEN
        errorFlag = .FALSE.
        DO j = 1,sNy+1
         DO i = 1,sNx+1
          IF ( kgu(i,j) .LE. 0 ) THEN
           WRITE(msgBuf,'(2A,I3,A,I3,A,1E14.6)')
     &          'S/R LAYERS_LOCATE: Could not find a bin in ',
     &          'layers_bounds for TatU(',i,',',j,',)=',TatU(i,j)
           CALL PRINT_ERROR( msgBuf, myThid )
           errorFlag = .TRUE.
          ENDIF
         ENDDO
        ENDDO
        IF ( errorFlag ) STOP 'ABNORMAL END: S/R LAYERS_FLUXCALC'
       ENDIF
#endif /* ndef TARGET_NEC_SX */
C
       DO j = 1,sNy+1
        DO i = 1,sNx+1

         kloc = kgu(i,j)
         dzfac = dZZf(kk) * hFacW(i,j,kci,bi,bj)

C ------ Augment the bin values
         UH(i,j,kloc,bi,bj) =
     &    UH(i,j,kloc,bi,bj) +
     &    dzfac * uVel(i,j,kci,bi,bj)

#ifdef ALLOW_GMREDI
         IF ( layers_bolus(iLa)  ) THEN
           IF ( .NOT.GM_AdvForm ) THEN
             delPsi = 0.25 _d 0 *(
     &              ( rA(i-1,j,bi,bj)*Kwx(i-1,j,kcip1,bi,bj)
     &               +rA( i ,j,bi,bj)*Kwx( i ,j,kcip1,bi,bj)
     &              ) * maskW(i,j,kcip1,bi,bj) * maskp1
     &            - ( rA(i-1,j,bi,bj)*Kwx(i-1,j, kci ,bi,bj)
     &               +rA( i ,j,bi,bj)*Kwx( i ,j, kci ,bi,bj)
     &              ) * maskW(i,j, kci ,bi,bj)
     &                           ) * recip_rAw(i,j,bi,bj)
#ifdef GM_BOLUS_ADVEC
           ELSE
             delPsi = GM_PsiX(i,j,kcip1,bi,bj)*maskp1
     &              - GM_PsiX(i,j, kci, bi,bj)
#endif
           ENDIF
           UH(i,j,kloc,bi,bj) = UH(i,j,kloc,bi,bj)
     &      + delPsi*recip_drF(kci)*_recip_hFacW(i,j,kci,bi,bj)
     &      * dzfac
         ENDIF
#endif /* ALLOW_GMREDI */

#ifdef LAYERS_THICKNESS
         Hw(i,j,kloc,bi,bj) = Hw(i,j,kloc,bi,bj) + dzfac
#endif /* LAYERS_THICKNESS */

        ENDDO
       ENDDO
#endif /* LAYERS_UFLUX */

#ifdef LAYERS_VFLUX
       DO j = 1,sNy+1
        DO i = 1,sNx+1
C ------ Find theta at the V point (south) on the fine Z grid
         kp1=k+1
         IF (maskS(i,j,kp1,bi,bj).EQ.zeroRS) kp1=k
         TatV(i,j) = MapFact(kk) *
     &    0.5 _d 0 * (tracer(i,j-1,k,bi,bj)+tracer(i,j,k,bi,bj)) +
     &    (1. _d 0 -MapFact(kk)) *
     &    0.5 _d 0 * (tracer(i,j-1,kp1,bi,bj)+tracer(i,j,kp1,bi,bj))

        ENDDO
       ENDDO
C ------ Now that we know T everywhere, determine the binning.
C        find the layer indices kgv
       CALL LAYERS_LOCATE(
     I      layers_bounds(1,iLa),Nlayers,mSteps,sNx,sNy,TatV,
     O      kgv,
     I      myThid )
#ifndef TARGET_NEC_SX
       IF ( debugLevel .GE. debLevC ) THEN
C     check for failures
        errorFlag = .FALSE.
        DO j = 1,sNy+1
         DO i = 1,sNx+1
          IF ( kgv(i,j) .LE. 0 ) THEN
           WRITE(msgBuf,'(2A,I3,A,I3,A,1E14.6)')
     &          'S/R LAYERS_LOCATE: Could not find a bin in ',
     &          'layers_bounds for TatV(',i,',',j,',)=',TatV(i,j)
           CALL PRINT_ERROR( msgBuf, myThid )
           errorFlag = .TRUE.
          ENDIF
         ENDDO
        ENDDO
        IF ( errorFlag ) STOP 'ABNORMAL END: S/R LAYERS_FLUXCALC'
       ENDIF
#endif /* ndef TARGET_NEC_SX */
C
       DO j = 1,sNy+1
        DO i = 1,sNx+1

         kloc = kgv(i,j)
         dzfac = dZZf(kk) * hFacS(i,j,kci,bi,bj)

C ------ debugging stuff
C         IF (i.EQ.10 .AND. j.EQ.10) THEN
C           WRITE(msgBuf,'(A,I3,A,F10.2,A,F6.2)')
C     &          '    kloc=', kloc,
C     &          ', TatV=',TatV(i,j),
C     &          ', dzfac=',dzfac
C           CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
C     &                         SQUEEZE_RIGHT, myThid )
C         ENDIF

C ------ Augment the bin values
         VH(i,j,kloc,bi,bj) =
     &    VH(i,j,kloc,bi,bj) + dzfac * vVel(i,j,kci,bi,bj)



#ifdef LAYERS_THICKNESS
         Hs(i,j,kloc,bi,bj) = Hs(i,j,kloc,bi,bj) + dzfac
#endif /* LAYERS_THICKNESS */

        ENDDO
       ENDDO
#endif /* LAYERS_VFLUX */
      ENDDO

C--   Now that we know the thicknesses, compute the heaviside function
C--   (Needs another loop through Ng)
#ifdef LAYERS_THICKNESS
      DO kg=1,Nlayers
       DO j = 1,sNy+1
        DO i = 1,sNx+1
#ifdef LAYERS_UFLUX
         IF (Hw(i,j,kg,bi,bj) .GT. 0.) THEN
          PIw(i,j,kg,bi,bj) = 1. _d 0
          Uw(i,j,kg,bi,bj) =
     &        UH(i,j,kg,bi,bj) / Hw(i,j,kg,bi,bj)
         ENDIF
#endif /* LAYERS_UFLUX */
#ifdef LAYERS_VFLUX
         IF (Hs(i,j,kg,bi,bj) .GT. 0.) THEN
          PIs(i,j,kg,bi,bj) = 1. _d 0
          Vs(i,j,kg,bi,bj) =
     &        VH(i,j,kg,bi,bj) / Hs(i,j,kg,bi,bj)
         ENDIF
#endif /* LAYERS_VFLUX */
        ENDDO
       ENDDO
      ENDDO
#endif /* LAYERS_THICKNESS */

C --- End bi,bj loop
      ENDDO
      ENDDO

      RETURN
      END

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|

C - holding location for tracerflux calc

C     !ROUTINE: LAYERS_TRACERFLUXCALC
C     !INTERFACE:
      SUBROUTINE LAYERS_TRACERFLUXCALC(
     I                  uFlux,vFlux,tracer,iLa, Trac_id,
     O                  UHTRACER,VHTRACER,
     I                  myThid )
C     !DESCRIPTION: \bv
C     *==========================================================*
C     | SUBROUTINE LAYERS_TRACERFLUXCALC
C     | Calculate the transport of a tracer in isotracer layers,
C     | for two tracers. Can be different tracers.
C     *==========================================================*
C     \ev

C !USES:
      IMPLICIT NONE
C     == Global variables ===
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "LAYERS_SIZE.h"
#include "LAYERS.h"


C !INPUT PARAMETERS:
C     myThid    :: my Thread Id number
C     uFlux  :: zonal flux (TRACER m**3/s, i=1 held at western face)
C     vFlux  :: meridional flux (TRACER m***3/s, j=1 held at southern face)
C     tracer :: potential temperature, salt or potential density prho
C      UHTRACER   :: zonal flux integrated over layer (TRACER m^3/s)
C      VHTRACER   :: zonal integrated over layer (TRACER m^3/s)
C      iLa  :: layer coordinate index
C     Trac_id :: which tracer is is it? (1=theta, 2 = salt)
      INTEGER myThid
      _RL uFlux   (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,2, nSx,nSy)
      _RL vFlux   (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,2, nSx,nSy)
      _RL tracer (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,  nSx,nSy)
#ifdef LAYERS_UFLUX
      _RL UHTRACER     (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers,nSx,nSy)
#else
      _RL UHTRACER(1)
#endif
#ifdef LAYERS_VFLUX
      _RL VHTRACER     (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers,nSx,nSy)
#else
      _RL VHTRACER(1)
#endif
      INTEGER iLa, Trac_id
CEOP

#ifdef LAYERS_TRACERFLUX

C !LOCAL VARIABLES:
C     bi, bj   :: tile indices
C     i,j      :: horizontal indices
C     k        :: vertical index for model grid
C     kci      :: index from CellIndex
C     kg       :: index for looping though layers_bounds
C     kk       :: vertical index for ZZ (fine) grid
C     kgu,kgv  :: vertical index for isopycnal grid
C     kloc     :: local copy of kgu/v to reduce accesses to index arrays
C     mSteps   :: maximum number of steps for bisection method
C     prho     :: pot. density (less 1000) referenced to layers_krho pressure
C     TatU     :: TRACER at U point
C     TatV     :: TRACER at V point
C     dzfac    :: temporary sublayer thickness
C     Tloc,Tp1 :: horizontally interpolated tracer values

      INTEGER bi, bj
      INTEGER i,j,k,kk,kg,kci,kp1,kloc
      INTEGER mSteps
      INTEGER kgu(sNx+1,sNy+1), kgv(sNx+1,sNy+1)
      _RL TatU(sNx+1,sNy+1), TatV(sNx+1,sNy+1)
      _RL dzfac
      LOGICAL errorFlag
      CHARACTER*(MAX_LEN_MBUF) msgBuf

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|

C     compute maximum number of steps for bisection method (approx.
C     log2(Nlayers)) as log2(Nlayers) + 1 for safety
      mSteps = int(log10(dble(Nlayers))/log10(2. _d 0))+1

C --- The tile loops
      DO bj=myByLo(myThid),myByHi(myThid)
      DO bi=myBxLo(myThid),myBxHi(myThid)

C     Initialize the search indices
      DO j = 1,sNy+1
        DO i = 1,sNx+1
C       The temperature index (layer_G) goes from cold to warm.
C       The water column goes from warm (k=1) to cold (k=Nr).
C       So initialize the search with the warmest value.
          kgu(i,j) = Nlayers
          kgv(i,j) = Nlayers
        ENDDO
      ENDDO

C     Reset the arrays
      DO kg=1,Nlayers
       DO j = 1-OLy,sNy+OLy
        DO i = 1-OLx,sNx+OLx
#ifdef LAYERS_UFLUX
         UHTRACER (i,j,kg,bi,bj) = 0. _d 0
#endif /* UH */
#ifdef LAYERS_VFLUX
         VHTRACER (i,j,kg,bi,bj) = 0. _d 0
#endif /* VH */
        ENDDO
       ENDDO
      ENDDO


      DO kk=1,NZZ
       k = MapIndex(kk)
       kci = CellIndex(kk)

#ifdef LAYERS_UFLUX
       DO j = 1,sNy+1
        DO i = 1,sNx+1

C ------ Find tracer at the U point (west) on the fine Z grid
         kp1=k+1
         IF (maskW(i,j,kp1,bi,bj).EQ.zeroRS) kp1=k
         TatU(i,j) = MapFact(kk) *
     &    0.5 _d 0 * (tracer(i-1,j,k,bi,bj)+tracer(i,j,k,bi,bj)) +
     &    (1. _d 0 -MapFact(kk)) *
     &    0.5 _d 0 * (tracer(i-1,j,kp1,bi,bj)+tracer(i,j,kp1,bi,bj))

        ENDDO
       ENDDO
C ------ Now that we know tracer everywhere, determine the binning.
C        find the layer indices kgu
       CALL LAYERS_LOCATE(
     I      layers_bounds(1,iLa),Nlayers,mSteps,sNx,sNy,TatU,
     O      kgu,
     I      myThid )
#ifndef TARGET_NEC_SX
C     check for failures
       IF ( debugLevel .GE. debLevC ) THEN
        errorFlag = .FALSE.
        DO j = 1,sNy+1
         DO i = 1,sNx+1
          IF ( kgu(i,j) .LE. 0 ) THEN
           WRITE(msgBuf,'(2A,I3,A,I3,A,1E14.6)')
     &          'S/R LAYERS_LOCATE: Could not find a bin in ',
     &          'layers_bounds for TatU(',i,',',j,',)=',TatU(i,j)
           CALL PRINT_ERROR( msgBuf, myThid )
           errorFlag = .TRUE.
          ENDIF
         ENDDO
        ENDDO
        IF ( errorFlag ) STOP 'ABNORMAL END: S/R LAYERS_FLUXCALC'
       ENDIF
#endif /* ndef TARGET_NEC_SX */
C
       DO j = 1,sNy+1
        DO i = 1,sNx+1

         kloc = kgu(i,j)
         dzfac = dZZf(kk) * hFacW(i,j,kci,bi,bj)

C ------ Augment the bin values
         UHTRACER(i,j,kloc,bi,bj) =
     &    UHTRACER(i,j,kloc,bi,bj) +
C ------ Multiply by recip_drF(k) to normalise, so sum is conserved.
     &    dzfac * uFlux(i,j,kci,Trac_id, bi,bj)  * recip_drF(kci)

        ENDDO
       ENDDO
#endif /* LAYERS_UFLUX */

#ifdef LAYERS_VFLUX
       DO j = 1,sNy+1
        DO i = 1,sNx+1
C ------ Find tracer at the V point (south) on the fine Z grid
         kp1=k+1
         IF (maskS(i,j,kp1,bi,bj).EQ.zeroRS) kp1=k
         TatV(i,j) = MapFact(kk) *
     &    0.5 _d 0 * (tracer(i,j-1,k,bi,bj)+tracer(i,j,k,bi,bj)) +
     &    (1. _d 0 -MapFact(kk)) *
     &    0.5 _d 0 * (tracer(i,j-1,kp1,bi,bj)+tracer(i,j,kp1,bi,bj))

        ENDDO
       ENDDO
C ------ Now that we know T everywhere, determine the binning.
C        find the layer indices kgv
       CALL LAYERS_LOCATE(
     I      layers_bounds(1,iLa),Nlayers,mSteps,sNx,sNy,TatV,
     O      kgv,
     I      myThid )
#ifndef TARGET_NEC_SX
       IF ( debugLevel .GE. debLevC ) THEN
C     check for failures
        errorFlag = .FALSE.
        DO j = 1,sNy+1
         DO i = 1,sNx+1
          IF ( kgv(i,j) .LE. 0 ) THEN
           WRITE(msgBuf,'(2A,I3,A,I3,A,1E14.6)')
     &          'S/R LAYERS_LOCATE: Could not find a bin in ',
     &          'layers_bounds for TatV(',i,',',j,',)=',TatV(i,j)
           CALL PRINT_ERROR( msgBuf, myThid )
           errorFlag = .TRUE.
          ENDIF
         ENDDO
        ENDDO
        IF ( errorFlag ) STOP 'ABNORMAL END: S/R LAYERS_FLUXCALC'
       ENDIF
#endif /* ndef TARGET_NEC_SX */
C
       DO j = 1,sNy+1
        DO i = 1,sNx+1

         kloc = kgv(i,j)
         dzfac = dZZf(kk) * hFacS(i,j,kci,bi,bj)

C ------ debugging stuff
C         IF (i.EQ.10 .AND. j.EQ.10) THEN
C           WRITE(msgBuf,'(A,I3,A,F10.2,A,F6.2)')
C     &          '    kloc=', kloc,
C     &          ', TatV=',TatV(i,j),
C     &          ', dzfac=',dzfac
C           CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
C     &                         SQUEEZE_RIGHT, myThid )
C         ENDIF

C ------ Augment the bin values
         VHTRACER(i,j,kloc,bi,bj) =
     &    VHTRACER(i,j,kloc,bi,bj) +
     &     dzfac * vFlux(i,j,kci,Trac_id, bi,bj) * recip_drF(kci)

        ENDDO
       ENDDO
#endif /* LAYERS_VFLUX */
      ENDDO

C --- End bi,bj loop
      ENDDO
      ENDDO

#endif /* LAYERS_TRACERFLUX */


      RETURN
      END


C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|


CBOP 0
C     !ROUTINE: LAYERS_DIAPYCNAL
C     !INTERFACE:
      SUBROUTINE LAYERS_DIAPYCNAL(
     I                  tracer,iLa,
     O                  TtendSurf, TtendDiffh, TtendDiffr,
     O                  TtendAdvh, TtendAdvr, Ttendtot,
     O                  StendSurf, StendDiffh, StendDiffr,
     O                  StendAdvh, StendAdvr, Stendtot,
     O                  Hc, PIc,
     I                  myThid )

C     !DESCRIPTION: \bv
C     *==========================================================*
C     | SUBROUTINE LAYERS_DIAPYCNAL
C     | Calculate the diapycnal velocity in isotracer layers, for a chosen
C     | tracer.
C     *==========================================================*
C     \ev
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#include "LAYERS_SIZE.h"
#include "LAYERS.h"

C !INPUT PARAMETERS:
C     myThid    :: my Thread Id number
C     tracer    :: potential temperature, salt or potential density prho
C     iLa       :: layer coordinate index
C     TtendSurf :: temperature tendency due to surface forcing times thickness
C     TtendDiffh:: temperature tendency due to horiz. diffusion times thickness
C     TtendDiffr:: temperature tendency due to vert. diffusion times thickness
C     TtendAdvh:: salinity tendency due to horiz. advection times thickness
C     TtendAdvr:: salinity tendency due to vert. advection times thickness
C     StendSurf :: salinity tendency due to surface forcing times thickness
C     StendDiffh:: salinity tendency due to horiz. diffusion times thickness
C     StendDiffr:: salinity tendency due to vert. diffusion times thickness
C     StendAdvh :: salinity tendency due to horiz. advection times thickness
C     StendAdvr :: salinity tendency due to vert. advection times thickness
C     Hc        :: Layer thickness at the tracer point (m)
C     PIw       :: 1 if layer exists, 0 otherwise (at tracer point)
      INTEGER iLa, myThid
      _RL tracer    (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nr,     nSx,nSy)
      _RL Hc        (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers,nSx,nSy)
      _RL PIc       (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers,nSx,nSy)
      _RL TtendSurf (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers-1,nSx,nSy)
      _RL TtendDiffh(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers-1,nSx,nSy)
      _RL TtendDiffr(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers-1,nSx,nSy)
      _RL TtendAdvh (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers-1,nSx,nSy)
      _RL TtendAdvr (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers-1,nSx,nSy)
      _RL Ttendtot  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers-1,nSx,nSy)
      _RL StendSurf (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers-1,nSx,nSy)
      _RL StendDiffh(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers-1,nSx,nSy)
      _RL StendDiffr(1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers-1,nSx,nSy)
      _RL StendAdvh (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers-1,nSx,nSy)
      _RL StendAdvr (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers-1,nSx,nSy)
      _RL Stendtot  (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers-1,nSx,nSy)
CEOP

#ifdef LAYERS_THERMODYNAMICS

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|
C !LOCAL VARIABLES:
C     bi, bj   :: tile indices
C     i,j      :: horizontal indices
C     k        :: vertical index for model grid
C     kp1      :: vertical index for model grid next cell
C     kci      :: index from CellIndex
C     kg       :: index for looping though layers_bounds
C     kk       :: vertical index for ZZ (fine) grid
C     kloc     :: local copy of kgu/v to reduce accesses to index arrays
C     mSteps   :: maximum number of steps for bisection method
C     TatC     :: temperature at C point
      _RL Hcw       (1-OLx:sNx+OLx,1-OLy:sNy+OLy,Nlayers-1,nSx,nSy)
      INTEGER bi, bj
      INTEGER i,j,k,kk,kg,kci,kloc
      INTEGER mSteps
      INTEGER kgc(sNx+1,sNy+1)
      INTEGER kgcw(sNx+1,sNy+1)
      _RL TatC(sNx+1,sNy+1), dzfac, Tfac, Sfac
      LOGICAL errorFlag
      CHARACTER*(MAX_LEN_MBUF) msgBuf
#ifdef LAYERS_FINEGRID_DIAPYCNAL
      INTEGER kp1
#endif

C --  constants for T and S forcing, gets reset later for rho
      Tfac = 1. _d 0
      Sfac = 1. _d 0

C     compute maximum number of steps for bisection method (approx.
C     log2(Nlayers)) as log2(Nlayers) + 1 for safety
      mSteps = int(log10(dble(Nlayers))/log10(2. _d 0))+1

C      STOP 'DEBUG END: S/R LAYERS_DIAPYCNAL'

C --- The tile loops
      DO bj=myByLo(myThid),myByHi(myThid)
      DO bi=myBxLo(myThid),myBxHi(myThid)

C     Initialize the search indices
      DO j = 1,sNy+1
        DO i = 1,sNx+1
C       The temperature index (layer_G) goes from cold to warm.
C       The water column goes from warm (k=1) to cold (k=Nr).
C       So initialize the search with the warmest value.
          kgc(i,j) = Nlayers
          kgcw(i,j) = Nlayers-1
        ENDDO
      ENDDO

C     Reset the arrays
C --- These are at the w point
      DO kg=1,Nlayers-1
       DO j = 1-OLy,sNy+OLy
        DO i = 1-OLx,sNx+OLx
         TtendSurf (i,j,kg,bi,bj) = 0. _d 0
         TtendDiffh(i,j,kg,bi,bj) = 0. _d 0
         TtendDiffr(i,j,kg,bi,bj) = 0. _d 0
         TtendAdvh(i,j,kg,bi,bj)  = 0. _d 0
         TtendAdvr(i,j,kg,bi,bj)  = 0. _d 0
         Ttendtot(i,j,kg,bi,bj)   = 0. _d 0
         StendSurf (i,j,kg,bi,bj) = 0. _d 0
         StendDiffh(i,j,kg,bi,bj) = 0. _d 0
         StendDiffr(i,j,kg,bi,bj) = 0. _d 0
         StendAdvh(i,j,kg,bi,bj)  = 0. _d 0
         StendAdvr(i,j,kg,bi,bj)  = 0. _d 0
         Stendtot(i,j,kg,bi,bj)   = 0. _d 0
         Hcw(i,j,kg,bi,bj) = 0. _d 0
        ENDDO
       ENDDO
      ENDDO
C --- These are at the c point
      DO kg=1,Nlayers
       DO j = 1-OLy,sNy+OLy
        DO i = 1-OLx,sNx+OLx
         Hc(i,j,kg,bi,bj) = 0. _d 0
         PIc(i,j,kg,bi,bj) = 0. _d 0
        ENDDO
       ENDDO
      ENDDO

#ifdef LAYERS_FINEGRID_DIAPYCNAL
      DO kk=1,NZZ
       k = MapIndex(kk)
       kci = CellIndex(kk)
       DO j = 1,sNy+1
        DO i = 1,sNx+1
C ------ Find theta at the V point (south) on the fine Z grid
         kp1=k+1
         IF (maskC(i,j,kp1,bi,bj).EQ.zeroRS) kp1=k
         TatC(i,j) = MapFact(kk) * tracer(i,j,k,bi,bj) +
     &    (1. _d 0 -MapFact(kk)) * tracer(i,j,kp1,bi,bj)
        ENDDO
       ENDDO
#else
      DO kk=1,Nr
       k = kk
       kci = kk
       DO j = 1,sNy+1
        DO i = 1,sNx+1
         TatC(i,j) = tracer(i,j,k,bi,bj)
        ENDDO
       ENDDO
#endif /* LAYERS_FINEGRID_DIAPYCNAL */

C ------ debugging stuff
c         IF (i.EQ.38 .AND. j.EQ.4 .AND. bi.EQ.1 .AND. bj.EQ.1) THEN
c           i=38
c           j=4
c           WRITE(msgBuf,
c     &       '(A,I3,A,I3,A,I3,A,F7.2,A,F7.2,A,F7.2,A,F7.2,A,F3.1)')
c     &          'LAYERS_DEBUG: iLa=', iLa,
c     &          ', kk=', kk,
c     &          ', k=', k,
c     &          ', tracer=', tracer(i,j,k,bi,bj),
c     &          ', TatC=',TatC(i,j),
c     &          ', hFacC=',hFacC(i,j,k,bi,bj)
c           CALL PRINT_MESSAGE( msgBuf, standardMessageUnit,
c     &                         SQUEEZE_RIGHT, myThid )
c         ENDIF
C ------ Now that we know T everywhere, determine the binning.
C        find the layer indices kgc for the center point
       CALL LAYERS_LOCATE(
     I      layers_bounds(1,iLa),Nlayers,mSteps,sNx,sNy,TatC,
     O      kgc,
     I      myThid )
#ifndef TARGET_NEC_SX
C     check for failures
       IF ( debugLevel .GE. debLevC ) THEN
        errorFlag = .FALSE.
        DO j = 1,sNy+1
         DO i = 1,sNx+1
          IF ( kgc(i,j) .LE. 0 ) THEN
           WRITE(msgBuf,'(2A,I3,A,I3,A,1E14.6)')
     &          'S/R LAYERS_LOCATE: Could not find a bin in ',
     &          'layers_bounds for TatC(',i,',',j,',)=',TatC(i,j)
           CALL PRINT_ERROR( msgBuf, myThid )
           errorFlag = .TRUE.
          ENDIF
         ENDDO
        ENDDO
        IF ( errorFlag ) STOP 'ABNORMAL END: S/R LAYERS_DIAPYCNAL'
       ENDIF
#endif /* ndef TARGET_NEC_SX */

C        find the layer indices kgcw for the w point
       CALL LAYERS_LOCATE(
     I      layers_bounds_w(1,iLa),Nlayers-1,mSteps,sNx,sNy,TatC,
     O      kgcw,
     I      myThid )
#ifndef TARGET_NEC_SX
C     check for failures
       IF ( debugLevel .GE. debLevC ) THEN
        errorFlag = .FALSE.
        DO j = 1,sNy+1
         DO i = 1,sNx+1
          IF ( kgcw(i,j) .LE. 0 ) THEN
           WRITE(msgBuf,'(2A,I3,A,I3,A,1E14.6)')
     &          'S/R LAYERS_LOCATE: Could not find a bin in ',
     &          'layers_bounds for TatC(',i,',',j,',)=',TatC(i,j)
           CALL PRINT_ERROR( msgBuf, myThid )
           errorFlag = .TRUE.
          ENDIF
         ENDDO
        ENDDO
        IF ( errorFlag ) STOP 'ABNORMAL END: S/R LAYERS_DIAPYCNAL'
       ENDIF
#endif /* ndef TARGET_NEC_SX */

C ------ Augment the bin values
       DO j = 1,sNy+1
        DO i = 1,sNx+1
#ifdef LAYERS_FINEGRID_DIAPYCNAL
         dzfac = dZZf(kk) * hFacC(i,j,kci,bi,bj)
#else
         dzfac = dRf(kci) * hFacC(i,j,kci,bi,bj)
#endif /* LAYERS_FINEGRID_DIAPYCNAL */
         kloc = kgcw(i,j)

C ------- Thickness at w point
         Hcw(i,j,kloc,bi,bj) = Hcw(i,j,kloc,bi,bj)
     &     + dzfac
C ------- Thickness at c point
         Hc(i,j,kgc(i,j),bi,bj) = Hc(i,j,kgc(i,j),bi,bj)
     &     + dzfac

C ------- Now rescale dzfac to include the layer coordinate spacing
         dzfac = dzfac * layers_recip_delta(kloc,iLa)

#ifdef LAYERS_PRHO_REF 
         IF ( layers_num(iLa) .EQ. 3 ) THEN
           Tfac = layers_alpha(i,j,kci,bi,bj)
           Sfac = layers_beta(i,j,kci,bi,bj)
         ENDIF
#endif
         IF (kci.EQ.1) THEN
C ------- We are in the surface layer
          TtendSurf(i,j,kloc,bi,bj) =
     &     TtendSurf(i,j,kloc,bi,bj) +
     &     Tfac * dzfac * layers_surfflux(i,j,1,1,bi,bj)
          StendSurf(i,j,kloc,bi,bj) =
     &     StendSurf(i,j,kloc,bi,bj) +
     &     Sfac * dzfac * layers_surfflux(i,j,1,2,bi,bj)
         ENDIF

#ifdef SHORTWAVE_HEATING
          TtendSurf(i,j,kloc,bi,bj) =
     &     TtendSurf(i,j,kloc,bi,bj) +
     &     Tfac * dzfac * layers_sw(i,j,kci,1,bi,bj)
#endif /* SHORTWAVE_HEATING */

C ------- Diffusion
         TtendDiffh(i,j,kloc,bi,bj) =
     &     TtendDiffh(i,j,kloc,bi,bj) + dzfac * Tfac *
     &    (layers_dfx(i,j,kci,1,bi,bj)+
     &     layers_dfy(i,j,kci,1,bi,bj))
         TtendDiffr(i,j,kloc,bi,bj) =
     &     TtendDiffr(i,j,kloc,bi,bj) +
     &     dzfac * Tfac * layers_dfr(i,j,kci,1,bi,bj)
         StendDiffh(i,j,kloc,bi,bj) =
     &     StendDiffh(i,j,kloc,bi,bj) + dzfac * Sfac *
     &    (layers_dfx(i,j,kci,2,bi,bj)+
     &     layers_dfy(i,j,kci,2,bi,bj))
         StendDiffr(i,j,kloc,bi,bj) =
     &     StendDiffr(i,j,kloc,bi,bj) +
     &     dzfac * Sfac * layers_dfr(i,j,kci,2,bi,bj)
C ------- Advection
         TtendAdvh(i,j,kloc,bi,bj) =
     &     TtendAdvh(i,j,kloc,bi,bj) + dzfac * Tfac *
     &    (layers_afx(i,j,kci,1,bi,bj)+
     &     layers_afy(i,j,kci,1,bi,bj))
         TtendAdvr(i,j,kloc,bi,bj) =
     &     TtendAdvr(i,j,kloc,bi,bj) +
     &     dzfac * Tfac * layers_afr(i,j,kci,1,bi,bj)
         StendAdvh(i,j,kloc,bi,bj) =
     &     StendAdvh(i,j,kloc,bi,bj) + dzfac * Sfac *
     &    (layers_afx(i,j,kci,2,bi,bj)+
     &     layers_afy(i,j,kci,2,bi,bj))
         StendAdvr(i,j,kloc,bi,bj) =
     &     StendAdvr(i,j,kloc,bi,bj) +
     &     dzfac * Sfac * layers_afr(i,j,kci,2,bi,bj)
C -------- Total Tendency
         Ttendtot(i,j,kloc,bi,bj) =
     &     Ttendtot(i,j,kloc,bi,bj) +
     &     dzfac * Tfac * layers_tottend(i,j,kci,1,bi,bj)
         Stendtot(i,j,kloc,bi,bj) =
     &     Stendtot(i,j,kloc,bi,bj) +
     &     dzfac * Sfac * layers_tottend(i,j,kci,2,bi,bj)
        ENDDO
       ENDDO
      ENDDO

C--   Now that we know the thicknesses, compute the heaviside function
C--   (Needs another loop through Ng)
      DO kg=1,Nlayers
       DO j = 1,sNy+1
        DO i = 1,sNx+1
         IF (Hc(i,j,kg,bi,bj) .GT. 0.) THEN
          PIc(i,j,kg,bi,bj) = 1. _d 0
         ENDIF
        ENDDO
       ENDDO
      ENDDO

C --- End bi,bj loop
      ENDDO
      ENDDO

#endif /* LAYERS_THERMODYNAMICS */

      RETURN
      END

C---+----1----+----2----+----3----+----4----+----5----+----6----+----7-|--+----|

CBOP
      SUBROUTINE LAYERS_LOCATE(
     I                          xx,n,m,sNx,sNy,x,
     O                          k,
     I                          myThid )

C     !DESCRIPTION: \bv
C     *==========================================================*
C     | Find the index(-array) k such that x is between xx(k)
C     | and xx(k+1) by bisection, following Press et al.,
C     | Numerical Recipes in Fortran. xx must be monotonic.
C     *==========================================================*
C     \ev

C !USES:
      IMPLICIT NONE
C !INPUT PARAMETERS:
C     xx        :: array of bin-boundaries (layers_boundaries)
C     n         :: length of xx
C     m         :: int(log2(n)) + 1 = length of bisection loop
C     sNx,sNy   :: size of index array and input x
C     x         :: input array of values
C     k         :: index array (output)
C     myThid    :: my Thread Id number
      INTEGER n,m,sNx,sNy
      _RL     xx(1:n+1)
      _RL     x(snx+1,sny+1)
      INTEGER k(snx+1,sny+1)
      INTEGER myThid

C !LOCAL VARIABLES:
C     i,j      :: horizontal indices
C     l        :: bisection loop index
C     kl,ku,km :: work arrays and variables
      INTEGER i,j
CEOP
#ifdef TARGET_NEC_SX
      INTEGER l, km
      INTEGER kl(sNx+1,sNy+1), ku(sNx+1,sNy+1)

C     bisection, following Press et al., Numerical Recipes in Fortran,
C     mostly, because it can be vectorized
      DO j = 1,sNy+1
       DO i = 1,sNx+1
        kl(i,j)=1
        ku(i,j)=n+1
       END DO
      END DO
      DO l = 1,m
       DO j = 1,sNy+1
        DO i = 1,sNx+1
         IF (ku(i,j)-kl(i,j).GT.1) THEN
          km=(ku(i,j)+kl(i,j))/2
CML       IF ((xx(n).GE.xx(1)).EQV.(x(i,j).GE.xx(km))) THEN
          IF ( ((xx(n).GE.xx(1)).AND.(x(i,j).GE.xx(km))).OR.
     &         ((xx(n).GE.xx(1)).AND.(x(i,j).GE.xx(km))) ) THEN
           kl(i,j)=km
          ELSE
           ku(i,j)=km
          END IF
         END IF
        END DO
       END DO
      END DO
      DO j = 1,sNy+1
       DO i = 1,sNx+1
        IF ( x(i,j).LT.xx(2) ) THEN
         k(i,j)=1
        ELSE IF ( x(i,j).GE.xx(n) ) THEN
         k(i,j)=n
        ELSE
         k(i,j)=kl(i,j)
        END IF
       END DO
      END DO
#else
C     the old way
      DO j = 1,sNy+1
       DO i = 1,sNx+1
        IF (x(i,j) .GE. xx(n)) THEN
C     the point is in the hottest bin or hotter
         k(i,j) = n
        ELSE IF (x(i,j) .LT. xx(2)) THEN
C        the point is in the coldest bin or colder
         k(i,j) = 1
        ELSE IF ( (x(i,j) .GE. xx(k(i,j)))
     &    .AND.   (x(i,j) .LT. xx(k(i,j)+1)) ) THEN
C     already on the right bin -- do nothing
        ELSE IF (x(i,j) .GE. xx(k(i,j))) THEN
C     have to hunt for the right bin by getting hotter
         DO WHILE (x(i,j) .GE. xx(k(i,j)+1))
          k(i,j) = k(i,j) + 1
         ENDDO
C     now xx(k) < x <= xx(k+1)
        ELSE IF (x(i,j) .LT. xx(k(i,j)+1)) THEN
C     have to hunt for the right bin by getting colder
         DO WHILE (x(i,j) .LT. xx(k(i,j)))
          k(i,j) = k(i,j) - 1
         ENDDO
C     now xx(k) <= x < xx(k+1)
        ELSE
C     that should have covered all the options
         k(i,j) = -1
        ENDIF

       ENDDO
      ENDDO
#endif /* TARGET_NEC_SX */

#endif /* ALLOW_LAYERS */

      RETURN
      END

