 &THSICE_CONST
#- with fractional ice:
 iceMaskMin = 0.001,
 hiMax      = 10.,
 hsMax      = 10.,
 dhSnowLin  = 0.1,
 fracEnFreez= 0.4,
 hNewIceMax = 1.,
#albIceMax  = 0.7,
#albIceMin  = 0.7,
 albColdSnow= 0.70,
 albWarmSnow= 0.60,
 tempSnowAlb= -5.,
#albOldSnow = 0.60,
#hNewSnowAge= 2.e-3,
#snowAgTime = 4320000.,
 albIceMax  = 0.60,
#albIceMin  = 0.20,
 hAlbIce    = 0.44,
 hAlbSnow   = 0.15,
 &

 &THSICE_PARM01
# if starting with no ice, set this to 1
# otherwise, if starting from a pickup with ice, set to 0
 StartIceModel=0,
#thSIce_skipThermo=.TRUE.,
 thSIceAdvScheme=77,
#thSIce_diffK   =800.,
 stressReduction=0.,
#- hack to balance FW difference to the target FW:
#-  sMxL_default = shift , tauRelax_MxL = Ampli , tauRelax_MxL_salt = Phase
 thSIceBalanceAtmFW=2,
 sMxL_default=-11.50859e-6,
 tauRelax_MxL=  3.67784e-6, 
 tauRelax_MxL_salt =  0.,
#----
# thSIceFract_InitFile='sice_one_upto56.bin',
# thSIceThick_InitFile='sice_one_upto56.bin',
#thSIce_diagFreq=2592000.,
 thSIce_monFreq = 2592000.,
 &


