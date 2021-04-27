#ifndef FRACDWELLFLUIDFLOW_H
#define FRACDWELLFLUIDFLOW_H
/*
 FracDWellFluidFlow.h
 (c) 2016-2018 Chukwudi CHukwudozie chdozie@gmail.com
 */

extern PetscErrorCode FracDMatrixApplyWellBottomHolePressureCondition(Mat K, FracDWell *well, PetscInt numwells, PetscScalar diagonalvalue);
extern PetscErrorCode FracDdRpbh_dP(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh);
extern PetscErrorCode FracDdRpbh_dSw(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh);
extern PetscErrorCode FracDdRpbh_dSg(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh);
extern PetscErrorCode FracDdRpbh_dPbh(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh);
extern PetscErrorCode FracDRpbh(void *user, Vec RPbh, Vec P, Vec Sw, Vec Sg, Vec Pbh);
extern PetscErrorCode FracDWellModelRates(PetscReal *Qw,PetscReal *Qo,PetscReal *Qg,FracDWell well,PetscReal G,PetscReal Pbh,PetscReal P,PetscReal Pb,PetscReal Sw,PetscReal Sg,PetscReal Pcow,PetscReal Pcog,FracDPVT WaterPVTData,FracDPVT OilPVTData,FracDPVT GasPVTData,FracDRelPerm RelPermData,FracDPbRs SolutionGasOilData);
extern PetscErrorCode FracDWellModel(PetscReal *Q,PetscReal *Qconstr,FracDWell well,PetscReal G,PetscReal Pbh,PetscReal P,PetscReal Pb,PetscReal Sw,PetscReal Sg,PetscReal Pcow,PetscReal Pcog,FracDPVT WaterPVTData,FracDPVT OilPVTData,FracDPVT GasPVTData,FracDRelPerm RelPermData,FracDPbRs SolutionGasOilData);
extern PetscErrorCode FracDQw(PetscReal *Qw,FracDWell well,PetscReal G,PetscReal Pbh,PetscReal P,PetscReal Pb,PetscReal Sw,PetscReal Sg,PetscReal Pcow,PetscReal Pcog,FracDPVT WaterPVTData,FracDPVT OilPVTData,FracDPVT GasPVTData,FracDRelPerm RelPermData);
extern PetscErrorCode FracDQo(PetscReal *Qo,FracDWell well,PetscReal G,PetscReal Pbh,PetscReal P,PetscReal Pb,PetscReal Sw,PetscReal Sg,PetscReal Pcow,PetscReal Pcog,FracDPVT WaterPVTData,FracDPVT OilPVTData,FracDPVT GasPVTData,FracDRelPerm RelPermData);
extern PetscErrorCode FracDQg(PetscReal *Qg,FracDWell well,PetscReal G,PetscReal Pbh,PetscReal P,PetscReal Pb,PetscReal Sw,PetscReal Sg,PetscReal Pcow,PetscReal Pcog,FracDPVT WaterPVTData,FracDPVT OilPVTData,FracDPVT GasPVTData,FracDRelPerm RelPermData,FracDPbRs SolutionGasOilRatio);

#endif /* FRACDWELLFLUIDFLOW_H */
