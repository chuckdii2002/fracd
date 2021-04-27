#ifndef FRACDOILFLUIDFLOW_H
#define FRACDOILFLUIDFLOW_H
/*
 FracDOilFluidFlow.h
 (c) 2016-2017 Chukwudi CHukwudozie chdozie@gmail.com
 */


extern PetscErrorCode FracDdRo_dP(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh);
extern PetscErrorCode FracDdRo_dSw(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh);
extern PetscErrorCode FracDdRo_dSg(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh);
extern PetscErrorCode FracDdRo_dPbh(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh);
extern PetscErrorCode FracDRo( void *user, Vec Rw, Vec P, Vec Sw, Vec Sg, Vec Pbh);

#endif /* FRACDOILFLUIDFLOW_H */
