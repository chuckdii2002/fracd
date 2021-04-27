#ifndef FRACDGASFLUIDFLOW_H
#define FRACDGASFLUIDFLOW_H
/*
 FracDGasFluidFlow.h
 (c) 2016-2017 Chukwudi CHukwudozie chdozie@gmail.com
 */


extern PetscErrorCode FracDdRg_dP(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh);
extern PetscErrorCode FracDdRg_dSw(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh);
extern PetscErrorCode FracDdRg_dSg(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh);
extern PetscErrorCode FracDdRg_dPbh(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh);
extern PetscErrorCode FracDRg( void *user, Vec Rw, Vec P, Vec Sw, Vec Sg, Vec Pbh);

#endif /* FRACDGASFLUIDFLOW_H */
