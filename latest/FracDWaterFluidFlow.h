#ifndef FRACDWATERFLUIDFLOW_H
#define FRACDWATERFLUIDFLOW_H
/*
 FracDFluidFlow.h
 (c) 2016-2018 Chukwudi CHukwudozie chdozie@gmail.com
 */

extern PetscErrorCode FracDdRw_dP(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh);
extern PetscErrorCode FracDdRw_dSw(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh);
extern PetscErrorCode FracDdRw_dSg(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh);
extern PetscErrorCode FracDdRw_dPbh(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh);
extern PetscErrorCode FracDRw( void *user, Vec Rw, Vec P, Vec Sw, Vec Sg, Vec Pbh);
extern PetscErrorCode FracDIntialialFlowGuess(SNES snesP,Vec V, void *user);

#endif /* FRACDWATERFLUIDFLOW_H */
