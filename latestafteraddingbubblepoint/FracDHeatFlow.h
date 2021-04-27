#ifndef FRACDHEATFLOW_H
#define FRACDHEATFLOW_H
/*
 FracDHeatFlow.h
 (c) 2016-2017 Chukwudi CHukwudozie chdozie@gmail.com
 */


extern PetscErrorCode FracDTJacobian(SNES snesT,Vec T,Mat K, Mat KPC, void *user);
extern PetscErrorCode FracDTResidual(SNES snesT,Vec T,Vec residual,void *user);
extern PetscErrorCode FracDSolveT(AppCtx *bag);





#endif /* FRACDHEATFLOW_H */
