#ifndef FRACDMECHANICS_H
#define FRACDMECHANICS_H
/*
 FracDMechanics.h
 (c) 2016-2018 Chukwudi CHukwudozie chdozie@gmail.com
 */



extern PetscErrorCode FracDUJacobian(SNES snesU,Vec U,Mat K, Mat KPC, void *user);
extern PetscErrorCode FracDUResidual(SNES snesU,Vec U,Vec residual,void *user);
extern PetscErrorCode FracDApplyBodyForce_local(PetscScalar *residual_local,PetscScalar *F_array, PetscReal *g, PetscReal rho, FracDFEElement *e);
extern PetscErrorCode FracDThermoPoroelastic_local(PetscScalar *residual_local,PetscScalar *p_array, PetscScalar *T_array, PetscReal E, PetscReal nu, PetscReal beta, PetscReal alpha, FracDFEElement *e);
extern PetscErrorCode FracDElasticity3D_local(PetscReal *K_local,PetscReal E,PetscReal nu,FracDFEElement *e);
extern PetscErrorCode FracDElasticity2DPlaneStrain_local(PetscReal *K_local,PetscReal E,PetscReal nu,FracDFEElement *e);
extern PetscErrorCode FracDElasticity2DPlaneStress_local(PetscReal *K_local,PetscReal E,PetscReal nu,FracDFEElement *e);

extern PetscErrorCode FracDSolveU(AppCtx *bag);





#endif /* FRACDMECHANICS_H */
