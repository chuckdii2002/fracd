#ifndef FRACDFLOW_H
#define FRACDFLOW_H
/*
 FracDFlow.h
 (c) 2016-2018 Chukwudi CHukwudozie chdozie@gmail.com
 */

extern PetscErrorCode FracDPJacobian(SNES snesP,Vec V,Mat K, Mat KPC, void *user);
extern PetscErrorCode FracDPResidual(SNES snesP,Vec V,Vec residual,void *user);
extern PetscErrorCode FracDSolveP(AppCtx *bag);

extern PetscErrorCode FracDDiffusiveFluxMatrixLocal(PetscReal *K_local, PetscReal *Conductivity, PetscReal *coords, FracDCVFEFace *f);
extern PetscErrorCode FracDAdvectiveFluxMatrixLocal(PetscReal *K_local, PetscReal rhoCw, PetscReal *V_array, PetscReal *coords, FracDCVFEFace *f);

extern PetscErrorCode FracDUpDateWaterDensity(PetscReal *rho, PetscReal rho_ref, PetscReal rho_coeff, PetscReal p, PetscReal p_ref);
extern PetscErrorCode FracDUpDateRockPorosity(PetscReal *phi, PetscReal phi_ref, PetscReal Cf,PetscReal p, PetscReal p_ref);
extern PetscErrorCode FracDUpDateRelativeWaterPermeability(PetscReal *kr, PetscReal kr_ref, PetscReal kr_coeff, PetscReal S, PetscReal S_ref);
extern PetscErrorCode FracDOilWaterKrow(PetscReal *kro, PetscReal Sw, PetscReal Sg, PetscReal extra, PetscReal **krowdata, PetscReal **krogdata, PetscReal **krwdata, PetscReal *stonedata, PetscInt numkrwdatarow, PetscInt numkrgdatarow);
extern PetscErrorCode FracDStone1Model(PetscReal *kro, PetscReal Sw, PetscReal Sg, PetscReal extra, PetscReal **krowdata, PetscReal **krogdata, PetscReal **krwdata, PetscReal *stonedata, PetscInt numkrwdatarow, PetscInt numkrgdatarow);
extern PetscErrorCode FracDUpdateCapillaryPressureAndDerivative(Vec Pc, Vec dervPc, Vec S,  PetscReal delta, PetscReal **PcTableData, PetscInt numdatarow);
extern PetscErrorCode FracDUpdateBubblePoint(Vec Pb, Vec Status, Vec P, PetscReal fixed_value);
extern PetscErrorCode FracDUpdateSolutionGasOilRatioAndDerivative(Vec Rs, Vec dervRs, Vec Pb, Vec P, PetscReal delta, PetscReal fixed_value, PetscReal *model_data, PetscReal **TableData, PetscInt numdatarow);
extern PetscErrorCode FracDInitializeSolutionGasOilRatio(Vec Rs, Vec P, Vec Sw, Vec Sg, FracDPVT OilPVTData, FracDPVT GasPVTData, PetscReal ini_Rs, PetscReal ini_pb);

extern PetscErrorCode FracDInitializeSaturationBubblePointAndSwitchingVariables(Vec INDC1, Vec INDC2, Vec Rs, Vec P, Vec Pb, Vec Sw, Vec Sg,  FracDPVT OilPVTData, FracDPVT GasPVTData, FracDPbRs            SolutionGasOilData);

extern PetscErrorCode FracDInitializePackedSgRsField(Vec SgRs, Vec Rs, Vec Sg, Vec INDC2);

#endif /* FRACDFLOW_H */
