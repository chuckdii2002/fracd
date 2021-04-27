#ifndef FRACDComputations_H
#define FRACDComputations_H
/*
 FRACDComputations.h
 (c) 2016-2018 Chukwudi CHukwudozie chdozie@gmail.com
 */

extern PetscErrorCode FracDComputeCubicSplineInterpolationCoefficients(PetscReal **data, PetscInt numdatarow);
extern PetscErrorCode FracDInterpolateUsingCubicSpline(PetscReal *yvalue, PetscReal xvalue, PetscReal extra, PetscReal **data, PetscReal *model_data, PetscInt numdatarow);
extern PetscErrorCode FracDInterpolateUsingAnalyticalAndCubicSpline(PetscReal *yvalue, PetscReal xvalue, PetscReal extra, PetscReal **data, PetscReal *model_data, PetscInt numdatarow);
extern PetscErrorCode FracDInterpolateUsingAnalyticalModel(PetscReal *yvalue, PetscReal xvalue, PetscReal extra, PetscReal **data, PetscReal *model_data, PetscInt numdatarow);

extern PetscErrorCode FracDQuantityAndDerivativeComputation(PetscReal *values, PetscReal x, PetscReal delta, PetscReal extra, PetscReal **data, PetscReal *model_data, PetscInt numdatarow, PetscErrorCode (*FracDInterpolateFunction)(PetscReal*,PetscReal,PetscReal,PetscReal**,PetscReal*, PetscInt));
extern PetscErrorCode FracDZeros(PetscReal *yvalue, PetscReal xvalue, PetscReal extra, PetscReal **data, PetscReal *model_data, PetscInt numdatarow);
extern PetscErrorCode FracDZeros1(PetscReal *kro, PetscReal Sw, PetscReal Sg, PetscReal extra, PetscReal **krowdata, PetscReal **krogdata, PetscReal **krwdata, PetscReal *stonedata, PetscInt numkrwdatarow, PetscInt numkrgdatarow);
extern PetscErrorCode FracDOnes(PetscReal *yvalue, PetscReal xvalue, PetscReal extra, PetscReal **data, PetscReal *model_data, PetscInt numdatarow);

#endif /* FRACDComputations_H */
