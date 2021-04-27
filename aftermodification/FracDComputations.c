/*
 (c) 2016-2018 Chukwudi Chukwudozie chdozie@gmail.com
 */

#include "petsc.h"
#include "FracDWell.h"
#include "FracDFiniteElement.h"
#include "FracDBasic.h"
#include "FracDFlow.h"




#undef __FUNCT__
#define __FUNCT__ "FracDComputeCubicSplineInterpolationCoefficients"
extern PetscErrorCode FracDComputeCubicSplineInterpolationCoefficients(PetscReal **data, PetscInt numdatarow)
{
    PetscErrorCode          ierr;
    Vec            x, b;
    Mat            A;
    KSP            ksp;
    PC             pc;
    PetscScalar    one = 1.0,value[3];
    PetscScalar    v_i,h_i,h_neg,b_i,b_neg,u_i;
    PetscScalar    v[numdatarow],h[numdatarow],bb[numdatarow],u[numdatarow];
    PetscInt       i,j,n,col[3];
    PetscScalar    *b_array=NULL,*x_array=NULL;
    
    PetscFunctionBegin;
    for(i = 0; i < numdatarow; i++)
    {
        h[i] = bb[i] = v[i] = u[i] = 0.;
    }
    for(i = 0; i < numdatarow-1; i++)
    {
        h[i] = data[0][i+1]-data[0][i];
        bb[i] = (data[1][i+1]-data[1][i])/h[i];
    }
    for(i = 1; i < numdatarow-1; i++)
    {
        v[i] = 2.*(h[i-1]+h[i]);
        u[i] = 6.*(bb[i]-bb[i-1]);
    }
    n = numdatarow;

    ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
    ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
    ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
    
    ierr = VecCreate(PETSC_COMM_SELF,&x);CHKERRQ(ierr);
    ierr = VecSetType(x, VECSEQ);CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE,n);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
    ierr = VecSet(x,.0);CHKERRQ(ierr);
    ierr = VecSet(b,.0);CHKERRQ(ierr);
    ierr = VecGetArray(b,&b_array);CHKERRQ(ierr);
    for (i = 2; i < n-2; i++) {
        j = i-1;
        h_neg = data[0][j+1]-data[0][j];
        h_i = data[0][i+1]-data[0][i];
        b_neg = (data[1][j+1]-data[1][j])/h_neg;
        b_i = (data[1][i+1]-data[1][i])/h_i;
        v_i = 2.*(h_i+h_neg);
        u_i = 6.*(b_i-b_neg);
        value[0] = h_neg;
        value[1] = v_i;
        value[2] = h_i;
 
        
        col[0] = i-1; col[1] = i; col[2] = i+1;
        ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
        b_array[i] = u_i;
    }
    i    = 0;
    ierr = MatSetValues(A,1,&i,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
    i = 1; col[0] = 1; col[1] = 2;
    j = i-1;
    h_neg = data[0][j+1]-data[0][j];
    h_i = data[0][i+1]-data[0][i];
    b_neg = (data[1][j+1]-data[1][j])/h_neg;
    b_i = (data[1][i+1]-data[1][i])/h_i;
    v_i = 2.*(h_i+h_neg);
    u_i = 6.*(b_i-b_neg);
    value[0] = v_i;
    value[1] = h_i;
    value[2] = 0;
    b_array[i] = u_i;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    i = n-1;
    ierr = MatSetValues(A,1,&i,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
    i = n-2; col[0] = n-3; col[1] = n-2;
    j = i-1;
    h_neg = data[0][j+1]-data[0][j];
    h_i = data[0][i+1]-data[0][i];
    b_neg = (data[1][j+1]-data[1][j])/h_neg;
    b_i = (data[1][i+1]-data[1][i])/h_i;
    v_i = 2.*(h_i+h_neg);
    u_i = 6.*(b_i-b_neg);
    value[0] = h_neg;
    value[1] = v_i;
    value[2] = 0;
    b_array[i] = u_i;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = VecRestoreArray(b,&b_array);CHKERRQ(ierr);
    ierr = KSPCreate(PETSC_COMM_SELF,&ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
    ierr = VecGetArray(x,&x_array);CHKERRQ(ierr);
    for (i = 0; i < n; i++)   data[2][i] = x_array[i];
    ierr = VecRestoreArray(x,&x_array);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDInterpolateUsingAnalyticalModel"
extern PetscErrorCode FracDInterpolateUsingAnalyticalModel(PetscReal *yvalue, PetscReal xvalue, PetscReal extra, PetscReal **table_data, PetscReal *model_data, PetscInt num)
{
    
    PetscReal Yo, C, P, Po;
    
    PetscFunctionBegin;
    P = xvalue;
    Po = model_data[2];
    C = model_data[1];
    Yo = model_data[0];
    *yvalue = Yo*PetscExpReal(C*(P-Po));
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FracDInterpolateUsingCubicSpline"
extern PetscErrorCode FracDInterpolateUsingCubicSpline(PetscReal *yvalue, PetscReal xvalue, PetscReal extra, PetscReal **data, PetscReal *model_data, PetscInt numdatarow)
{
    PetscInt       i = 0;
    PetscReal      h;
    
    PetscFunctionBegin;
    i = 0;
    if(xvalue <= data[0][0]){
      *yvalue = data[1][0];
    }
    else if(xvalue >= data[0][numdatarow-1]){
        *yvalue = data[1][numdatarow-1];
    }
    else{
        while(xvalue > data[0][i]){
            i++;
        }
        i--;
        h = data[0][i+1]-data[0][i];
        *yvalue = data[2][i+1]/(6*h)*(PetscPowScalar(xvalue-data[0][i],3)) + data[2][i]/(6*h)*(PetscPowScalar(data[0][i+1]-xvalue,3))
        + ((data[1][i+1]/h)-(data[2][i+1]*h/6))*(xvalue-data[0][i]) + ((data[1][i]/h)-(data[2][i]*h/6))*(data[0][i+1]-xvalue);
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDInterpolateUsingAnalyticalAndCubicSpline"
extern PetscErrorCode FracDInterpolateUsingAnalyticalAndCubicSpline(PetscReal *yvalue, PetscReal xvalue, PetscReal extra, PetscReal **data, PetscReal *model_data, PetscInt numdatarow)
{
    PetscErrorCode              ierr;
    PetscReal                   hold;
    PetscFunctionBegin;
    
    hold = model_data[0];
    ierr = FracDInterpolateUsingCubicSpline(yvalue,xvalue,PETSC_NULL,data,model_data,numdatarow);CHKERRQ(ierr);
    if(xvalue >= model_data[2]){
        model_data[0] = *yvalue;
        ierr = FracDInterpolateUsingAnalyticalModel(yvalue,xvalue,PETSC_NULL,data,model_data,numdatarow);CHKERRQ(ierr);
    }
    model_data[0] = hold;

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDQuantityAndDerivativeComputation"
extern PetscErrorCode FracDQuantityAndDerivativeComputation(PetscReal *values, PetscReal x, PetscReal delta, PetscReal extra, PetscReal **data, PetscReal *model_data, PetscInt numdatarow, PetscErrorCode (*FracDInterpolateFunction)(PetscReal*,PetscReal,PetscReal,PetscReal**,PetscReal*, PetscInt))
{
    PetscErrorCode              ierr;
    PetscReal                   y,y_new,x_new;
    
    PetscFunctionBegin;
    x_new = delta+x;
    ierr =  FracDInterpolateFunction(&y,x,PETSC_NULL,data,model_data,numdatarow);CHKERRQ(ierr);
    ierr =  FracDInterpolateFunction(&y_new,x_new,PETSC_NULL,data,model_data,numdatarow);CHKERRQ(ierr);
    values[0] = y;
    values[1] = (y-y_new)/(x-x_new);
    values[2] = (1./y-1./y_new)/(x-x_new);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDZeros"
extern PetscErrorCode FracDZeros(PetscReal *yvalue, PetscReal xvalue, PetscReal extra, PetscReal **data, PetscReal *model_data, PetscInt numdatarow)
{
    PetscFunctionBegin;
    *yvalue = 0.;    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDZeros1"
extern PetscErrorCode FracDZeros1(PetscReal *kro, PetscReal Sw, PetscReal Sg, PetscReal extra, PetscReal **krowdata, PetscReal **krogdata, PetscReal **krwdata, PetscReal *stonedata, PetscInt numkrwdatarow, PetscInt numkrgdatarow)
{
    PetscFunctionBegin;
    *kro = 0.;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDOnes"
extern PetscErrorCode FracDOnes(PetscReal *yvalue, PetscReal xvalue, PetscReal extra, PetscReal **data, PetscReal *model_data, PetscInt numdatarow)
{
    PetscFunctionBegin;
    *yvalue = 1.;
    PetscFunctionReturn(0);
}
