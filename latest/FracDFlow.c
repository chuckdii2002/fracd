/*
 (c) 2016-2018 Chukwudi Chukwudozie chdozie@gmail.com
 */

#include "petsc.h"
#include "FracDWell.h"
#include "FracDFiniteElement.h"
#include "FracDBasic.h"
#include "FracDFlow.h"
#include "FracDWaterFluidFlow.h"
#include "FracDOilFluidFlow.h"
#include "FracDGasFluidFlow.h"
#include "FracDWellFluidFlow.h"
#include "FracDComputations.h"


#undef __FUNCT__
#define __FUNCT__ "FracDPJacobian"
extern PetscErrorCode FracDPJacobian(SNES snesP,Vec V,Mat K, Mat KPC, void *user)
{
    
    PetscErrorCode ierr;
    AppCtx         *bag=(AppCtx*)user;
    Vec            P,Sw = NULL,Sg = NULL,Pbh = NULL;
    IS             *is;
    Mat            KPP = NULL,KPSw = NULL,KPSg = NULL,KPPbh = NULL;
    Mat            KSwP = NULL,KSwSw = NULL,KSwSg = NULL,KSwPbh = NULL;
    Mat            KSgP = NULL,KSgSw = NULL,KSgSg = NULL,KSgPbh = NULL;
    Mat            KPbhP = NULL,KPbhSw = NULL,KPbhSg = NULL,KPbhPbh = NULL;
    Mat            KPPC = NULL,KPSwC = NULL,KPSgC = NULL,KPPbhC = NULL;
    Mat            KSwPC = NULL,KSwSwC = NULL,KSwSgC = NULL,KSwPbhC = NULL;
    Mat            KSgPC = NULL,KSgSwC = NULL,KSgSgC = NULL,KSgPbhC = NULL;
    Mat            KPbhPC = NULL,KPbhSwC = NULL,KPbhSgC = NULL,KPbhPbhC = NULL;
    
    PetscFunctionBegin;
    ierr = MatZeroEntries(K);CHKERRQ(ierr);
    if(KPC != K){
        ierr = MatZeroEntries(KPC);CHKERRQ(ierr);
    }
    //    printf("\n\n In Jacobian: %g %g %g\n\n\n\n\n",bag->ppties.OilPVTData.B_ModelData[0],bag->ppties.OilPVTData.B_ModelData[1],bag->ppties.OilPVTData.B_ModelData[2]);
    
    switch (bag->fluid) {
        case 0:
        {
            ierr = DMCompositeGetGlobalISs(bag->MultiPhasePacker,&is);CHKERRQ(ierr);
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,V,&P,&Pbh);CHKERRQ(ierr);
            if (bag->numWells == 0)
            {
                if(KPC != K){
                    ierr = FracDdRw_dP(bag,K,KPC,P,bag->fields.Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                }
                else{
                    ierr = FracDdRw_dP(bag,K,K,P,bag->fields.Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                }
            }
            else
            {
                ierr = MatCreateSubMatrix(K,is[0],is[0],MAT_INITIAL_MATRIX,&KPP);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[0],is[1],MAT_INITIAL_MATRIX,&KPPbh);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[1],is[0],MAT_INITIAL_MATRIX,&KPbhP);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[1],is[1],MAT_INITIAL_MATRIX,&KPbhPbh);CHKERRQ(ierr);
                if(KPC != K){
                    ierr = MatCreateSubMatrix(KPC,is[0],is[0],MAT_INITIAL_MATRIX,&KPPC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[0],is[1],MAT_INITIAL_MATRIX,&KPPbhC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[1],is[0],MAT_INITIAL_MATRIX,&KPbhPC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[1],is[1],MAT_INITIAL_MATRIX,&KPbhPbhC);CHKERRQ(ierr);
                    ierr = FracDdRw_dP(bag,KPP,KPPC,P,bag->fields.Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRw_dPbh(bag,KPPbh,KPPbhC,P,bag->fields.Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRpbh_dP(bag,KPbhP,KPbhPC,P,bag->fields.Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRpbh_dPbh(bag,KPbhPbh,KPbhPbhC,P,bag->fields.Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                }
                else{
                    ierr = FracDdRw_dP(bag,KPP,KPP,P,bag->fields.Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRw_dPbh(bag,KPPbh,KPPbh,P,bag->fields.Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRpbh_dP(bag,KPbhP,KPbhP,P,bag->fields.Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRpbh_dPbh(bag,KPbhPbh,KPbhPbh,P,bag->fields.Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                }
            }
            ierr = DMCompositeRestoreAccess(bag->MultiPhasePacker,V,&P,&Pbh);CHKERRQ(ierr);
            if(bag->numWells){
                ierr = ISDestroy(&is[0]);CHKERRQ(ierr);
                ierr = ISDestroy(&is[1]);CHKERRQ(ierr);
                ierr = PetscFree(is);CHKERRQ(ierr);
                ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                if (KPC != K) {
                    ierr = MatAssemblyBegin(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                    ierr = MatAssemblyEnd  (KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                    
                    ierr = MatDestroy(&KPPC);CHKERRQ(ierr);
                    ierr = MatDestroy(&KPPbhC);CHKERRQ(ierr);
                    ierr = MatDestroy(&KPbhPC);CHKERRQ(ierr);
                    ierr = MatDestroy(&KPbhPbhC);CHKERRQ(ierr);
                }
                ierr = MatDestroy(&KPP);CHKERRQ(ierr);
                ierr = MatDestroy(&KPPbh);CHKERRQ(ierr);
                ierr = MatDestroy(&KPbhP);CHKERRQ(ierr);
                ierr = MatDestroy(&KPbhPbh);CHKERRQ(ierr);
            }
        }
        break;
        case 1:{
            
        }
        break;
        
        case 2:
        {
            ierr = DMCompositeGetGlobalISs(bag->MultiPhasePacker,&is);CHKERRQ(ierr);
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,V,&P,&Sw,&Pbh);CHKERRQ(ierr);
            if (bag->numWells == 0)
            {
                ierr = MatCreateSubMatrix(K,is[0],is[0],MAT_INITIAL_MATRIX,&KPP);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[0],is[1],MAT_INITIAL_MATRIX,&KPSw);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[1],is[0],MAT_INITIAL_MATRIX,&KSwP);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[1],is[1],MAT_INITIAL_MATRIX,&KSwSw);CHKERRQ(ierr);
                if(KPC != K){
                    ierr = MatCreateSubMatrix(KPC,is[0],is[0],MAT_INITIAL_MATRIX,&KPPC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[0],is[1],MAT_INITIAL_MATRIX,&KPSwC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[1],is[0],MAT_INITIAL_MATRIX,&KSwPC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[1],is[1],MAT_INITIAL_MATRIX,&KSwSwC);CHKERRQ(ierr);
                    
                    //                    ierr = FracDdRw_dP(bag,KPP,KPPC,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //                    ierr = FracDdRw_dSw(bag,KPSw,KPSwC,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //
                    //                    ierr = FracDdRo_dP(bag,KSwP,KSwPC,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //                    ierr = FracDdRo_dSw(bag,KSwSw,KSwSwC,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                }
                else{
                    //                    ierr = FracDdRw_dP(bag,KPP,KPP,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //                    ierr = FracDdRw_dSw(bag,KPSw,KPSw,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //
                    //                    ierr = FracDdRo_dP(bag,KSwP,KSwP,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //                    ierr = FracDdRo_dSw(bag,KSwSw,KSwSw,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                }
            }
            else
            {
                ierr = MatCreateSubMatrix(K,is[0],is[0],MAT_INITIAL_MATRIX,&KPP);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[0],is[1],MAT_INITIAL_MATRIX,&KPSw);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[0],is[2],MAT_INITIAL_MATRIX,&KPPbh);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[1],is[0],MAT_INITIAL_MATRIX,&KSwP);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[1],is[1],MAT_INITIAL_MATRIX,&KSwSw);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[1],is[2],MAT_INITIAL_MATRIX,&KSwPbh);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[2],is[0],MAT_INITIAL_MATRIX,&KPbhP);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[2],is[1],MAT_INITIAL_MATRIX,&KPbhSw);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[2],is[2],MAT_INITIAL_MATRIX,&KPbhPbh);CHKERRQ(ierr);
                if(KPC != K){
                    ierr = MatCreateSubMatrix(KPC,is[0],is[0],MAT_INITIAL_MATRIX,&KPPC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[0],is[1],MAT_INITIAL_MATRIX,&KPSwC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[0],is[2],MAT_INITIAL_MATRIX,&KPPbhC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[1],is[0],MAT_INITIAL_MATRIX,&KSwPC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[1],is[1],MAT_INITIAL_MATRIX,&KSwSwC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[1],is[2],MAT_INITIAL_MATRIX,&KSwPbhC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[2],is[0],MAT_INITIAL_MATRIX,&KPbhPC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[2],is[1],MAT_INITIAL_MATRIX,&KPbhSwC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[2],is[2],MAT_INITIAL_MATRIX,&KPbhPbhC);CHKERRQ(ierr);
                    
                    ierr = FracDdRw_dP(bag,KPP,KPPC,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRw_dSw(bag,KPSw,KPSwC,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRw_dPbh(bag,KPPbh,KPPbhC,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //
                    ierr = FracDdRo_dP(bag,KSwP,KSwPC,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRo_dSw(bag,KSwSw,KSwSwC,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRo_dPbh(bag,KSwPbh,KSwPbhC,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //
                    //
                    ierr = FracDdRpbh_dP(bag,KPbhP,KPbhPC,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRpbh_dSw(bag,KPbhSw,KPbhSwC,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRpbh_dPbh(bag,KPbhPbh,KPbhPbhC,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                }
                else{
                    //                    ierr = FracDdRw_dP(bag,KPP,KPP,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //                    ierr = FracDdRw_dSw(bag,KPSw,KPSw,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //                    ierr = FracDdRw_dPbh(bag,KPPbh,KPPbh,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //
                    //                    ierr = FracDdRo_dP(bag,KSwP,KSwP,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //                    ierr = FracDdRo_dSw(bag,KSwSw,KSwSw,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //                    ierr = FracDdRo_dPbh(bag,KSwPbh,KSwPbh,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //
                    //                    ierr = FracDdRpbh_dP(bag,KPbhP,KPbhP,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //                    ierr = FracDdRpbh_dSw(bag,KPbhSw,KPbhSw,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //                    ierr = FracDdRpbh_dPbh(bag,KPbhPbh,KPbhPbh,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                }
            }
            ierr = DMCompositeRestoreAccess(bag->MultiPhasePacker,V,&P,&Sw,&Pbh);CHKERRQ(ierr);
            ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
            ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
            if (KPC != K) {
                ierr = MatAssemblyBegin(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                ierr = MatAssemblyEnd  (KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                
            }
            ierr = ISDestroy(&is[0]);CHKERRQ(ierr);
            ierr = ISDestroy(&is[1]);CHKERRQ(ierr);
            if(bag->numWells){
                ierr = ISDestroy(&is[2]);CHKERRQ(ierr);
            }
            ierr = PetscFree(is);CHKERRQ(ierr);
            ierr = MatDestroy(&KPP);CHKERRQ(ierr);
            ierr = MatDestroy(&KPSw);CHKERRQ(ierr);
            ierr = MatDestroy(&KSwP);CHKERRQ(ierr);
            ierr = MatDestroy(&KSwSw);CHKERRQ(ierr);
            if (KPC != K) {
                ierr = MatDestroy(&KPPC);CHKERRQ(ierr);
                ierr = MatDestroy(&KPSwC);CHKERRQ(ierr);
                ierr = MatDestroy(&KSwPC);CHKERRQ(ierr);
                ierr = MatDestroy(&KSwSwC);CHKERRQ(ierr);
            }
            if(bag->numWells){
                ierr = MatDestroy(&KPPbh);CHKERRQ(ierr);
                ierr = MatDestroy(&KSwPbh);CHKERRQ(ierr);
                ierr = MatDestroy(&KPbhP);CHKERRQ(ierr);
                ierr = MatDestroy(&KPbhSw);CHKERRQ(ierr);
                ierr = MatDestroy(&KPbhPbh);CHKERRQ(ierr);
                if (KPC != K) {
                    ierr = MatDestroy(&KPPbhC);CHKERRQ(ierr);
                    ierr = MatDestroy(&KSwPbhC);CHKERRQ(ierr);
                    ierr = MatDestroy(&KPbhPC);CHKERRQ(ierr);
                    ierr = MatDestroy(&KPbhSwC);CHKERRQ(ierr);
                    ierr = MatDestroy(&KPbhPbhC);CHKERRQ(ierr);
                }
            }
        }
        break;
        case 3:
        {
            ierr = DMCompositeGetGlobalISs(bag->MultiPhasePacker,&is);CHKERRQ(ierr);
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,V,&P,&Sw,&Sg,&Pbh);CHKERRQ(ierr);
            if (bag->numWells == 0)
            {
                ierr = MatCreateSubMatrix(K,is[0],is[0],MAT_INITIAL_MATRIX,&KPP);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[0],is[1],MAT_INITIAL_MATRIX,&KPSw);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[0],is[2],MAT_INITIAL_MATRIX,&KPSg);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[1],is[0],MAT_INITIAL_MATRIX,&KSwP);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[1],is[1],MAT_INITIAL_MATRIX,&KSwSw);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[1],is[2],MAT_INITIAL_MATRIX,&KSwSg);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[2],is[0],MAT_INITIAL_MATRIX,&KSgP);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[2],is[1],MAT_INITIAL_MATRIX,&KSgSw);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[2],is[2],MAT_INITIAL_MATRIX,&KSgSg);CHKERRQ(ierr);
                if(KPC != K){
                    ierr = MatCreateSubMatrix(KPC,is[0],is[0],MAT_INITIAL_MATRIX,&KPPC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[0],is[1],MAT_INITIAL_MATRIX,&KPSwC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[0],is[2],MAT_INITIAL_MATRIX,&KPSgC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[1],is[0],MAT_INITIAL_MATRIX,&KSwPC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[1],is[1],MAT_INITIAL_MATRIX,&KSwSwC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[1],is[2],MAT_INITIAL_MATRIX,&KSwSgC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[2],is[0],MAT_INITIAL_MATRIX,&KSgPC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[2],is[1],MAT_INITIAL_MATRIX,&KSgSwC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(KPC,is[2],is[2],MAT_INITIAL_MATRIX,&KSgSgC);CHKERRQ(ierr);
                    
                    //                    ierr = FracDdRw_dP(bag,KPP,KPPC,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //                    ierr = FracDdRw_dSw(bag,KPSw,KPSwC,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //
                    //                    ierr = FracDdRo_dP(bag,KSwP,KSwPC,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //                    ierr = FracDdRo_dSw(bag,KSwSw,KSwSwC,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                }
                else{
                    //                    ierr = FracDdRw_dP(bag,KPP,KPP,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //                    ierr = FracDdRw_dSw(bag,KPSw,KPSw,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //
                    //                    ierr = FracDdRo_dP(bag,KSwP,KSwP,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //                    ierr = FracDdRo_dSw(bag,KSwSw,KSwSw,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                }
            }
            else
            {
                ierr = MatCreateSubMatrix(K,is[0],is[0],MAT_INITIAL_MATRIX,&KPP);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[0],is[1],MAT_INITIAL_MATRIX,&KPSw);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[0],is[2],MAT_INITIAL_MATRIX,&KPSg);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[0],is[3],MAT_INITIAL_MATRIX,&KPPbh);CHKERRQ(ierr);
                
                ierr = MatCreateSubMatrix(K,is[1],is[0],MAT_INITIAL_MATRIX,&KSwP);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[1],is[1],MAT_INITIAL_MATRIX,&KSwSw);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[1],is[2],MAT_INITIAL_MATRIX,&KSwSg);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[1],is[3],MAT_INITIAL_MATRIX,&KSwPbh);CHKERRQ(ierr);
                
                ierr = MatCreateSubMatrix(K,is[2],is[0],MAT_INITIAL_MATRIX,&KSgP);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[2],is[1],MAT_INITIAL_MATRIX,&KSgSw);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[2],is[2],MAT_INITIAL_MATRIX,&KSgSg);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[2],is[3],MAT_INITIAL_MATRIX,&KSgPbh);CHKERRQ(ierr);
                
                ierr = MatCreateSubMatrix(K,is[3],is[0],MAT_INITIAL_MATRIX,&KPbhP);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[3],is[1],MAT_INITIAL_MATRIX,&KPbhSw);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[3],is[2],MAT_INITIAL_MATRIX,&KPbhSg);CHKERRQ(ierr);
                ierr = MatCreateSubMatrix(K,is[3],is[3],MAT_INITIAL_MATRIX,&KPbhPbh);CHKERRQ(ierr);
                if(KPC != K){
                    ierr = MatCreateSubMatrix(K,is[0],is[0],MAT_INITIAL_MATRIX,&KPPC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(K,is[0],is[1],MAT_INITIAL_MATRIX,&KPSwC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(K,is[0],is[2],MAT_INITIAL_MATRIX,&KPSgC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(K,is[0],is[3],MAT_INITIAL_MATRIX,&KPPbhC);CHKERRQ(ierr);
                    
                    ierr = MatCreateSubMatrix(K,is[1],is[0],MAT_INITIAL_MATRIX,&KSwPC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(K,is[1],is[1],MAT_INITIAL_MATRIX,&KSwSwC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(K,is[1],is[2],MAT_INITIAL_MATRIX,&KSwSgC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(K,is[1],is[3],MAT_INITIAL_MATRIX,&KSwPbhC);CHKERRQ(ierr);
                    
                    ierr = MatCreateSubMatrix(K,is[2],is[0],MAT_INITIAL_MATRIX,&KSgPC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(K,is[2],is[1],MAT_INITIAL_MATRIX,&KSgSwC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(K,is[2],is[2],MAT_INITIAL_MATRIX,&KSgSgC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(K,is[2],is[3],MAT_INITIAL_MATRIX,&KSgPbhC);CHKERRQ(ierr);
                    
                    ierr = MatCreateSubMatrix(K,is[3],is[0],MAT_INITIAL_MATRIX,&KPbhPC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(K,is[3],is[1],MAT_INITIAL_MATRIX,&KPbhSwC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(K,is[3],is[2],MAT_INITIAL_MATRIX,&KPbhSgC);CHKERRQ(ierr);
                    ierr = MatCreateSubMatrix(K,is[3],is[3],MAT_INITIAL_MATRIX,&KPbhPbhC);CHKERRQ(ierr);
                    
                    ierr = FracDdRw_dP(bag,KPP,KPPC,P,Sw,Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRw_dSw(bag,KPSw,KPSwC,P,Sw,Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRw_dPbh(bag,KPPbh,KPPbhC,P,Sw,Sg,Pbh);CHKERRQ(ierr);
                    
                    ierr = FracDdRo_dP(bag,KSwP,KSwPC,P,Sw,Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRo_dSw(bag,KSwSw,KSwSwC,P,Sw,Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRo_dSg(bag,KSwSg,KSwSgC,P,Sw,Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRo_dPbh(bag,KSwPbh,KSwPbhC,P,Sw,Sg,Pbh);CHKERRQ(ierr);
                    
                    ierr = FracDdRg_dP(bag,KSgP,KSgPC,P,Sw,Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRg_dSw(bag,KSgSw,KSgSwC,P,Sw,Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRg_dSg(bag,KSgSg,KSgSgC,P,Sw,Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRg_dPbh(bag,KSgPbh,KSgPbhC,P,Sw,Sg,Pbh);CHKERRQ(ierr);
                    //
                    
                    ierr = FracDdRpbh_dP(bag,KPbhP,KPbhPC,P,Sw,Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRpbh_dSw(bag,KPbhSw,KPbhSwC,P,Sw,Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRpbh_dSg(bag,KPbhSg,KPbhSgC,P,Sw,Sg,Pbh);CHKERRQ(ierr);
                    ierr = FracDdRpbh_dPbh(bag,KPbhPbh,KPbhPbhC,P,Sw,Sg,Pbh);CHKERRQ(ierr);
                }
                else{
                    //                    ierr = FracDdRw_dP(bag,KPP,KPP,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //                    ierr = FracDdRw_dSw(bag,KPSw,KPSw,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //                    ierr = FracDdRw_dPbh(bag,KPPbh,KPPbh,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //
                    //                    ierr = FracDdRo_dP(bag,KSwP,KSwP,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //                    ierr = FracDdRo_dSw(bag,KSwSw,KSwSw,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //                    ierr = FracDdRo_dPbh(bag,KSwPbh,KSwPbh,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //
                    //                    ierr = FracDdRpbh_dP(bag,KPbhP,KPbhP,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //                    ierr = FracDdRpbh_dSw(bag,KPbhSw,KPbhSw,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                    //                    ierr = FracDdRpbh_dPbh(bag,KPbhPbh,KPbhPbh,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
                }
            }
            ierr = DMCompositeRestoreAccess(bag->MultiPhasePacker,V,&P,&Sw,&Sg,&Pbh);CHKERRQ(ierr);
            ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
            ierr = MatAssemblyEnd  (K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
            if (KPC != K) {
                ierr = MatAssemblyBegin(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                ierr = MatAssemblyEnd  (KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                
            }
            ierr = ISDestroy(&is[0]);CHKERRQ(ierr);
            ierr = ISDestroy(&is[1]);CHKERRQ(ierr);
            if(bag->numWells){
                ierr = ISDestroy(&is[2]);CHKERRQ(ierr);
            }
            ierr = PetscFree(is);CHKERRQ(ierr);
            ierr = MatDestroy(&KPP);CHKERRQ(ierr);
            ierr = MatDestroy(&KPSw);CHKERRQ(ierr);
            ierr = MatDestroy(&KPSg);CHKERRQ(ierr);
            ierr = MatDestroy(&KSwP);CHKERRQ(ierr);
            ierr = MatDestroy(&KSwSw);CHKERRQ(ierr);
            ierr = MatDestroy(&KSwSg);CHKERRQ(ierr);
            if (KPC != K) {
                ierr = MatDestroy(&KPPC);CHKERRQ(ierr);
                ierr = MatDestroy(&KPSwC);CHKERRQ(ierr);
                ierr = MatDestroy(&KPSgC);CHKERRQ(ierr);
                ierr = MatDestroy(&KSwPC);CHKERRQ(ierr);
                ierr = MatDestroy(&KSwSwC);CHKERRQ(ierr);
                ierr = MatDestroy(&KSwSgC);CHKERRQ(ierr);
            }
            if(bag->numWells){
                ierr = MatDestroy(&KPPbh);CHKERRQ(ierr);
                ierr = MatDestroy(&KSwPbh);CHKERRQ(ierr);
                ierr = MatDestroy(&KSgPbh);CHKERRQ(ierr);
                ierr = MatDestroy(&KPbhP);CHKERRQ(ierr);
                ierr = MatDestroy(&KPbhSw);CHKERRQ(ierr);
                ierr = MatDestroy(&KPbhSg);CHKERRQ(ierr);
                ierr = MatDestroy(&KPbhPbh);CHKERRQ(ierr);
                if (KPC != K) {
                    ierr = MatDestroy(&KPPbhC);CHKERRQ(ierr);
                    ierr = MatDestroy(&KSwPbhC);CHKERRQ(ierr);
                    ierr = MatDestroy(&KSgPbhC);CHKERRQ(ierr);
                    ierr = MatDestroy(&KPbhPC);CHKERRQ(ierr);
                    ierr = MatDestroy(&KPbhSwC);CHKERRQ(ierr);
                    ierr = MatDestroy(&KPbhSgC);CHKERRQ(ierr);
                    ierr = MatDestroy(&KPbhPbhC);CHKERRQ(ierr);
                }
            }
        }
        break;
    }
    /*
     PetscViewer viewer;
     ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"MatrixP11.txt",&viewer);CHKERRQ(ierr);
     ierr = MatView(K,viewer);CHKERRQ(ierr);
     */
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDPResidual"
extern PetscErrorCode FracDPResidual(SNES snesP,Vec V,Vec residual,void *user)
{
    
    PetscErrorCode ierr;
    
    AppCtx         *bag=(AppCtx*)user;
    Vec            Rw,Ro,Rg,RPbh;
    Vec            P,Sw=NULL,Sg=NULL,Pbh;
    PetscViewer viewer;
    
    PetscFunctionBegin;
    //    printf("\n\n In residual: %g %g %g\n\n\n\n\n",bag->ppties.OilPVTData.B_ModelData[0],bag->ppties.OilPVTData.B_ModelData[1],bag->ppties.OilPVTData.B_ModelData[2]);
    switch (bag->fluid) {
        case 0:
        {
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,V,&P,&Pbh);CHKERRQ(ierr);
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,residual,&Rw,&RPbh);CHKERRQ(ierr);
            ierr = FracDRw(bag,Rw,P,bag->fields.Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
            ierr = FracDRpbh(bag,RPbh,P,bag->fields.Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
            ierr = DMCompositeRestoreAccess(bag->MultiPhasePacker,V,&P,&Pbh);CHKERRQ(ierr);
            ierr = DMCompositeRestoreAccess(bag->MultiPhasePacker,residual,&Rw,&RPbh);CHKERRQ(ierr);
        }
        break;
        case 1:
        {
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,V,&P,&Pbh);CHKERRQ(ierr);
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,residual,&Rw,&RPbh);CHKERRQ(ierr);
            ierr = FracDRw(bag,Rw,P,bag->fields.Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
            ierr = FracDRpbh(bag,RPbh,P,bag->fields.Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
            ierr = DMCompositeRestoreAccess(bag->MultiPhasePacker,V,&P,&Pbh);CHKERRQ(ierr);
            ierr = DMCompositeRestoreAccess(bag->MultiPhasePacker,residual,&Rw,&RPbh);CHKERRQ(ierr);
        }
        break;
        case 2:
        {
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,V,&P,&Sw,&Pbh);CHKERRQ(ierr);
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,residual,&Rw,&Ro,&RPbh);CHKERRQ(ierr);
            ierr = FracDUpdateCapillaryPressureAndDerivative(bag->fields.Pcow,bag->fields.dervPcow,Sw,bag->SMALL_SATURATION,bag->ppties.CapPressData.Pcow_TableData,bag->ppties.CapPressData.numwaterdatarow);CHKERRQ(ierr);
            ierr = FracDRw(bag,Rw,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
            ierr = FracDRo(bag,Ro,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
            ierr = FracDRpbh(bag,RPbh,P,Sw,bag->fields.Sg,Pbh);CHKERRQ(ierr);
            //
            //            ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Pp1.txt",&viewer);CHKERRQ(ierr);
            //            ierr = VecView(P,viewer);CHKERRQ(ierr);
            ierr = DMCompositeRestoreAccess(bag->MultiPhasePacker,V,&P,&Sw,&Pbh);CHKERRQ(ierr);
            ierr = DMCompositeRestoreAccess(bag->MultiPhasePacker,residual,&Rw,&Ro,&RPbh);CHKERRQ(ierr);
        }
        break;
        case 3:
        {
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,V,&P,&Sw,&Sg,&Pbh);CHKERRQ(ierr);
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,residual,&Rw,&Ro,&Rg,&RPbh);CHKERRQ(ierr);
            ierr = FracDUpdateCapillaryPressureAndDerivative(bag->fields.Pcow,bag->fields.dervPcow,Sw,bag->SMALL_SATURATION,bag->ppties.CapPressData.Pcow_TableData,bag->ppties.CapPressData.numwaterdatarow);CHKERRQ(ierr);
            ierr = FracDUpdateCapillaryPressureAndDerivative(bag->fields.Pcog,bag->fields.dervPcog,Sg,bag->SMALL_SATURATION,bag->ppties.CapPressData.Pcog_TableData,bag->ppties.CapPressData.numwaterdatarow);CHKERRQ(ierr);
            
            ierr = FracDUpdateBubblePoint(bag->fields.Pb,bag->fields.SaturatedStateIndicator,P,bag->ppties.SolutionGasOilData.BubblePointFixed);CHKERRQ(ierr);
            ierr = FracDUpdateSolutionGasOilRatioAndDerivative(bag->fields.Rs,bag->fields.dervRs,bag->fields.Pb,P,bag->SMALL_PRESSURE,bag->ppties.SolutionGasOilData.SolutionGasOilRatioFixed,bag->ppties.SolutionGasOilData.ModelData,bag->ppties.SolutionGasOilData.TableData,bag->ppties.SolutionGasOilData.numdatarow);CHKERRQ(ierr);
            ierr = FracDRw(bag,Rw,P,Sw,Sg,Pbh);CHKERRQ(ierr);
            ierr = FracDRo(bag,Ro,P,Sw,Sg,Pbh);CHKERRQ(ierr);
            ierr = FracDRg(bag,Rg,P,Sw,Sg,Pbh);CHKERRQ(ierr);
            ierr = FracDRpbh(bag,RPbh,P,Sw,Sg,Pbh);CHKERRQ(ierr);
            
//            ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Sg3phase.txt",&viewer);CHKERRQ(ierr);
//            ierr = VecView(Sg,viewer);CHKERRQ(ierr);
            
            ierr = DMCompositeRestoreAccess(bag->MultiPhasePacker,V,&P,&Sw,&Sg,&Pbh);CHKERRQ(ierr);
            ierr = DMCompositeRestoreAccess(bag->MultiPhasePacker,residual,&Rw,&Ro,&Rg,&RPbh);CHKERRQ(ierr);
            
        }
        break;
    }
//        ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"PRes.txt",&viewer);CHKERRQ(ierr);
//        ierr = VecView(residual,viewer);CHKERRQ(ierr);
//    
//        ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Pp.txt",&viewer);CHKERRQ(ierr);
//        ierr = VecView(bag->fields.P,viewer);CHKERRQ(ierr);
//    
//        ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Rss.txt",&viewer);CHKERRQ(ierr);
//        ierr = VecView(bag->fields.Rs,viewer);CHKERRQ(ierr);
//    
//        ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Pbb.txt",&viewer);CHKERRQ(ierr);
//        ierr = VecView(bag->fields.Pb,viewer);CHKERRQ(ierr);
//    
//        ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"SaturatedStateIndicator.txt",&viewer);CHKERRQ(ierr);
//        ierr = VecView(bag->fields.SaturatedStateIndicator,viewer);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FracDSolveP"
extern PetscErrorCode FracDSolveP(AppCtx *bag)
{
    PetscErrorCode          ierr;
    SNESConvergedReason     reason;
    PetscInt                its;
    
    PetscFunctionBegin;
    ierr = SNESSolve(bag->snesP,PETSC_NULL,bag->fields.FlowPacker);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(bag->snesP,&reason);CHKERRQ(ierr);
    if (reason < 0) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"[ERROR] snesP diverged with reason %d\n",(int)reason);CHKERRQ(ierr);
    } else {
        ierr = SNESGetIterationNumber(bag->snesP,&its);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"      snesP converged in %d iterations %d.\n",(int)its,(int)reason);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDDiffusiveFluxMatrixLocal"
extern PetscErrorCode FracDDiffusiveFluxMatrixLocal(PetscReal *K_local, PetscReal *Conductivity, PetscReal *coords, FracDCVFEFace *f)
{
    PetscInt       i,j;
    PetscReal      height=0,*h,coeff=0;
    
    PetscFunctionBegin;
    h = (PetscReal *) malloc(f->dim*sizeof(PetscReal));
    for (i = 0; i < f->elemnodes; i++)    K_local[i] = 0;
    for (i = 0; i < f->dim; i++)
    {
        h[i] = f->int_point[i]-coords[i];
        height += h[i]*f->n[i];
    }
    for (i = 0; i < f->dim; i++) height += h[i]*f->n[i];
    
    if(height < 0){
        for (i = 0; i < f->dim; i++)    f->n[i] = -1*f->n[i];
    }
    for (i = 0; i < f->elemnodes; i++) {
        coeff = 0;
        for (j = 0; j < f->dim; j++) {
            coeff += Conductivity[j]*f->dphi[j][i]*f->n[j];
        }
        K_local[i] =  -1.*coeff * f->faceArea;
    }
    free(h);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDAdvectiveFluxMatrixLocal"
extern PetscErrorCode FracDAdvectiveFluxMatrixLocal(PetscReal *K_local, PetscReal rhoCw, PetscReal *V_array, PetscReal *coords, FracDCVFEFace *f)
{
    PetscInt       i,j;
    PetscScalar    v_elem[f->dim];
    PetscReal      height=0,*h,coeff=0;
    
    PetscFunctionBegin;
    h = (PetscReal *) malloc(f->dim*sizeof(PetscReal));
    for (i = 0; i < f->elemnodes; i++)    K_local[i] = 0;
    for (i = 0; i < f->dim; i++)
    {
        h[i] = f->int_point[i]-coords[i];
        height += h[i]*f->n[i];
    }
    for(i = 0; i < f->dim; i++){
        v_elem[i] = 0;
        for(j = 0; j < f->elemnodes; j++){
            v_elem[i] += V_array[i+j*f->dim]*f->phi[j];
        }
    }
    for (i = 0; i < f->elemnodes; i++) {
        coeff = 0;
        for (j = 0; j < f->dim; j++) {
            coeff += v_elem[j]*f->n[j];
        }
        K_local[i] =  rhoCw * coeff * f->faceArea;
    }
    free(h);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDOilWaterKrow"
extern PetscErrorCode FracDOilWaterKrow(PetscReal *kro, PetscReal Sw, PetscReal Sg, PetscReal extra, PetscReal **krowdata, PetscReal **krogdata, PetscReal **krwdata, PetscReal *stonedata, PetscInt numkrwdatarow, PetscInt numkrgdatarow)
{
    PetscErrorCode              ierr;
    
    PetscFunctionBegin;
    ierr =  FracDInterpolateUsingCubicSpline(kro,Sw,PETSC_NULL,krowdata,PETSC_NULL,numkrwdatarow);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDStone1Model"
extern PetscErrorCode FracDStone1Model(PetscReal *kro, PetscReal Sw, PetscReal Sg, PetscReal extra, PetscReal **krowdata, PetscReal **krogdata, PetscReal **krwdata, PetscReal *stonedata, PetscInt numkrwdatarow, PetscInt numkrgdatarow)
{
    PetscErrorCode              ierr;
    PetscReal Swc,Sor,Snw,Sno,Sng,Sdel,krow,krog,krc;
    PetscReal beta_w,beta_g;
    
    PetscFunctionBegin;
    Swc = stonedata[0];
    Sor = stonedata[1];
    krc = stonedata[2];
    Sdel = 1.0-Swc-Sor;
    Snw = (Sw-Swc)/Sdel;
    Sno = (1.0-Sw-Sg-Sor)/Sdel;
    Sng = Sg/Sdel;
    ierr =  FracDInterpolateUsingCubicSpline(&krow,Sw,PETSC_NULL,krowdata,PETSC_NULL,numkrwdatarow);CHKERRQ(ierr);
    ierr =  FracDInterpolateUsingCubicSpline(&krog,Sg,PETSC_NULL,krogdata,PETSC_NULL,numkrgdatarow);CHKERRQ(ierr);
    beta_w = krow/(1.-Snw)*1./krc;
    beta_g = krog/(1.-Sng)*1./krc;
    *kro = krc * Sno * beta_w * beta_g;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDUpdateCapillaryPressureAndDerivative"
extern PetscErrorCode FracDUpdateCapillaryPressureAndDerivative(Vec Pc, Vec dervPc, Vec S,  PetscReal delta, PetscReal **PcTableData, PetscInt numdatarow)
{
    PetscErrorCode      ierr;
    DM                  dm;
    PetscInt            v,vStart,vEnd,offset;
    PetscSection        localSection;
    PetscScalar         *Pc_array,*dervPc_array,*S_array;
    Vec                 local_Pc, local_dervPc, local_S;
    PetscReal           pcData[3],svalue;
    
    PetscFunctionBegin;
    
    ierr = VecGetDM(Pc,&dm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(dm,&localSection);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    
    ierr = VecSet(Pc,0.);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm,&local_Pc);CHKERRQ(ierr);
    ierr = VecSet(local_Pc,0.);CHKERRQ(ierr);
    ierr = VecGetArray(local_Pc,&Pc_array);CHKERRQ(ierr);

    ierr = VecSet(dervPc,0.);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm,&local_dervPc);CHKERRQ(ierr);
    ierr = VecSet(local_dervPc,0.);CHKERRQ(ierr);
    ierr = VecGetArray(local_dervPc,&dervPc_array);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(dm,&local_S);CHKERRQ(ierr);
    ierr = VecSet(local_S,0.);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,S,INSERT_VALUES,local_S);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,S,INSERT_VALUES,local_S);CHKERRQ(ierr);
    ierr = VecGetArray(local_S,&S_array);CHKERRQ(ierr);
    for(v = vStart; v < vEnd; ++v){
        ierr = PetscSectionGetOffset(localSection, v, &offset);CHKERRQ(ierr);
        svalue = S_array[offset];
        ierr =  FracDQuantityAndDerivativeComputation(pcData,svalue,delta,PETSC_NULL,PcTableData,PETSC_NULL,numdatarow,FracDInterpolateUsingCubicSpline);CHKERRQ(ierr);
        Pc_array[offset] = pcData[0];
        dervPc_array[offset] = pcData[1];
    }
    ierr = VecRestoreArray(local_Pc,&Pc_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm,local_Pc,INSERT_VALUES,Pc);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,local_Pc,INSERT_VALUES,Pc);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_Pc);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(local_dervPc,&dervPc_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm,local_dervPc,INSERT_VALUES,dervPc);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,local_dervPc,INSERT_VALUES,dervPc);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_dervPc);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(local_S,&S_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_S);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDUpdateBubblePoint"
extern PetscErrorCode FracDUpdateBubblePoint(Vec Pb, Vec Status, Vec P, PetscReal fixed_value)
{
    PetscErrorCode      ierr;
    DM                  dm;
    PetscInt            v,vStart,vEnd,offset;
    PetscSection        localSection;
    PetscScalar         *Pb_array,*P_array,*Status_array;
    Vec                 local_Pb, local_P, local_Status;
    PetscInt            count = 0;
    
    PetscFunctionBegin;
    ierr = VecGetDM(Pb,&dm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(dm,&localSection);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    
    ierr = VecSet(Pb,0.);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm,&local_Pb);CHKERRQ(ierr);
    ierr = VecSet(local_Pb,0.);CHKERRQ(ierr);
    ierr = VecGetArray(local_Pb,&Pb_array);CHKERRQ(ierr);
    
    ierr = VecSet(Status,0.);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm,&local_Status);CHKERRQ(ierr);
    ierr = VecSet(local_Status,0.);CHKERRQ(ierr);
    ierr = VecGetArray(local_Status,&Status_array);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(dm,&local_P);CHKERRQ(ierr);
    ierr = VecSet(local_P,0.);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,P,INSERT_VALUES,local_P);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,P,INSERT_VALUES,local_P);CHKERRQ(ierr);
    ierr = VecGetArray(local_P,&P_array);CHKERRQ(ierr);
    
    for(v = vStart; v < vEnd; ++v){
        ierr = PetscSectionGetOffset(localSection, v, &offset);CHKERRQ(ierr);
        if(P_array[offset] <= fixed_value){
            Pb_array[offset] = P_array[offset];
            Status_array[offset] = 1.0;
        }
        else{
            Pb_array[offset] = fixed_value;
            Status_array[offset] = 0.0;
            count++;
        }
    }
    
    ierr = VecRestoreArray(local_Pb,&Pb_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm,local_Pb,INSERT_VALUES,Pb);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,local_Pb,INSERT_VALUES,Pb);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_Pb);CHKERRQ(ierr);

    ierr = VecRestoreArray(local_Status,&Status_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm,local_Status,INSERT_VALUES,Status);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,local_Status,INSERT_VALUES,Status);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_Status);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(local_P,&P_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_P);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDUpdateSolutionGasOilRatioAndDerivative"
extern PetscErrorCode FracDUpdateSolutionGasOilRatioAndDerivative(Vec Rs, Vec dervRs, Vec Pb, Vec P, PetscReal delta, PetscReal fixed_value, PetscReal *model_data, PetscReal **TableData, PetscInt numdatarow)
{
    PetscErrorCode      ierr;
    DM                  dm;
    PetscInt            v,vStart,vEnd,offset;
    PetscSection        localSection;
    PetscScalar         *Rs_array,*dervRs_array,*P_array,*Pb_array;
    Vec                 local_Rs, local_dervRs, local_P, local_Pb;
    PetscReal           RsData[3];
    
    PetscFunctionBegin;
    
    ierr = VecGetDM(Rs,&dm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(dm,&localSection);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    
    ierr = VecSet(Rs,0.);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm,&local_Rs);CHKERRQ(ierr);
    ierr = VecSet(local_Rs,0.);CHKERRQ(ierr);
    ierr = VecGetArray(local_Rs,&Rs_array);CHKERRQ(ierr);
    
    ierr = VecSet(dervRs,0.);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm,&local_dervRs);CHKERRQ(ierr);
    ierr = VecSet(local_dervRs,0.);CHKERRQ(ierr);
    ierr = VecGetArray(local_dervRs,&dervRs_array);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(dm,&local_Pb);CHKERRQ(ierr);
    ierr = VecSet(local_Pb,0.);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,Pb,INSERT_VALUES,local_Pb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,Pb,INSERT_VALUES,local_Pb);CHKERRQ(ierr);
    ierr = VecGetArray(local_Pb,&Pb_array);CHKERRQ(ierr);

    ierr = DMGetLocalVector(dm,&local_P);CHKERRQ(ierr);
    ierr = VecSet(local_P,0.);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,P,INSERT_VALUES,local_P);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,P,INSERT_VALUES,local_P);CHKERRQ(ierr);
    ierr = VecGetArray(local_P,&P_array);CHKERRQ(ierr);

    for(v = vStart; v < vEnd; ++v){
        ierr = PetscSectionGetOffset(localSection, v, &offset);CHKERRQ(ierr);
        ierr =  FracDQuantityAndDerivativeComputation(RsData,P_array[offset],delta,Pb_array[offset],TableData,model_data,numdatarow,FracDInterpolateUsingCubicSpline);CHKERRQ(ierr);
        Rs_array[offset] = RsData[0];
        dervRs_array[offset] = RsData[1];
    }
    ierr = VecRestoreArray(local_Rs,&Rs_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm,local_Rs,INSERT_VALUES,Rs);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,local_Rs,INSERT_VALUES,Rs);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_Rs);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(local_dervRs,&dervRs_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm,local_dervRs,INSERT_VALUES,dervRs);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,local_dervRs,INSERT_VALUES,dervRs);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_dervRs);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(local_Pb,&Pb_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_Pb);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(local_P,&P_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_P);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


/*
#undef __FUNCT__
#define __FUNCT__ "FracDInitializeSolutionGasOilRatio"
extern PetscErrorCode FracDInitializeSolutionGasOilRatio(Vec Rs, Vec P, Vec Sw, Vec Sg, FracDPVT OilPVTData, FracDPVT GasPVTData, PetscReal ini_Rs, PetscReal ini_pb)
{
    PetscErrorCode      ierr;
    DM                  dm;
    PetscInt            v,vStart,vEnd,offset;
    PetscSection        localSection;
    PetscScalar         *Rs_array,*Sw_array,*Sg_array, *P_array,;
    Vec                 local_Rs, local_Sw, local_Sg, local_P;
    PetscReal           So,Bo,Bg;
    
    PetscFunctionBegin;
    
    ierr = VecGetDM(Rs,&dm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(dm,&localSection);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    
    ierr = VecSet(Rs,0.);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm,&local_Rs);CHKERRQ(ierr);
    ierr = VecSet(local_Rs,0.);CHKERRQ(ierr);
    ierr = VecGetArray(local_Rs,&Rs_array);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(dm,&local_Sw);CHKERRQ(ierr);
    ierr = VecSet(local_Sw,0.);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,Sw,INSERT_VALUES,local_Sw);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,Sw,INSERT_VALUES,local_Sw);CHKERRQ(ierr);
    ierr = VecGetArray(local_Sw,&Sw_array);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(dm,&local_Sg);CHKERRQ(ierr);
    ierr = VecSet(local_Sg,0.);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,Sg,INSERT_VALUES,local_Sg);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,Sg,INSERT_VALUES,local_Sg);CHKERRQ(ierr);
    ierr = VecGetArray(local_Sg,&Sg_array);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(dm,&local_P);CHKERRQ(ierr);
    ierr = VecSet(local_P,0.);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,P,INSERT_VALUES,local_P);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,P,INSERT_VALUES,local_P);CHKERRQ(ierr);
    ierr = VecGetArray(local_P,&P_array);CHKERRQ(ierr);
    
    for(v = vStart; v < vEnd; ++v){
        ierr = PetscSectionGetOffset(localSection, v, &offset);CHKERRQ(ierr);
        So = 1.-Sw_array[offset]-Sg_array[offset];
        ierr =  OilPVTData.FracDUpDateFVF(&Bo,P_array[offset],ini_pb,OilPVTData.B_TableData,OilPVTData.B_ModelData,OilPVTData.numdatarow);CHKERRQ(ierr);
        ierr =  GasPVTData.FracDUpDateFVF(&Bg,P_array[offset],PETSC_NULL,GasPVTData.B_TableData,GasPVTData.B_ModelData,GasPVTData.numdatarow);CHKERRQ(ierr);
        Rs_array[offset] = ini_Rs + Sg_array[offset]*Bo/(So*Bg);
    }
    ierr = VecRestoreArray(local_Rs,&Rs_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm,local_Rs,INSERT_VALUES,Rs);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,local_Rs,INSERT_VALUES,Rs);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_Rs);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(local_Sw,&Sw_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_Sw);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(local_Sg,&Sg_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_Sg);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(local_P,&P_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_P);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

*/






#undef __FUNCT__
#define __FUNCT__ "FracDInitializeSaturationBubblePointAndSwitchingVariables"
extern PetscErrorCode FracDInitializeSaturationBubblePointAndSwitchingVariables(Vec INDC1, Vec INDC2, Vec Rs, Vec P, Vec Pb, Vec Sw, Vec Sg,  FracDPVT OilPVTData, FracDPVT GasPVTData, FracDPbRs            SolutionGasOilData)
{
    PetscErrorCode      ierr;
    DM                  dm;
    PetscInt            v,vStart,vEnd,offset;
    PetscSection        localSection;
    PetscScalar         *IND1_array,*IND2_array,*Sw_array,*Sg_array,*P_array,*Pb_array,*Rs_array;
    Vec                 local_IND1, local_IND2, local_Sw, local_Sg, local_P, local_Pb, local_Rs;
    PetscReal           So,Bo,Bg,pb_value;
    
    PetscFunctionBegin;
    
    ierr = VecGetDM(Sg,&dm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(dm,&localSection);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    
    ierr = VecSet(INDC1,0.);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm,&local_IND1);CHKERRQ(ierr);
    ierr = VecSet(local_IND1,0.);CHKERRQ(ierr);
    ierr = VecGetArray(local_IND1,&IND1_array);CHKERRQ(ierr);
    
    ierr = VecSet(INDC2,0.);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm,&local_IND2);CHKERRQ(ierr);
    ierr = VecSet(local_IND2,0.);CHKERRQ(ierr);
    ierr = VecGetArray(local_IND2,&IND2_array);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(dm,&local_Rs);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,Rs,INSERT_VALUES,local_Rs);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,Rs,INSERT_VALUES,local_Rs);CHKERRQ(ierr);
    ierr = VecGetArray(local_Rs,&Rs_array);CHKERRQ(ierr);
    
    ierr = VecSet(Pb,0.);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm,&local_Pb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,Pb,INSERT_VALUES,local_Pb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,Pb,INSERT_VALUES,local_Pb);CHKERRQ(ierr);
    ierr = VecGetArray(local_Pb,&Pb_array);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(dm,&local_P);CHKERRQ(ierr);
    ierr = VecSet(local_P,0.);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,P,INSERT_VALUES,local_P);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,P,INSERT_VALUES,local_P);CHKERRQ(ierr);
    ierr = VecGetArray(local_P,&P_array);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(dm,&local_Sw);CHKERRQ(ierr);
    ierr = VecSet(local_Sw,0.);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,Sw,INSERT_VALUES,local_Sw);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,Sw,INSERT_VALUES,local_Sw);CHKERRQ(ierr);
    ierr = VecGetArray(local_Sw,&Sw_array);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(dm,&local_Sg);CHKERRQ(ierr);
    ierr = VecSet(local_Sg,0.);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,Sg,INSERT_VALUES,local_Sg);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,Sg,INSERT_VALUES,local_Sg);CHKERRQ(ierr);
    ierr = VecGetArray(local_Sg,&Sg_array);CHKERRQ(ierr);
    
    for(v = vStart; v < vEnd; ++v){
        ierr = PetscSectionGetOffset(localSection, v, &offset);CHKERRQ(ierr);
        if(Sg_array[offset] > 0.){
            IND1_array[offset] = IND2_array[offset] = 1.;
            So = 1.-Sw_array[offset]-Sg_array[offset];
            ierr = OilPVTData.FracDUpDateFVF(&Bo,P_array[offset],Pb_array[offset],OilPVTData.B_TableData,OilPVTData.B_ModelData,OilPVTData.numdatarow);CHKERRQ(ierr);
            ierr = GasPVTData.FracDUpDateFVF(&Bg,P_array[offset],PETSC_NULL,GasPVTData.B_TableData,GasPVTData.B_ModelData,GasPVTData.numdatarow);CHKERRQ(ierr);
            Rs_array[offset] += Sg_array[offset] * Bo/(So*Bg);
        }
        else{
            IND1_array[offset] = 0.;
            IND2_array[offset] = 3.;
        }
        if(SolutionGasOilData.isbubblepointvarying == PETSC_FALSE){
            Pb_array[offset] = SolutionGasOilData.BubblePointFixed;
        }
        else{
            pb_value = 0;
            ierr = SolutionGasOilData.FracDUpDateBubblePoint(&pb_value,Rs_array[offset],PETSC_NULL,SolutionGasOilData.TableDataInv,PETSC_NULL,SolutionGasOilData.numdatarow);CHKERRQ(ierr);
            Pb_array[offset] = pb_value;
        }
    }
    ierr = VecRestoreArray(local_IND1,&IND1_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm,local_IND1,INSERT_VALUES,INDC1);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,local_IND1,INSERT_VALUES,INDC1);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_IND1);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(local_IND2,&IND2_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm,local_IND2,INSERT_VALUES,INDC2);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,local_IND2,INSERT_VALUES,INDC2);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_IND2);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(local_Rs,&Rs_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm,local_Rs,INSERT_VALUES,Rs);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,local_Rs,INSERT_VALUES,Rs);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_Rs);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(local_Pb,&Pb_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm,local_Pb,INSERT_VALUES,Pb);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,local_Pb,INSERT_VALUES,Pb);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_Pb);CHKERRQ(ierr);

    ierr = VecRestoreArray(local_P,&P_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_P);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(local_Sw,&Sw_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_Sw);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(local_Sg,&Sg_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_Sg);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDInitializePackedSgRsField"
extern PetscErrorCode FracDInitializePackedSgRsField(Vec SgRs, Vec Rs, Vec Sg, Vec INDC2)
{
    PetscErrorCode      ierr;
    DM                  dm;
    PetscInt            v,vStart,vEnd,offset;
    PetscSection        localSection;
    PetscScalar         *IND2_array,*Sg_array,*Rs_array,*SgRs_array;
    Vec                 local_IND2,local_Sg,local_Rs,local_SgRs;
    
    PetscFunctionBegin;
    ierr = VecGetDM(Sg,&dm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(dm,&localSection);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    
    ierr = VecSet(SgRs,0.);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm,&local_SgRs);CHKERRQ(ierr);
    ierr = VecSet(local_SgRs,0.);CHKERRQ(ierr);
    ierr = VecGetArray(local_SgRs,&SgRs_array);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(dm,&local_IND2);CHKERRQ(ierr);
    ierr = VecSet(local_IND2,0.);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,INDC2,INSERT_VALUES,local_IND2);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,INDC2,INSERT_VALUES,local_IND2);CHKERRQ(ierr);
    ierr = VecGetArray(local_IND2,&IND2_array);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(dm,&local_Rs);CHKERRQ(ierr);
    ierr = VecSet(local_Rs,0.);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,Rs,INSERT_VALUES,local_Rs);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,Rs,INSERT_VALUES,local_Rs);CHKERRQ(ierr);
    ierr = VecGetArray(local_Rs,&Rs_array);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(dm,&local_Sg);CHKERRQ(ierr);
    ierr = VecSet(local_Sg,0.);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,Sg,INSERT_VALUES,local_Sg);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,Sg,INSERT_VALUES,local_Sg);CHKERRQ(ierr);
    ierr = VecGetArray(local_Sg,&Sg_array);CHKERRQ(ierr);
    
    for(v = vStart; v < vEnd; ++v){
        ierr = PetscSectionGetOffset(localSection, v, &offset);CHKERRQ(ierr);
        if( PetscAbs(IND2_array[offset]-1.) < PETSC_SMALL){
            SgRs_array[offset] = Sg_array[offset];
        }
        if( PetscAbs(IND2_array[offset]-3.) < PETSC_SMALL){
            SgRs_array[offset] = Rs_array[offset];
        }
    }
    
    ierr = VecRestoreArray(local_SgRs,&SgRs_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm,local_SgRs,INSERT_VALUES,SgRs);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,local_SgRs,INSERT_VALUES,SgRs);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_SgRs);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(local_IND2,&IND2_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_IND2);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(local_Rs,&Rs_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_Rs);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(local_Sg,&Sg_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&local_Sg);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}



