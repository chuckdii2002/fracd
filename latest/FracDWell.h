#ifndef FRACDWELL_H
#define FRACDWELL_H
/*
 FracDWell.h
 (c) 2016-2018 Chukwudi CHukwudozie chdozie@gmail.com
 */

typedef enum {
    PRESSURE,
    WATERRATE,
    OILRATE,
    GASRATE,
    LIQUIDRATE,
    TOTALRATE
} FracDWellConstraint;

typedef enum {
    INJECTORWATER,
    INJECTORGAS,
    PRODUCER
} FracDWellType;

typedef struct {
    char           name[PETSC_MAX_PATH_LEN];
    PetscInt       dim;
    PetscReal      *top;
    PetscReal      *bottom;
    PetscReal      *coordinates;
    PetscReal      Qws;
    PetscReal      Qos;
    PetscReal      Qgs;
    PetscReal      QL;
    PetscReal      QT;
    PetscReal	   Pbh;
    PetscReal	   minPbh;
    PetscReal      h;
    PetscReal      rw;
    PetscReal      re;
    PetscReal      sk;
    FracDWellConstraint condition;
    FracDWellType  type;
    PetscInt       index;
} FracDWell;

//#ifndef FracDWellDescription
extern PetscErrorCode FracDGetWell(const char prefix[],FracDWell *well);
extern PetscErrorCode FracDWellCreate(FracDWell *well, PetscInt dim);
extern PetscErrorCode FracDWellView(FracDWell *well,PetscViewer viewer);
extern PetscErrorCode FracDWellDestroy(FracDWell *well, PetscInt dim);
//#endif
#endif /* FRACDWELL_H */
