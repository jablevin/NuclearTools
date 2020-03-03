#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

typedef double real;
typedef struct { real x, y; } VecR;
typedef struct { VecR r, rv, ra; } Mol;
typedef struct { int x, y; } VecI;
typedef struct { real val, sum, sum2; } Prop;

// Random Constants
#define MASK 2147483647
#define SCALE 0.4656612873e-9
#define IADD 453806245
#define IMUL 314159269
#define M_PI 3.14159269

// Dimensionally-Based Macros
#define NDIM 2
#define VVAdd(v1, v2) VAdd (v1, v1, v2)
#define VAdd(v1, v2, v3) (v1).x = (v2).x + (v3).x, (v1).y = (v2).y + (v3).y
#define VSAdd(v1, v2, s3, v3) (v1).x = (v2).x + (s3) * (v3).x, (v1).y = (v2).y + (s3) * (v3).y
#define VVSAdd(v1, s2, v2) VSAdd (v1, v1, s2, v2)
#define VSub(v1, v2, v3) (v1).x = (v2).x - (v3).x, (v1).y = (v2).y - (v3).y
#define VDot(v1, v2) ((v1).x * (v2).x + (v1).y * (v2).y)
#define VLen(v) sqrt (VDot (v, v))
#define VLenSq(v) VDot (v, v)
#define VSet(v, sx, sy) (v).x = sx, (v).y = sy
#define VSetAll(v, s) VSet (v, s, s)
#define VZero(v) VSetAll (v, 0)
#define VMul(v1, v2, v3) (v1).x = (v2).x * (v3).x, (v1).y = (v2).y * (v3).y
#define VDiv(v1, v2, v3) (v1).x = (v2).x / (v3).x, (v1).y = (v2).y / (v3).y
#define VCSum(v) ((v).x + (v).y)
#define VSCopy(v2, s1, v1) (v2).x = (s1) * (v1).x, (v2).y = (s1) * (v1).y
#define VProd(v) ((v).x * (v).y)
#define VScale(v, s) (v).x *= s, (v).y *= s
#define VWrapAll(v) {VWrap (v, x); VWrap (v, y);}
#define VWrap(v, t) if (v.t >= 0.5 * region.t) v.t -= region.t; \
                    else if (v.t < -0.5 * region.t) v.t += region.t

// Other Macros
#define Max(x, y) (((x) > (y)) ? (x) : (y))
#define Min(x, y) (((x) < (y)) ? (x) : (y))
#define Sqr(x) ((x) * (x))
#define Cube(x) ((x) * (x) * (x))
#define DO_MOL for (n = 0; n < nMol; n ++)
#define AllocMem(a, n, t) a = (t *) malloc ((n) * sizeof (t))
#define PropZero(v) v.sum = 0., v.sum2 = 0.
#define PropAccum(v) v.sum += v.val, v.sum2 += Sqr (v.val)
#define PropAvg(v, n) v.sum /= n, v.sum2 = sqrt(Max(v.sum2/n - Sqr (v.sum), 0.))
#define PropEst(v) v.sum, v.sum2

// Variables
Mol *mol;
VecR region, vSum;
VecI initUcell;
Prop kinEnergy, pressure, totEnergy;
real deltaT, density, rCut, temperature, timeNow, uSum, velMag, virSum, vvSum;
real *histVel, rangeVel, hFunction;
int moreCycles, nMol, stepAvg, stepCount, stepEquil, stepLimit, randseedP;
int countVel, limitVel, sizeHistVel, stepVel, randSeedP, printout;
FILE *file;

// Function Declarations
void SingleStep(), SetParams(), SetupJob(), LeapfrogStep();
void ApplyBoundaryCond(), ComputeForces(), EvalProps(), VRand();
void AccumProps(), PrintSummary(), InitCoords(), PrintVelDist();
void EvalVelDist(), AllocArrays (), InitVels (), InitAccels (), InitRand ();
real RandR();

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Begin Main Program
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static PyObject* MD2D(PyObject* self, PyObject* args){
    FILE* in_file;
    char *s;
    if (!PyArg_ParseTuple(args, "s", &s)) {
        return NULL;}
    if ((in_file = fopen(s,"r")) == NULL){
        printf("Error! opening file");
        exit(1);}
        char* pend;
        char array[50], outfile[50];
        fscanf(in_file, "%s", array);
        deltaT = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        density = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        initUcell.x = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        initUcell.y = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        stepAvg = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        stepEquil = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        stepLimit = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        temperature = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        limitVel = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        rangeVel = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        sizeHistVel = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        stepVel = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        randseedP = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        printout = strtof(array, &pend);
        fscanf(in_file, "%s", outfile);
    file = fopen(outfile,"w");
    SetParams ();
    SetupJob ();
    moreCycles = 1;
    while (moreCycles) {
        SingleStep ();
        if (stepCount >= stepLimit) moreCycles = 0;}
    return Py_None;}


void SingleStep (){
    ++ stepCount;
    timeNow = stepCount * deltaT;
    LeapfrogStep (1);
    ApplyBoundaryCond ();
    ComputeForces ();
    LeapfrogStep (2);
    EvalProps ();
    AccumProps (1);
    if (stepCount % stepAvg == 0) {
        AccumProps (2);
        PrintSummary ();
        AccumProps (0);}
    if (stepCount >= stepEquil &&
        (stepCount - stepEquil) % stepVel == 0) EvalVelDist ();}


void SetupJob (){
    countVel = 0;
    AllocArrays ();
    stepCount = 0;
    InitCoords ();
    InitVels ();
    InitAccels ();
    AccumProps (0);}
    // InitRand (randSeed);}


void AllocArrays (){
    AllocMem (mol, nMol, Mol);
    AllocMem (histVel, sizeHistVel, real);}


void ComputeForces (){
    VecR dr;
    real fcVal, rr, rrCut, rri, rri3;
    int j1, j2, n;
    rrCut = Sqr (rCut);
    DO_MOL VZero (mol[n].ra);
    uSum = 0.;
    virSum = 0.;
    for (j1 = 0; j1 < nMol - 1; j1 ++) {
        for (j2 = j1 + 1; j2 < nMol; j2 ++) {
            VSub (dr, mol[j1].r, mol[j2].r);
            VWrapAll (dr);
            rr = VLenSq (dr);
            if (rr < rrCut) {
                rri = 1. / rr;
                rri3 = Cube (rri);
                fcVal = 48. * rri3 * (rri3 - 0.5) * rri;
                VVSAdd (mol[j1].ra, fcVal, dr);
                VVSAdd (mol[j2].ra, - fcVal, dr);
                uSum += 4. * rri3 * (rri3 - 1.) + 1.;
                virSum += fcVal * rr;}}}}


void LeapfrogStep (int part){
    int n;
    if (part == 1) {
        DO_MOL {
            VVSAdd (mol[n].rv, 0.5 * deltaT, mol[n].ra);
            VVSAdd (mol[n].r, deltaT, mol[n].rv);}}
    else {
        DO_MOL VVSAdd (mol[n].rv, 0.5 * deltaT, mol[n].ra);}}


void ApplyBoundaryCond (){
    int n;
    DO_MOL VWrapAll (mol[n].r);}


void InitCoords (){
    VecR c, gap;
    int n = 0, nx, ny;
    VDiv (gap, region, initUcell);
    for (ny = 0; ny < initUcell.y; ny ++) {
        for (nx = 0; nx < initUcell.x; nx ++) {
            VSet (c, nx + 0.5, ny + 0.5);
            VMul (c, c, gap);
            VVSAdd (c, -0.5, region);
            mol[n].r = c;
            ++ n;}}}


void InitVels (){
    int n;
    VZero (vSum);
    DO_MOL {
        VRand (&mol[n].rv);
        VScale (mol[n].rv, velMag);
        VVAdd (vSum, mol[n].rv);}
    DO_MOL VVSAdd (mol[n].rv, - 1. / nMol, vSum);}


void VRand (VecR *p){
    real s = 2. * M_PI * RandR ();
    p->x = cos (s);
    p->y = sin (s);}


real RandR (){
    randSeedP = (randSeedP * IMUL + IADD) & MASK;
    return (randSeedP * SCALE);}


void InitAccels (){
    int n;
    DO_MOL VZero (mol[n].ra);}


void SetParams (){
    rCut = pow (2., 1./6.);
    VSCopy (region, 1. / sqrt (density), initUcell);
    nMol = VProd (initUcell);
    velMag = sqrt (NDIM * (1. - 1. / nMol) * temperature);}


void EvalProps (){
    real vv;
    int n;
    VZero (vSum);
    vvSum = 0.;
    DO_MOL {
        VVAdd (vSum, mol[n].rv);
        vv = VLenSq (mol[n].rv);
        vvSum += vv;}
    kinEnergy.val = 0.5 * vvSum / nMol;
    totEnergy.val = kinEnergy.val + uSum / nMol;
    pressure.val = density * (vvSum + virSum) / (nMol * NDIM);}


void AccumProps (int icode){
    if (icode == 0) {
        PropZero (totEnergy);
        PropZero (kinEnergy);
        PropZero (pressure);
    } else if (icode == 1) {
        PropAccum (totEnergy);
        PropAccum (kinEnergy);
        PropAccum (pressure);
    } else if (icode == 2) {
        PropAvg (totEnergy, stepAvg);
        PropAvg (kinEnergy, stepAvg);
        PropAvg (pressure, stepAvg);}}


void EvalVelDist (){
    real deltaV, histSum;
    int j, n;
    if (countVel == 0) {
        for (j = 0; j < sizeHistVel; j ++) histVel[j] = 0.;}
    deltaV = rangeVel / sizeHistVel;
    DO_MOL {
        j = VLen (mol[n].rv) / deltaV;
        ++ histVel[Min (j, sizeHistVel - 1)];}
    ++ countVel;
    if (countVel == limitVel) {
        histSum = 0.;
        for (j = 0; j < sizeHistVel; j ++) histSum += histVel[j];
        for (j = 0; j < sizeHistVel; j ++) histVel[j] /= histSum;
        PrintVelDist ();
        countVel = 0;
        hFunction = 0.;
        for (j = 0; j < sizeHistVel; j ++) {
            if (histVel[j] > 0.) hFunction += histVel[j] * log (histVel[j] /
                ((j + 0.5) * deltaV));}}}


void PrintVelDist (){
    real vBin;
    int n;
    if (printout == 1){
        printf ("vdist (%.3f)\n", timeNow);}
    for (n = 0; n < sizeHistVel; n ++) {
        vBin = (n + 0.5) * rangeVel / sizeHistVel;
        fprintf (file, "%8.3f %8.3f\n", vBin, histVel[n]);}
    fprintf (file, "hfun: %8.3f %8.3f\n", timeNow, hFunction);}


void PrintSummary (){
    fprintf (file, "Summary:");
    fprintf (file,
    "%5d %8.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f\n",
    stepCount, timeNow, VCSum (vSum) / nMol, PropEst (totEnergy),
    PropEst (kinEnergy), PropEst (pressure), uSum, vvSum);}



static PyMethodDef myMethods[] = {
    { "MD2D", MD2D, METH_VARARGS, "Solves Molecular Dynamics 2D" },
    { NULL, NULL, 0, NULL }};


static struct PyModuleDef MolecDy = {
    PyModuleDef_HEAD_INIT,
    "MolecDy",
    "Dynamics Simulation",
    -1,
    myMethods};


PyMODINIT_FUNC PyInit_MD2D(void){
    return PyModule_Create(&MolecDy);}
