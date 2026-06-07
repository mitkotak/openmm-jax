extern "C" __global__
void copyInputs(real* __restrict__ packedPositions, real* __restrict__ boxVectors,
        const real4* __restrict__ posq, const int* __restrict__ atomIndex,
        int numAtoms, bool usePeriodic, real4 periodicBoxVecX,
        real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    if (usePeriodic && blockIdx.x == 0 && threadIdx.x == 0) {
        boxVectors[0] = periodicBoxVecX.x;
        boxVectors[1] = periodicBoxVecX.y;
        boxVectors[2] = periodicBoxVecX.z;
        boxVectors[3] = periodicBoxVecY.x;
        boxVectors[4] = periodicBoxVecY.y;
        boxVectors[5] = periodicBoxVecY.z;
        boxVectors[6] = periodicBoxVecZ.x;
        boxVectors[7] = periodicBoxVecZ.y;
        boxVectors[8] = periodicBoxVecZ.z;
    }
    for (int atom = blockIdx.x*blockDim.x+threadIdx.x; atom < numAtoms; atom += blockDim.x*gridDim.x) {
        int index = atomIndex[atom];
        real4 p = posq[atom];
        packedPositions[3*index] = p.x;
        packedPositions[3*index+1] = p.y;
        packedPositions[3*index+2] = p.z;
    }
}

extern "C" __global__
void addForces(const real* __restrict__ forces, long long* __restrict__ forceBuffers,
        const int* __restrict__ atomIndex, int numAtoms, int paddedNumAtoms, int forceSign) {
    for (int atom = blockIdx.x*blockDim.x+threadIdx.x; atom < numAtoms; atom += blockDim.x*gridDim.x) {
        int index = atomIndex[atom];
        forceBuffers[atom] += (long long) (forceSign*forces[3*index]*0x100000000);
        forceBuffers[atom+paddedNumAtoms] += (long long) (forceSign*forces[3*index+1]*0x100000000);
        forceBuffers[atom+2*paddedNumAtoms] += (long long) (forceSign*forces[3*index+2]*0x100000000);
    }
}
