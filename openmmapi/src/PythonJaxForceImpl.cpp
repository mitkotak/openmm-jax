#include "internal/PythonJaxForceImpl.h"
#include "JaxKernels.h"
#include "openmm/internal/ContextImpl.h"

using namespace JaxPlugin;
using namespace OpenMM;
using namespace std;

PythonJaxForceImpl::PythonJaxForceImpl(const PythonJaxForce& owner) : owner(owner) {
}

void PythonJaxForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcPythonJaxForceKernel::Name(), context);
    kernel.getAs<CalcPythonJaxForceKernel>().initialize(context, owner);
}

double PythonJaxForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces,
        bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcPythonJaxForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

vector<string> PythonJaxForceImpl::getKernelNames() {
    vector<string> names;
    names.push_back(CalcPythonJaxForceKernel::Name());
    return names;
}

map<string, double> PythonJaxForceImpl::getDefaultParameters() {
    return owner.getGlobalParameters();
}
