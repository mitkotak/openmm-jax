#include "internal/JaxForceImpl.h"
#include "JaxKernels.h"
#include "openmm/internal/ContextImpl.h"

using namespace JaxPlugin;
using namespace OpenMM;
using namespace std;

JaxForceImpl::JaxForceImpl(const JaxForce& owner) : owner(owner) {
}

void JaxForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcJaxForceKernel::Name(), context);
    kernel.getAs<CalcJaxForceKernel>().initialize(context.getSystem(), owner);
}

double JaxForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcJaxForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

vector<string> JaxForceImpl::getKernelNames() {
    vector<string> names;
    names.push_back(CalcJaxForceKernel::Name());
    return names;
}

map<string, double> JaxForceImpl::getDefaultParameters() {
    // No support for global parameters currently
    return {};
}
