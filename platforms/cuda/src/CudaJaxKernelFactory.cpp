#include "CudaJaxKernelFactory.h"
#include "CudaJaxKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <exception>
using namespace JaxPlugin;
using namespace OpenMM;
using namespace std;

extern "C" void registerPlatforms() {
}

extern "C" void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        platform.registerKernelFactory(CalcJaxForceKernel::Name(), new CudaJaxKernelFactory());
        platform.registerKernelFactory(CalcPythonJaxForceKernel::Name(), new CudaJaxKernelFactory());
    }
    catch (std::exception& ex) {
        // Ignore
    }
}

extern "C" void registerJaxCudaKernelFactories() {
    try {
        Platform::getPlatformByName("CUDA");
    }
    catch (...) {
        Platform::registerPlatform(new CudaPlatform());
    }
    registerKernelFactories();
}

KernelImpl* CudaJaxKernelFactory::createKernelImpl(string name, const Platform& platform, ContextImpl& context) const {
    CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == CalcJaxForceKernel::Name())
        return new CudaCalcJaxForceKernel(name, platform, cu);
    if (name == CalcPythonJaxForceKernel::Name())
        return new CudaCalcPythonJaxForceKernel(name, platform, cu);
    std::string ex_msg = "Tried to create kernel with illegal kernel name '"+name+"'";
    throw OpenMMException(ex_msg);
}
