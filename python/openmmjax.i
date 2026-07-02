%module openmmjax

%pythonbegin %{
# Load OpenMM first so its package-specific library paths are available before
# the OpenMM-JAX extension resolves libOpenMM.so.
import openmm as _openmm
%}

%include "factory.i"
%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>
%include <std_vector.i>

namespace std {
    %template(vectori) vector<int>;
}

%{
#include "JaxForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
#include <stdexcept>
%}

%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

namespace JaxPlugin {

class JaxForce : public OpenMM::Force {
public:
    JaxForce(const std::string& forceMlir, const std::string& energyMlir,
             const std::string& energyAndForcesMlir,
             const std::string& compileOptionsBase64);
    void setUsesPeriodicBoundaryConditions(bool periodic);
    bool usesPeriodicBoundaryConditions() const;
    void setPjrtPluginPath(const std::string& path);
    const std::string& getPjrtPluginPath() const;
    void setParticles(const std::vector<int>& particles);
    const std::vector<int>& getParticles() const;

    %extend {
        static JaxPlugin::JaxForce& cast(OpenMM::Force& force) {
            return dynamic_cast<JaxPlugin::JaxForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<JaxPlugin::JaxForce*>(&force) != NULL);
        }
    }
};

}

%pythoncode %{
def _load_bundled_platform_plugins():
    import ctypes as _ctypes
    import platform as _platform
    from pathlib import Path as _Path

    import openmm as _openmm

    def _has_cuda_driver():
        system = _platform.system()
        if system == "Windows":
            driver_names = ("nvcuda.dll",)
        elif system == "Darwin":
            driver_names = ()
        else:
            driver_names = ("libcuda.so.1", "libcuda.so")
        for name in driver_names:
            try:
                _ctypes.CDLL(name)
                return True
            except OSError:
                pass
        return False

    if not _has_cuda_driver():
        return

    module_dir = _Path(__file__).resolve().parent
    system = _platform.system()
    if system == "Windows":
        names = ("OpenMMJaxCUDA.dll",)
    elif system == "Darwin":
        names = ("libOpenMMJaxCUDA.dylib",)
    else:
        names = ("libOpenMMJaxCUDA.so",)
    for name in names:
        path = module_dir / name
        if path.is_file():
            _openmm.Platform.loadPluginLibrary(str(path))


_load_bundled_platform_plugins()
%}
