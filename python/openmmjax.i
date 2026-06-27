%module openmmjax

%include "factory.i"
%import(module="openmm.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>

%{
#include "JaxForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
#include <stdexcept>

static void* unwrapOpenMMSwigPointer(PyObject* object, const char* typeName) {
    PyObject* thisObject = PyObject_GetAttrString(object, "this");
    if (thisObject == NULL)
        throw std::runtime_error(std::string("Expected an OpenMM Python object for ") + typeName);
    SwigPyObject* swigObject = SWIG_Python_GetSwigThis(thisObject);
    void* pointer = (swigObject == NULL ? NULL : swigObject->ptr);
    if (pointer == NULL) {
        PyObject* pointerInt = PyNumber_Long(thisObject);
        if (pointerInt != NULL) {
            pointer = PyLong_AsVoidPtr(pointerInt);
            Py_DECREF(pointerInt);
        }
        PyErr_Clear();
    }
    Py_DECREF(thisObject);
    if (pointer == NULL)
        throw std::runtime_error(std::string("Could not unwrap OpenMM Python object for ") + typeName);
    return pointer;
}
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
    const std::string& getForceMlir() const;
    const std::string& getEnergyMlir() const;
    const std::string& getEnergyAndForcesMlir() const;
    const std::string& getCompileOptions() const;
    std::string getCompileOptionsBase64() const;
    void setUsesPeriodicBoundaryConditions(bool periodic);
    bool usesPeriodicBoundaryConditions() const;
    void setOutputsForces(bool);
    bool getOutputsForces() const;
    void setPjrtPluginPath(const std::string& path);
    const std::string& getPjrtPluginPath() const;

    %extend {
        void setForceGroup(int group) {
            self->OpenMM::Force::setForceGroup(group);
        }

        int getForceGroup() const {
            return self->OpenMM::Force::getForceGroup();
        }

        int addToSystem(PyObject* system) {
            OpenMM::System* systemPointer = reinterpret_cast<OpenMM::System*>(
                    unwrapOpenMMSwigPointer(system, "OpenMM::System"));
            return systemPointer->addForce(self);
        }

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

_JaxForce_addToSystem = JaxForce.addToSystem

def _jax_force_add_to_system(self, system):
    index = _JaxForce_addToSystem(self, system)
    self.thisown = False
    return index

JaxForce.addToSystem = _jax_force_add_to_system

%}
