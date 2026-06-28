%module openmmjax

%include "factory.i"
%import(module="openmm.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>
%include <std_map.i>

%{
#include "internal/DLPackTypes.h"
#include "JaxForce.h"
#include "PythonJaxForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/serialization/SerializationProxy.h"
#include "openmm/serialization/SerializationNode.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace JaxPlugin {
class PythonJaxForce;
PythonJaxForce* _createPythonJaxForce(PyObject* energyComputation,
        PyObject* forcesComputation, PyObject* energyAndForcesComputation,
        const std::map<std::string, double>& globalParameters);
void registerPythonJaxForceProxy();
}

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

static int addForceToOpenMMSystem(OpenMM::Force* force, PyObject* system) {
    OpenMM::System* systemPointer = reinterpret_cast<OpenMM::System*>(
            unwrapOpenMMSwigPointer(system, "OpenMM::System"));
    return systemPointer->addForce(force);
}
%}

namespace std {
    %template(StringDoubleMap) map<string, double>;
}

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
        int addToSystem(PyObject* system) {
            return addForceToOpenMMSystem(self, system);
        }

        static JaxPlugin::JaxForce& cast(OpenMM::Force& force) {
            return dynamic_cast<JaxPlugin::JaxForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<JaxPlugin::JaxForce*>(&force) != NULL);
        }

    }
};

%newobject PythonJaxForce::PythonJaxForce;
class PythonJaxForce : public OpenMM::Force {
public:
    ~PythonJaxForce();
    const std::map<std::string, double>& getGlobalParameters() const;
    void setUsesPeriodicBoundaryConditions(bool periodic);
    bool usesPeriodicBoundaryConditions() const;

    %extend {
        static JaxPlugin::PythonJaxForce& cast(OpenMM::Force& force) {
            return dynamic_cast<JaxPlugin::PythonJaxForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<JaxPlugin::PythonJaxForce*>(&force) != NULL);
        }

        int addToSystem(PyObject* system) {
            return addForceToOpenMMSystem(self, system);
        }
    }
};

}

%{

namespace JaxPlugin {

struct OpenMmDlpackManager {
    DLManagedTensor managed;
    int64_t shape[2];
};

static void openmmDlpackDeleter(DLManagedTensor* tensor) {
    delete reinterpret_cast<OpenMmDlpackManager*>(tensor->manager_ctx);
}

static void openmmDlpackCapsuleDestructor(PyObject* capsule) {
    if (PyCapsule_IsValid(capsule, "dltensor")) {
        DLManagedTensor* tensor = reinterpret_cast<DLManagedTensor*>(
                PyCapsule_GetPointer(capsule, "dltensor"));
        if (tensor != NULL && tensor->deleter != NULL)
            tensor->deleter(tensor);
    }
}

static PyObject* createDlpackCapsule(uintptr_t pointer, int64_t rows, int64_t cols,
        int deviceIndex, bool useDoublePrecision) {
    OpenMmDlpackManager* manager = new OpenMmDlpackManager();
    manager->shape[0] = rows;
    manager->shape[1] = cols;
    manager->managed.dl_tensor.data = reinterpret_cast<void*>(pointer);
    manager->managed.dl_tensor.device = {kDLCUDA, deviceIndex};
    manager->managed.dl_tensor.ndim = 2;
    manager->managed.dl_tensor.dtype = {
        static_cast<uint8_t>(kDLFloat),
        static_cast<uint8_t>(useDoublePrecision ? 64 : 32),
        1
    };
    manager->managed.dl_tensor.shape = manager->shape;
    manager->managed.dl_tensor.strides = NULL;
    manager->managed.dl_tensor.byte_offset = 0;
    manager->managed.manager_ctx = manager;
    manager->managed.deleter = openmmDlpackDeleter;
    PyObject* capsule = PyCapsule_New(&manager->managed, "dltensor",
            openmmDlpackCapsuleDestructor);
    if (capsule == NULL) {
        delete manager;
        return NULL;
    }
    return capsule;
}

typedef struct {
    PyObject_HEAD
    uintptr_t pointer;
    int64_t rows;
    int64_t cols;
    int deviceIndex;
    int useDoublePrecision;
} CudaDlpackViewObject;

static PyObject* CudaDlpackView_dlpack_device(PyObject* self, PyObject*) {
    CudaDlpackViewObject* view = reinterpret_cast<CudaDlpackViewObject*>(self);
    return Py_BuildValue("(ii)", static_cast<int>(kDLCUDA), view->deviceIndex);
}

static PyObject* CudaDlpackView_dlpack(PyObject* self, PyObject*, PyObject*) {
    CudaDlpackViewObject* view = reinterpret_cast<CudaDlpackViewObject*>(self);
    return createDlpackCapsule(view->pointer, view->rows, view->cols,
            view->deviceIndex, view->useDoublePrecision != 0);
}

static void CudaDlpackView_dealloc(PyObject* self) {
    PyObject_Del(self);
}

static PyMethodDef CudaDlpackView_methods[] = {
    {"__dlpack_device__", reinterpret_cast<PyCFunction>(CudaDlpackView_dlpack_device), METH_NOARGS, NULL},
    {"__dlpack__", reinterpret_cast<PyCFunction>(CudaDlpackView_dlpack), METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject CudaDlpackViewType = {
    PyVarObject_HEAD_INIT(NULL, 0)
};

static void ensureCudaDlpackViewTypeReady() {
    if (CudaDlpackViewType.tp_name == NULL) {
        CudaDlpackViewType.tp_name = "_openmmjax.CudaDlpackView";
        CudaDlpackViewType.tp_basicsize = sizeof(CudaDlpackViewObject);
        CudaDlpackViewType.tp_dealloc = CudaDlpackView_dealloc;
        CudaDlpackViewType.tp_flags = Py_TPFLAGS_DEFAULT;
        CudaDlpackViewType.tp_methods = CudaDlpackView_methods;
        CudaDlpackViewType.tp_new = PyType_GenericNew;
    }
    if ((CudaDlpackViewType.tp_flags & Py_TPFLAGS_READY) == 0 &&
            PyType_Ready(&CudaDlpackViewType) < 0)
        throw OpenMM::OpenMMException("PythonJaxForce: failed to initialize DLPack view type");
}

static PyObject* createCudaDlpackView(uintptr_t pointer, int64_t rows, int64_t cols,
        int deviceIndex, bool useDoublePrecision) {
    ensureCudaDlpackViewTypeReady();
    CudaDlpackViewObject* view = PyObject_New(CudaDlpackViewObject, &CudaDlpackViewType);
    if (view == NULL)
        return NULL;
    view->pointer = pointer;
    view->rows = rows;
    view->cols = cols;
    view->deviceIndex = deviceIndex;
    view->useDoublePrecision = useDoublePrecision ? 1 : 0;
    return reinterpret_cast<PyObject*>(view);
}

static std::string currentPythonExceptionMessage() {
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 12
    PyObject *type = NULL;
    PyObject *exception = NULL;
    PyObject *traceback = NULL;
    PyErr_Fetch(&type, &exception, &traceback);
#else
    PyObject* exception = PyErr_GetRaisedException();
#endif
    if (exception == NULL)
        return "unknown Python exception";
    PyObject* message = PyObject_Str(exception);
    std::string text = "unknown Python exception";
    if (message != NULL) {
        const char* chars = PyUnicode_AsUTF8(message);
        if (chars != NULL)
            text = chars;
    }
    Py_XDECREF(message);
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 12
    Py_XDECREF(type);
    Py_XDECREF(traceback);
#endif
    Py_XDECREF(exception);
    return text;
}

static PyObject* importJaxDlpackFunction() {
    PyObject* module = PyImport_ImportModule("jax.dlpack");
    if (module == NULL)
        return NULL;
    PyObject* function = PyObject_GetAttrString(module, "from_dlpack");
    Py_DECREF(module);
    return function;
}

static PyObject* jaxArrayFromDlpack(PyObject* fromDlpack, PyObject* view) {
    PyObject* args = PyTuple_Pack(1, view);
    if (args == NULL)
        return NULL;
    PyObject* kwargs = PyDict_New();
    if (kwargs == NULL) {
        Py_DECREF(args);
        return NULL;
    }
    if (PyDict_SetItemString(kwargs, "copy", Py_False) != 0) {
        Py_DECREF(args);
        Py_DECREF(kwargs);
        return NULL;
    }
    PyObject* array = PyObject_Call(fromDlpack, args, kwargs);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    return array;
}

static PyObject* parametersToDict(const std::map<std::string, double>& parameters) {
    PyObject* dict = PyDict_New();
    if (dict == NULL)
        return NULL;
    for (auto& item : parameters) {
        PyObject* value = PyFloat_FromDouble(item.second);
        if (value == NULL || PyDict_SetItemString(dict, item.first.c_str(), value) != 0) {
            Py_XDECREF(value);
            Py_DECREF(dict);
            return NULL;
        }
        Py_DECREF(value);
    }
    return dict;
}

struct PythonJaxForceOutputLifetime {
    PyObject* result = NULL;
    PyObject* forces = NULL;
    PyObject* capsule = NULL;
    ~PythonJaxForceOutputLifetime() {
        PyGILState_STATE gstate = PyGILState_Ensure();
        Py_XDECREF(capsule);
        Py_XDECREF(forces);
        Py_XDECREF(result);
        PyGILState_Release(gstate);
    }
};

static uintptr_t pointerFromForceArray(PyObject* forces, int64_t expectedRows,
        int deviceIndex, bool useDoublePrecision, PyObject*& capsuleOut) {
    PyObject* block = PyObject_CallMethod(forces, "block_until_ready", NULL);
    if (block == NULL)
        return 0;
    Py_DECREF(block);

    PyObject* capsule = PyObject_CallMethod(forces, "__dlpack__", NULL);
    if (capsule == NULL)
        return 0;
    DLManagedTensor* managed = reinterpret_cast<DLManagedTensor*>(
            PyCapsule_GetPointer(capsule, "dltensor"));
    if (managed == NULL) {
        Py_DECREF(capsule);
        return 0;
    }
    DLTensor& tensor = managed->dl_tensor;
    if (tensor.device.device_type != kDLCUDA || tensor.device.device_id != deviceIndex) {
        Py_DECREF(capsule);
        throw OpenMM::OpenMMException("PythonJaxForce: forces must be a JAX array on the OpenMM CUDA device");
    }
    int expectedBits = useDoublePrecision ? 64 : 32;
    if (tensor.dtype.code != kDLFloat || tensor.dtype.bits != expectedBits || tensor.dtype.lanes != 1) {
        Py_DECREF(capsule);
        throw OpenMM::OpenMMException("PythonJaxForce: force dtype does not match OpenMM CUDA precision");
    }
    if (tensor.ndim != 2 || tensor.shape == NULL ||
            tensor.shape[0] != expectedRows || tensor.shape[1] != 3) {
        Py_DECREF(capsule);
        throw OpenMM::OpenMMException("PythonJaxForce: forces must have shape (particles, 3)");
    }
    if (tensor.strides != NULL && (tensor.strides[0] != 3 || tensor.strides[1] != 1)) {
        Py_DECREF(capsule);
        throw OpenMM::OpenMMException("PythonJaxForce: forces must be contiguous row-major arrays");
    }
    capsuleOut = capsule;
    return reinterpret_cast<uintptr_t>(
            static_cast<char*>(tensor.data) + tensor.byte_offset);
}

class ComputationWrapper : public PythonJaxForceComputation {
public:
    ComputationWrapper(PyObject* computation, bool hasParameters) :
            computation(computation), hasParameters(hasParameters) {
        Py_INCREF(computation);
    }
    ~ComputationWrapper() {
        PyGILState_STATE gstate = PyGILState_Ensure();
        Py_XDECREF(computation);
        PyGILState_Release(gstate);
    }

    PythonJaxForceComputationResult compute(const PythonJaxForceComputationInputs& inputs,
            bool includeForces, bool includeEnergy) const override {
        PyGILState_STATE gstate = PyGILState_Ensure();
        PythonJaxForceComputationResult output;
        try {
            PyObject* fromDlpack = importJaxDlpackFunction();
            if (fromDlpack == NULL)
                throw OpenMM::OpenMMException(currentPythonExceptionMessage());
            PyObject* positionView = createCudaDlpackView(inputs.positions,
                    inputs.numParticles, 3, inputs.deviceIndex,
                    inputs.useDoublePrecision);
            if (positionView == NULL) {
                Py_DECREF(fromDlpack);
                throw OpenMM::OpenMMException(currentPythonExceptionMessage());
            }
            PyObject* positions = jaxArrayFromDlpack(fromDlpack, positionView);
            Py_DECREF(positionView);
            if (positions == NULL) {
                Py_DECREF(fromDlpack);
                throw OpenMM::OpenMMException(currentPythonExceptionMessage());
            }

            PyObject* boxVectors = NULL;
            if (inputs.usePeriodic) {
                PyObject* boxView = createCudaDlpackView(inputs.boxVectors, 3, 3,
                        inputs.deviceIndex, inputs.useDoublePrecision);
                if (boxView == NULL) {
                    Py_DECREF(positions);
                    Py_DECREF(fromDlpack);
                    throw OpenMM::OpenMMException(currentPythonExceptionMessage());
                }
                boxVectors = jaxArrayFromDlpack(fromDlpack, boxView);
                Py_DECREF(boxView);
                if (boxVectors == NULL) {
                    Py_DECREF(positions);
                    Py_DECREF(fromDlpack);
                    throw OpenMM::OpenMMException(currentPythonExceptionMessage());
                }
            }
            Py_DECREF(fromDlpack);

            PyObject* parameters = NULL;
            if (hasParameters) {
                parameters = parametersToDict(inputs.parameters);
                if (parameters == NULL) {
                    Py_DECREF(positions);
                    Py_XDECREF(boxVectors);
                    throw OpenMM::OpenMMException(currentPythonExceptionMessage());
                }
            }

            PyObject* result = NULL;
            if (inputs.usePeriodic && hasParameters)
                result = PyObject_CallFunctionObjArgs(computation, positions, boxVectors, parameters, NULL);
            else if (inputs.usePeriodic)
                result = PyObject_CallFunctionObjArgs(computation, positions, boxVectors, NULL);
            else if (hasParameters)
                result = PyObject_CallFunctionObjArgs(computation, positions, parameters, NULL);
            else
                result = PyObject_CallFunctionObjArgs(computation, positions, NULL);
            Py_DECREF(positions);
            Py_XDECREF(boxVectors);
            Py_XDECREF(parameters);
            if (result == NULL)
                throw OpenMM::OpenMMException(currentPythonExceptionMessage());
            PyObject* energy = result;
            PyObject* forces = result;
            if (includeEnergy && includeForces) {
                if (!PyTuple_Check(result) || PyTuple_Size(result) != 2) {
                    Py_DECREF(result);
                    throw OpenMM::OpenMMException("PythonJaxForce: expected energy_and_forces function to return two values");
                }
                energy = PyTuple_GetItem(result, 0);
                forces = PyTuple_GetItem(result, 1);
            }
            if (includeEnergy) {
                PyObject* energyFloat = PyNumber_Float(energy);
                if (energyFloat == NULL) {
                    Py_DECREF(result);
                    throw OpenMM::OpenMMException(currentPythonExceptionMessage());
                }
                output.energy = PyFloat_AsDouble(energyFloat);
                Py_DECREF(energyFloat);
                if (PyErr_Occurred()) {
                    Py_DECREF(result);
                    throw OpenMM::OpenMMException(currentPythonExceptionMessage());
                }
            }
            if (includeForces) {
                if (forces == Py_None) {
                    Py_DECREF(result);
                    throw OpenMM::OpenMMException("PythonJaxForce: callback returned None for forces");
                }
                Py_INCREF(forces);
                PyObject* capsule = NULL;
                uintptr_t pointer = pointerFromForceArray(forces, inputs.numParticles,
                        inputs.deviceIndex, inputs.useDoublePrecision, capsule);
                if (pointer == 0) {
                    Py_DECREF(forces);
                    Py_DECREF(result);
                    throw OpenMM::OpenMMException(currentPythonExceptionMessage());
                }
                auto lifetime = std::make_shared<PythonJaxForceOutputLifetime>();
                lifetime->result = result;
                lifetime->forces = forces;
                lifetime->capsule = capsule;
                output.forces = pointer;
                output.lifetime = lifetime;
            }
            else {
                Py_DECREF(result);
            }
            PyGILState_Release(gstate);
            return output;
        }
        catch (...) {
            PyGILState_Release(gstate);
            throw;
        }
    }

private:
    PyObject* computation;
    bool hasParameters;
};

static void picklePythonFunction(PyObject* pickle, PyObject* function,
        void (PythonJaxForce::*setter)(char*, int), PythonJaxForce* force) {
    PyObject* result = PyObject_CallMethod(pickle, "dumps", "O", function);
    if (result == NULL) {
        PyErr_Clear();
        return;
    }
    char* buffer;
    Py_ssize_t len;
    if (PyBytes_AsStringAndSize(result, &buffer, &len) == 0)
        (force->*setter)(buffer, static_cast<int>(len));
    Py_DECREF(result);
}

PythonJaxForce* _createPythonJaxForce(PyObject* energyComputation,
        PyObject* forcesComputation, PyObject* energyAndForcesComputation,
        const std::map<std::string, double>& globalParameters) {
    PythonJaxForce* force = new PythonJaxForce(
            new ComputationWrapper(energyComputation, !globalParameters.empty()),
            new ComputationWrapper(forcesComputation, !globalParameters.empty()),
            new ComputationWrapper(energyAndForcesComputation, !globalParameters.empty()),
            globalParameters);
    PyObject* pickle = PyImport_ImportModule("pickle");
    if (pickle != NULL) {
        picklePythonFunction(pickle, energyComputation,
                &PythonJaxForce::setPickledEnergyFunction, force);
        picklePythonFunction(pickle, forcesComputation,
                &PythonJaxForce::setPickledForcesFunction, force);
        picklePythonFunction(pickle, energyAndForcesComputation,
                &PythonJaxForce::setPickledEnergyAndForcesFunction, force);
        Py_DECREF(pickle);
    }
    else {
        PyErr_Clear();
    }
    return force;
}

class PythonJaxForceProxy : public OpenMM::SerializationProxy {
public:
    PythonJaxForceProxy() : OpenMM::SerializationProxy("PythonJaxForce") {
    }

    static std::string hexEncode(const std::vector<char>& input) {
        std::stringstream ss;
        ss << std::hex << std::setfill('0');
        for (unsigned char i : input)
            ss << std::setw(2) << static_cast<int>(i);
        return ss.str();
    }

    static std::vector<char> hexDecode(const std::string& input) {
        std::vector<char> result;
        result.reserve(input.size()/2);
        for (size_t i = 0; i < input.size(); i += 2) {
            std::istringstream iss(input.substr(i, 2));
            int value;
            iss >> std::hex >> value;
            result.push_back(static_cast<char>(value));
        }
        return result;
    }

    void serialize(const void* object, OpenMM::SerializationNode& node) const override {
        const PythonJaxForce& force = *reinterpret_cast<const PythonJaxForce*>(object);
        if (force.getPickledEnergyFunction().empty() ||
                force.getPickledForcesFunction().empty() ||
                force.getPickledEnergyAndForcesFunction().empty())
            throw OpenMM::OpenMMException("PythonJaxForceProxy: functions could not be pickled");
        node.setIntProperty("version", 1);
        node.setStringProperty("energyFunction", hexEncode(force.getPickledEnergyFunction()));
        node.setStringProperty("forcesFunction", hexEncode(force.getPickledForcesFunction()));
        node.setStringProperty("energyAndForcesFunction", hexEncode(force.getPickledEnergyAndForcesFunction()));
        node.setIntProperty("forceGroup", force.getForceGroup());
        node.setStringProperty("name", force.getName());
        node.setBoolProperty("usesPeriodic", force.usesPeriodicBoundaryConditions());
        OpenMM::SerializationNode& params = node.createChildNode("GlobalParameters");
        for (auto& param : force.getGlobalParameters()) {
            OpenMM::SerializationNode& child = params.createChildNode("Parameter");
            child.setStringProperty("name", param.first);
            child.setDoubleProperty("default", param.second);
        }
    }

    void* deserialize(const OpenMM::SerializationNode& node) const override {
        std::vector<char> pickledEnergy = hexDecode(node.getStringProperty("energyFunction"));
        std::vector<char> pickledForces = hexDecode(node.getStringProperty("forcesFunction"));
        std::vector<char> pickledEnergyAndForces = hexDecode(node.getStringProperty("energyAndForcesFunction"));
        PyGILState_STATE gstate = PyGILState_Ensure();
        PyObject* pickle = PyImport_ImportModule("pickle");
        if (pickle == NULL) {
            PyGILState_Release(gstate);
            throw OpenMM::OpenMMException("PythonJaxForceProxy: could not import pickle");
        }
        PyObject* energyBytes = PyBytes_FromStringAndSize(pickledEnergy.data(), pickledEnergy.size());
        PyObject* forcesBytes = PyBytes_FromStringAndSize(pickledForces.data(), pickledForces.size());
        PyObject* energyAndForcesBytes = PyBytes_FromStringAndSize(
                pickledEnergyAndForces.data(), pickledEnergyAndForces.size());
        if (energyBytes == NULL || forcesBytes == NULL || energyAndForcesBytes == NULL) {
            std::string message = currentPythonExceptionMessage();
            Py_XDECREF(energyBytes);
            Py_XDECREF(forcesBytes);
            Py_XDECREF(energyAndForcesBytes);
            Py_DECREF(pickle);
            PyGILState_Release(gstate);
            throw OpenMM::OpenMMException("PythonJaxForceProxy: could not create pickled function bytes: " + message);
        }
        PyObject* energyFunction = PyObject_CallMethod(pickle, "loads", "O", energyBytes);
        PyObject* forcesFunction = PyObject_CallMethod(pickle, "loads", "O", forcesBytes);
        PyObject* energyAndForcesFunction = PyObject_CallMethod(
                pickle, "loads", "O", energyAndForcesBytes);
        Py_DECREF(energyBytes);
        Py_DECREF(forcesBytes);
        Py_DECREF(energyAndForcesBytes);
        Py_DECREF(pickle);
        if (energyFunction == NULL || forcesFunction == NULL || energyAndForcesFunction == NULL) {
            std::string message = currentPythonExceptionMessage();
            Py_XDECREF(energyFunction);
            Py_XDECREF(forcesFunction);
            Py_XDECREF(energyAndForcesFunction);
            PyGILState_Release(gstate);
            throw OpenMM::OpenMMException("PythonJaxForceProxy: could not unpickle functions: " + message);
        }
        std::map<std::string, double> params;
        const OpenMM::SerializationNode& paramNode = node.getChildNode("GlobalParameters");
        for (auto& child : paramNode.getChildren())
            params[child.getStringProperty("name")] = child.getDoubleProperty("default");
        PythonJaxForce* force = _createPythonJaxForce(
                energyFunction, forcesFunction, energyAndForcesFunction, params);
        Py_DECREF(energyFunction);
        Py_DECREF(forcesFunction);
        Py_DECREF(energyAndForcesFunction);
        PyGILState_Release(gstate);
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setName(node.getStringProperty("name", force->getName()));
        force->setUsesPeriodicBoundaryConditions(node.getBoolProperty("usesPeriodic", false));
        return force;
    }
};

void registerPythonJaxForceProxy() {
    OpenMM::SerializationProxy::registerProxy(typeid(PythonJaxForce), new PythonJaxForceProxy());
}

}

%}

%extend JaxPlugin::PythonJaxForce {
    PythonJaxForce(PyObject* energyComputation, PyObject* forcesComputation,
            PyObject* energyAndForcesComputation,
            const std::map<std::string, double>& globalParameters=std::map<std::string, double>()) {
        return JaxPlugin::_createPythonJaxForce(
                energyComputation, forcesComputation, energyAndForcesComputation,
                globalParameters);
    }
}

%init %{
    JaxPlugin::registerPythonJaxForceProxy();
%}

%pythoncode %{

def _disown_after_add_to_system(cls):
    original = cls.addToSystem

    def add_to_system(self, system):
        index = original(self, system)
        self.thisown = False
        return index

    cls.addToSystem = add_to_system

_disown_after_add_to_system(JaxForce)
_disown_after_add_to_system(PythonJaxForce)

%}
