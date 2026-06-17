SET(WRAP_CPP "${CMAKE_SOURCE_DIR}/python/JaxPluginWrapper.cpp")
SET(WRAP_PY "${CMAKE_SOURCE_DIR}/python/openmmjax.py")

ADD_CUSTOM_COMMAND(
    OUTPUT "${WRAP_CPP}" "${WRAP_PY}"
    COMMAND "${SWIG_EXECUTABLE}"
        -python -c++
        -outdir "${CMAKE_SOURCE_DIR}/python"
        -o "${WRAP_CPP}"
        "-I${OPENMM_DIR}/include"
        "${CMAKE_SOURCE_DIR}/python/openmmjax.i"
    DEPENDS
        "${CMAKE_SOURCE_DIR}/python/openmmjax.i"
        "${CMAKE_SOURCE_DIR}/openmmapi/include/JaxForce.h"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/python"
)

ADD_CUSTOM_TARGET(PythonWrapper DEPENDS "${WRAP_CPP}" "${WRAP_PY}")
ADD_CUSTOM_TARGET(PythonInstall DEPENDS PythonWrapper
    COMMAND "${CMAKE_COMMAND}" -E env "OPENMM_DIR=${OPENMM_DIR}"
        "PIP_REQUIRE_VIRTUALENV=false"
        "${PYTHON_EXECUTABLE}" -m pip install -e "${CMAKE_SOURCE_DIR}" --no-build-isolation --no-deps
)
