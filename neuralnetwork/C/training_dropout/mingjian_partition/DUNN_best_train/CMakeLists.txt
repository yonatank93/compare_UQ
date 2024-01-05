#
# Contributors:
#    KLIFF (https://kliff.readthedocs.io)
#

cmake_minimum_required(VERSION 3.4)

list(APPEND CMAKE_PREFIX_PATH $ENV{KIM_API_CMAKE_PREFIX_DIR})
find_package(KIM-API 2.0 REQUIRED CONFIG)
if(NOT TARGET kim-api)
  enable_testing()
  project("${KIM_API_PROJECT_NAME}" VERSION "${KIM_API_VERSION}"
    LANGUAGES CXX C Fortran)
endif()

add_kim_api_model_library(
  NAME            "DUNN_best_train"
  DRIVER_NAME     "DUNN__MD_292677547454_000"
  PARAMETER_FILES "descriptor.params" "NN.params" "dropout_binary.params"
  )
