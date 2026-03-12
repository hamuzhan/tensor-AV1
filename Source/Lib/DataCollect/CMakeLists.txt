#
# Data Collection Pipeline for ML Training
# Only built when -DENABLE_DATA_COLLECTION=ON
#

if(ENABLE_DATA_COLLECTION)
    find_package(HDF5 REQUIRED COMPONENTS C)

    add_library(DATA_COLLECT OBJECT
        data_collector.c
        data_collector.h
        hdf5_writer.c
        hdf5_writer.h
        partition_tree_flatten.c
        partition_tree_flatten.h
        me_data_extract.c
        me_data_extract.h)

    target_include_directories(DATA_COLLECT PRIVATE
        ${PROJECT_SOURCE_DIR}/Source/Lib/Codec/
        ${PROJECT_SOURCE_DIR}/Source/Lib/Globals/
        ${PROJECT_SOURCE_DIR}/Source/API/
        ${HDF5_INCLUDE_DIRS})

    target_compile_definitions(DATA_COLLECT PRIVATE ENABLE_DATA_COLLECTION=1)
endif()
