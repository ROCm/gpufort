
IF(NOT CMAKE_Fortran_COMPILER_ID MATCHES "Cray")
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fno-underscoring")
ENDIF(NOT CMAKE_Fortran_COMPILER_ID MATCHES "Cray")
file(GLOB HIPFORT_SRC_HIP     "${CMAKE_CURRENT_SOURCE_DIR}/hipfort/*.f*")
#file(GLOB HIPFORT_SRC_amdgcn  "${CMAKE_CURRENT_SOURCE_DIR}/modules-amdgcn/*.f*")
#file(GLOB HIPFORT_SRC_nvptx   "${CMAKE_CURRENT_SOURCE_DIR}/modules-nvptx/*.f*")
file(GLOB HIPFORT_SRC_CONTRIB "${CMAKE_CURRENT_SOURCE_DIR}/modules-contrib/*.f*")

set(HIPFORT_ARCH "amdgcn")
    # amdgcn
    set(HIPFORT_LIB      "hipfort-${HIPFORT_ARCH}")
    set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/include/hipfort/${HIPFORT_ARCH})
    ADD_LIBRARY(${HIPFORT_LIB} STATIC 
        ${HIPFORT_SRC_HIP} 
        #${HIPFORT_SRC_amdgcn} 
        ${HIPFORT_SRC_CONTRIB} 
        )
    IF(CMAKE_Fortran_COMPILER_SUPPORTS_F08)
    	target_compile_definitions(${HIPFORT_LIB} PRIVATE USE_FPOINTER_INTERFACES)
    ENDIF(CMAKE_Fortran_COMPILER_SUPPORTS_F08)
    target_compile_definitions(${HIPFORT_LIB} PRIVATE _HIPFORT_ARCH='${HIPFORT_ARCH}')
    install(TARGETS ${HIPFORT_LIB} 
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib)

set(HIPFORT_ARCH "nvptx")
    # nvptx
    set(HIPFORT_LIB     "hipfort-${HIPFORT_ARCH}")
    set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/include/hipfort/${HIPFORT_ARCH})
    ADD_LIBRARY(${HIPFORT_LIB} STATIC 
         ${HIPFORT_SRC_HIP} 
         #${HIPFORT_SRC_nvptx} 
         ${HIPFORT_SRC_CONTRIB} 
         )
    IF(CMAKE_Fortran_COMPILER_SUPPORTS_F08)
    	target_compile_definitions(${HIPFORT_LIB} PRIVATE USE_FPOINTER_INTERFACES)
    ENDIF(CMAKE_Fortran_COMPILER_SUPPORTS_F08)
    target_compile_definitions(${HIPFORT_LIB} PRIVATE USE_CUDA_NAMES)
    target_compile_definitions(${HIPFORT_LIB} PRIVATE _HIPFORT_ARCH='${HIPFORT_ARCH}')
    install(TARGETS ${HIPFORT_LIB} 
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib)
   
install(DIRECTORY ${CMAKE_BINARY_DIR}/include/hipfort DESTINATION include)

#   target_link_libraries(${HIPFORT_LIB} PUBLIC 
#   /usr/local/cuda/targets/x86_64-linux/lib/libcudart_static.a)
