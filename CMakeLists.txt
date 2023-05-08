cmake_minimum_required(VERSION 3.0.2)

project(gpufort)

if (NOT DEFINED ROCM_PATH )
     set ( ROCM_PATH "/opt/rocm"  CACHE STRING "Default ROCM installation directory." )
endif ()



install(DIRECTORY bin 
        DESTINATION gpufort
        PATTERN "bin/*"
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                    GROUP_EXECUTE GROUP_READ
                    WORLD_READ WORLD_EXECUTE)

install(DIRECTORY config 
        DESTINATION gpufort
        PATTERN "config/*"
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                    GROUP_EXECUTE GROUP_READ
                    WORLD_READ WORLD_EXECUTE)


install(DIRECTORY examples/openacc/
        DESTINATION gpufort/examples/openacc
        USE_SOURCE_PERMISSIONS
        PATTERN "examples/openacc/*/*.f90"
        PERMISSIONS OWNER_WRITE OWNER_READ 
                    GROUP_READ
                    WORLD_READ)




#### ####### hipfort section ########  
install(DIRECTORY hipfort_util/bin/
        DESTINATION gpufort/hipfort_util/bin
        USE_SOURCE_PERMISSIONS
        PATTERN "hipfort_util/bin/*"
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                    GROUP_EXECUTE GROUP_READ
                    WORLD_EXECUTE WORLD_READ)

install(DIRECTORY hipfort_util/include/hipfort/amdgcn/
        DESTINATION gpufort/hipfort_util/include/hipfort/amdgcn
        USE_SOURCE_PERMISSIONS
        PATTERN "hipfort_util/include/hipfort/amdgcn/*"
        PERMISSIONS OWNER_WRITE OWNER_READ
                    GROUP_READ
                    WORLD_READ)
            
install(DIRECTORY hipfort_util/include/hipfort/nvptx/
        DESTINATION gpufort/hipfort_util/include/hipfort/nvptx
        USE_SOURCE_PERMISSIONS
        PATTERN "hipfort_util/include/hipfort/nvptx/*"
        PERMISSIONS OWNER_WRITE OWNER_READ
                GROUP_READ
                WORLD_READ)

install(DIRECTORY hipfort_util/lib/
        DESTINATION gpufort/hipfort_util/lib
        USE_SOURCE_PERMISSIONS
        PATTERN "hipfort_util/lib/*"
        PERMISSIONS OWNER_WRITE OWNER_READ
                    GROUP_READ
                    WORLD_READ)


install(DIRECTORY hipfort_util/libexec/
        DESTINATION gpufort/hipfort_util/libexec
        USE_SOURCE_PERMISSIONS
        PATTERN "hipfort_util/libexec/hipfort/*"
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                    GROUP_EXECUTE GROUP_READ
                    WORLD_EXECUTE WORLD_READ)


install(DIRECTORY hipfort_util/share/hipfort/
        DESTINATION gpufort/hipfort_util/share/hipfort
        USE_SOURCE_PERMISSIONS
        PATTERN "hipfort_util/share/hipfort/*"
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                    GROUP_EXECUTE GROUP_READ
                    WORLD_EXECUTE WORLD_READ)

######################################


 
 install(DIRECTORY include
        DESTINATION gpufort
        USE_SOURCE_PERMISSIONS)


 install(DIRECTORY lib
        DESTINATION gpufort
        PATTERN "lib/*"
        PERMISSIONS OWNER_WRITE OWNER_READ
                    GROUP_READ
                    WORLD_READ)



### python section ###
install(DIRECTORY python/
DESTINATION gpufort/python
USE_SOURCE_PERMISSIONS
FILES_MATCHING
PATTERN "*.py")

install(DIRECTORY python/
DESTINATION gpufort/python
USE_SOURCE_PERMISSIONS
FILES_MATCHING
PATTERN "*.py.in"
PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                    GROUP_EXECUTE GROUP_READ
                    WORLD_EXECUTE WORLD_READ)

install(DIRECTORY python/
DESTINATION gpufort/python
USE_SOURCE_PERMISSIONS
FILES_MATCHING
PATTERN "*.inp"
PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                    GROUP_EXECUTE GROUP_READ
                    WORLD_EXECUTE WORLD_READ)
                
install(DIRECTORY python/
        DESTINATION gpufort/python
        USE_SOURCE_PERMISSIONS
        FILES_MATCHING
        PATTERN "*.cpp"
        PERMISSIONS OWNER_WRITE OWNER_READ
                    GROUP_READ
                    WORLD_READ)

install(DIRECTORY python/
        DESTINATION gpufort/python
        USE_SOURCE_PERMISSIONS
        FILES_MATCHING
        PATTERN "*.f03"
        PERMISSIONS OWNER_WRITE OWNER_READ
                     GROUP_READ
                     WORLD_READ)


install(DIRECTORY python/
        DESTINATION gpufort/python
        USE_SOURCE_PERMISSIONS
        FILES_MATCHING
        PATTERN "*.h"
        PERMISSIONS OWNER_WRITE OWNER_READ
                   GROUP_READ
                   WORLD_READ)



install(FILES python/requirements.txt
        DESTINATION gpufort/python
        PERMISSIONS OWNER_WRITE OWNER_READ
                    GROUP_READ
                    WORLD_READ)

########################################

 install(DIRECTORY requirements_whl
        DESTINATION gpufort
        PATTERN "requirements_whl/*"
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                    GROUP_EXECUTE GROUP_READ
                    WORLD_EXECUTE WORLD_READ)

install(FILES env_gpufort.sh Makefile 
       DESTINATION gpufort
       PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                    GROUP_EXECUTE GROUP_READ
                    WORLD_READ WORLD_EXECUTE)





#############################
# Packaging steps
#############################
set(CPACK_PACKAGE_NAME "gpufort")
set(CPACK_PACKAGE_FILE_NAME gpufort-1.0.0)


set(CPACK_GENERATOR "RPM")
set(CPACK_BINARY_RPM "ON")
set(CPACK_RPM_SPEC_MORE_DEFINE "%define __python python3")


message("================= Build GPUFort =================" )
message("---------------ROCM_PATH: " ${ROCM_PATH})
message("----CPACK_INSTALL_PREFIX: " ${CPACK_INSTALL_PREFIX})
message("----CPACK_PACKAGING_INSTALL_PREFIX: " ${CPACK_PACKAGING_INSTALL_PREFIX})

include(CPack)






