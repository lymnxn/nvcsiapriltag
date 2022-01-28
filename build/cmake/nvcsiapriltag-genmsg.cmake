# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "nvcsiapriltag: 2 messages, 0 services")

set(MSG_I_FLAGS "-Invcsiapriltag:/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg;-Istd_msgs:/opt/ros/melodic/share/std_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/melodic/share/geometry_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(nvcsiapriltag_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetection.msg" NAME_WE)
add_custom_target(_nvcsiapriltag_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "nvcsiapriltag" "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetection.msg" ""
)

get_filename_component(_filename "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetectionArray.msg" NAME_WE)
add_custom_target(_nvcsiapriltag_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "nvcsiapriltag" "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetectionArray.msg" "nvcsiapriltag/AprilTagDetection:std_msgs/Header"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(nvcsiapriltag
  "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetection.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/nvcsiapriltag
)
_generate_msg_cpp(nvcsiapriltag
  "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetectionArray.msg"
  "${MSG_I_FLAGS}"
  "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetection.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/nvcsiapriltag
)

### Generating Services

### Generating Module File
_generate_module_cpp(nvcsiapriltag
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/nvcsiapriltag
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(nvcsiapriltag_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(nvcsiapriltag_generate_messages nvcsiapriltag_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetection.msg" NAME_WE)
add_dependencies(nvcsiapriltag_generate_messages_cpp _nvcsiapriltag_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetectionArray.msg" NAME_WE)
add_dependencies(nvcsiapriltag_generate_messages_cpp _nvcsiapriltag_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(nvcsiapriltag_gencpp)
add_dependencies(nvcsiapriltag_gencpp nvcsiapriltag_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS nvcsiapriltag_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(nvcsiapriltag
  "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetection.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/nvcsiapriltag
)
_generate_msg_eus(nvcsiapriltag
  "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetectionArray.msg"
  "${MSG_I_FLAGS}"
  "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetection.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/nvcsiapriltag
)

### Generating Services

### Generating Module File
_generate_module_eus(nvcsiapriltag
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/nvcsiapriltag
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(nvcsiapriltag_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(nvcsiapriltag_generate_messages nvcsiapriltag_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetection.msg" NAME_WE)
add_dependencies(nvcsiapriltag_generate_messages_eus _nvcsiapriltag_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetectionArray.msg" NAME_WE)
add_dependencies(nvcsiapriltag_generate_messages_eus _nvcsiapriltag_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(nvcsiapriltag_geneus)
add_dependencies(nvcsiapriltag_geneus nvcsiapriltag_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS nvcsiapriltag_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(nvcsiapriltag
  "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetection.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/nvcsiapriltag
)
_generate_msg_lisp(nvcsiapriltag
  "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetectionArray.msg"
  "${MSG_I_FLAGS}"
  "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetection.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/nvcsiapriltag
)

### Generating Services

### Generating Module File
_generate_module_lisp(nvcsiapriltag
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/nvcsiapriltag
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(nvcsiapriltag_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(nvcsiapriltag_generate_messages nvcsiapriltag_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetection.msg" NAME_WE)
add_dependencies(nvcsiapriltag_generate_messages_lisp _nvcsiapriltag_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetectionArray.msg" NAME_WE)
add_dependencies(nvcsiapriltag_generate_messages_lisp _nvcsiapriltag_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(nvcsiapriltag_genlisp)
add_dependencies(nvcsiapriltag_genlisp nvcsiapriltag_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS nvcsiapriltag_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(nvcsiapriltag
  "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetection.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/nvcsiapriltag
)
_generate_msg_nodejs(nvcsiapriltag
  "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetectionArray.msg"
  "${MSG_I_FLAGS}"
  "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetection.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/nvcsiapriltag
)

### Generating Services

### Generating Module File
_generate_module_nodejs(nvcsiapriltag
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/nvcsiapriltag
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(nvcsiapriltag_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(nvcsiapriltag_generate_messages nvcsiapriltag_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetection.msg" NAME_WE)
add_dependencies(nvcsiapriltag_generate_messages_nodejs _nvcsiapriltag_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetectionArray.msg" NAME_WE)
add_dependencies(nvcsiapriltag_generate_messages_nodejs _nvcsiapriltag_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(nvcsiapriltag_gennodejs)
add_dependencies(nvcsiapriltag_gennodejs nvcsiapriltag_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS nvcsiapriltag_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(nvcsiapriltag
  "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetection.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/nvcsiapriltag
)
_generate_msg_py(nvcsiapriltag
  "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetectionArray.msg"
  "${MSG_I_FLAGS}"
  "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetection.msg;/opt/ros/melodic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/nvcsiapriltag
)

### Generating Services

### Generating Module File
_generate_module_py(nvcsiapriltag
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/nvcsiapriltag
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(nvcsiapriltag_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(nvcsiapriltag_generate_messages nvcsiapriltag_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetection.msg" NAME_WE)
add_dependencies(nvcsiapriltag_generate_messages_py _nvcsiapriltag_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/helabnx/code/vicon_ws/src/nvcsiapriltag/msg/AprilTagDetectionArray.msg" NAME_WE)
add_dependencies(nvcsiapriltag_generate_messages_py _nvcsiapriltag_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(nvcsiapriltag_genpy)
add_dependencies(nvcsiapriltag_genpy nvcsiapriltag_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS nvcsiapriltag_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/nvcsiapriltag)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/nvcsiapriltag
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(nvcsiapriltag_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(nvcsiapriltag_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/nvcsiapriltag)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/nvcsiapriltag
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(nvcsiapriltag_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(nvcsiapriltag_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/nvcsiapriltag)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/nvcsiapriltag
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(nvcsiapriltag_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(nvcsiapriltag_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/nvcsiapriltag)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/nvcsiapriltag
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(nvcsiapriltag_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(nvcsiapriltag_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/nvcsiapriltag)
  install(CODE "execute_process(COMMAND \"/usr/bin/python2\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/nvcsiapriltag\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/nvcsiapriltag
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(nvcsiapriltag_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(nvcsiapriltag_generate_messages_py geometry_msgs_generate_messages_py)
endif()
