project(Calibration_Camera_Velodyne)
cmake_minimum_required(VERSION 2.8)

set(OpenCV_DIR /usr/local/share/OpenCV)
set(VTK_DIR /usr/local/lib/cmake/vtk-7.1/)
set(Qt5_DIR /opt/Qt5.4.2/5.4/gcc_64/lib/cmake/Qt5)
set(Qt5core_DIR /opt/Qt5.4.2/5.4/gcc_64/lib/cmake/Qt5Core)
set(PCL_DIR /usr/local/share/pcl-1.8)

#include pcl library
find_package(PCL 1.8  REQUIRED COMPONENTS common io visualization octree features segmentation)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


find_package(Boost REQUIRED COMPONENTS
thread
filesystem
system
)

find_package(OpenCV REQUIRED)

#include vtk library
find_package(VTK COMPONENTS
  vtkCommonCore
  vtkFiltersCore
  vtkInfovisCore
  vtkInteractionStyle
  vtkViewsQt
  vtkIOCore
  vtkIOGeometry
  vtkRenderingLOD
)
#include(${VTK_USE_FILE})
#if("${VTK_QT_VERSION}" STREQUAL "")
#  message(FATAL_ERROR "VTK was not built with Qt")
#endif()


include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set your files and resources here
set( Srcs Calibration.cpp main.cpp pointgreycamera.cpp calibrateExtrinsic.cpp)
set( Hdrs Calibration.h pointgreycamera.h simplevlpviewer.hpp calibrateExtrinsic.h)
set( MOC_Hdrs Calibration.h)
set( UIs Calibration.ui)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
# We have ui files, this will bring in the macro: qt5_wrap_ui
find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED )

qt5_wrap_ui(UI_Srcs ${UIs})
qt5_add_resources(QRC_Srcs ${QRCs})
source_group("Resources" FILES
  ${UIs}
  ${QRCs}
  ${EXE_ICON} # Not present
)
source_group("Generated" FILES
  ${UI_Srcs}
  ${MOC_Srcs}
  ${QRC_Srcs}
  ${QRC_Srcs}
)


add_executable(Calibration MACOSX_BUNDLE
  ${Srcs} ${Hdrs} ${UI_Srcs} ${MOC_Hdrs} ${QRC_Srcs})
qt5_use_modules(Calibration Core Gui Widgets)
target_link_libraries(Calibration
flycapture
${QT_LIBRARIES}
${PCL_COMMON_LIBRARIES}
${PCL_IO_LIBRARIES}
${PCL_VISUALIZATION_LIBRARIES}
${PCL_OCTREE_LIBRARIES}
${PCL_FEATURES_LIBRARIES}
${PCL_SEGMENTATION_LIBRARIES}
${VTK_LIBRARIES}
${OpenCV_LIBS}
${Boost_LIBRARIES}
pthread
)



