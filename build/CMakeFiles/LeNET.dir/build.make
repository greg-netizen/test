# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cristian/Desktop/LeNET

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cristian/Desktop/LeNET/build

# Include any dependencies generated for this target.
include CMakeFiles/LeNET.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/LeNET.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/LeNET.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/LeNET.dir/flags.make

CMakeFiles/LeNET.dir/src/main.cu.o: CMakeFiles/LeNET.dir/flags.make
CMakeFiles/LeNET.dir/src/main.cu.o: CMakeFiles/LeNET.dir/includes_CUDA.rsp
CMakeFiles/LeNET.dir/src/main.cu.o: /home/cristian/Desktop/LeNET/src/main.cu
CMakeFiles/LeNET.dir/src/main.cu.o: CMakeFiles/LeNET.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/cristian/Desktop/LeNET/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/LeNET.dir/src/main.cu.o"
	/usr/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/LeNET.dir/src/main.cu.o -MF CMakeFiles/LeNET.dir/src/main.cu.o.d -x cu -rdc=true -c /home/cristian/Desktop/LeNET/src/main.cu -o CMakeFiles/LeNET.dir/src/main.cu.o

CMakeFiles/LeNET.dir/src/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/LeNET.dir/src/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/LeNET.dir/src/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/LeNET.dir/src/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/LeNET.dir/src/utils.cu.o: CMakeFiles/LeNET.dir/flags.make
CMakeFiles/LeNET.dir/src/utils.cu.o: CMakeFiles/LeNET.dir/includes_CUDA.rsp
CMakeFiles/LeNET.dir/src/utils.cu.o: /home/cristian/Desktop/LeNET/src/utils.cu
CMakeFiles/LeNET.dir/src/utils.cu.o: CMakeFiles/LeNET.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/cristian/Desktop/LeNET/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/LeNET.dir/src/utils.cu.o"
	/usr/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/LeNET.dir/src/utils.cu.o -MF CMakeFiles/LeNET.dir/src/utils.cu.o.d -x cu -rdc=true -c /home/cristian/Desktop/LeNET/src/utils.cu -o CMakeFiles/LeNET.dir/src/utils.cu.o

CMakeFiles/LeNET.dir/src/utils.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/LeNET.dir/src/utils.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/LeNET.dir/src/utils.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/LeNET.dir/src/utils.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target LeNET
LeNET_OBJECTS = \
"CMakeFiles/LeNET.dir/src/main.cu.o" \
"CMakeFiles/LeNET.dir/src/utils.cu.o"

# External object files for target LeNET
LeNET_EXTERNAL_OBJECTS =

CMakeFiles/LeNET.dir/cmake_device_link.o: CMakeFiles/LeNET.dir/src/main.cu.o
CMakeFiles/LeNET.dir/cmake_device_link.o: CMakeFiles/LeNET.dir/src/utils.cu.o
CMakeFiles/LeNET.dir/cmake_device_link.o: CMakeFiles/LeNET.dir/build.make
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_alphamat.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_barcode.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_cvv.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_intensity_transform.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_mcc.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_rapid.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_wechat_qrcode.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.6.0
CMakeFiles/LeNET.dir/cmake_device_link.o: CMakeFiles/LeNET.dir/deviceLinkLibs.rsp
CMakeFiles/LeNET.dir/cmake_device_link.o: CMakeFiles/LeNET.dir/deviceObjects1.rsp
CMakeFiles/LeNET.dir/cmake_device_link.o: CMakeFiles/LeNET.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/cristian/Desktop/LeNET/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA device code CMakeFiles/LeNET.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LeNET.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/LeNET.dir/build: CMakeFiles/LeNET.dir/cmake_device_link.o
.PHONY : CMakeFiles/LeNET.dir/build

# Object files for target LeNET
LeNET_OBJECTS = \
"CMakeFiles/LeNET.dir/src/main.cu.o" \
"CMakeFiles/LeNET.dir/src/utils.cu.o"

# External object files for target LeNET
LeNET_EXTERNAL_OBJECTS =

LeNET: CMakeFiles/LeNET.dir/src/main.cu.o
LeNET: CMakeFiles/LeNET.dir/src/utils.cu.o
LeNET: CMakeFiles/LeNET.dir/build.make
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_alphamat.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_barcode.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_cvv.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_intensity_transform.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_mcc.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_rapid.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_wechat_qrcode.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.6.0
LeNET: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.6.0
LeNET: CMakeFiles/LeNET.dir/cmake_device_link.o
LeNET: CMakeFiles/LeNET.dir/linkLibs.rsp
LeNET: CMakeFiles/LeNET.dir/objects1.rsp
LeNET: CMakeFiles/LeNET.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/cristian/Desktop/LeNET/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CUDA executable LeNET"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LeNET.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/LeNET.dir/build: LeNET
.PHONY : CMakeFiles/LeNET.dir/build

CMakeFiles/LeNET.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/LeNET.dir/cmake_clean.cmake
.PHONY : CMakeFiles/LeNET.dir/clean

CMakeFiles/LeNET.dir/depend:
	cd /home/cristian/Desktop/LeNET/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cristian/Desktop/LeNET /home/cristian/Desktop/LeNET /home/cristian/Desktop/LeNET/build /home/cristian/Desktop/LeNET/build /home/cristian/Desktop/LeNET/build/CMakeFiles/LeNET.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/LeNET.dir/depend

