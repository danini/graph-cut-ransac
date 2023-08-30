# Build and install and opencv on Windows

git clone --depth 1 --branch 3.4.5 https://github.com/opencv/opencv.git
mkdir opencv/build
cd opencv/build

# Compile and install OpenCV with contrib modules
cmake -G Ninja `
    -D CMAKE_BUILD_TYPE=RELEASE `
    -D CMAKE_INSTALL_PREFIX="C:/opencv" `
    -D INSTALL_C_EXAMPLES=OFF `
    -D INSTALL_PYTHON_EXAMPLES=OFF `
    -D BUILD_opencv_cudacodec=OFF `
    -D WITH_1394:BOOL=OFF `
    -D WITH_ARAVIS:BOOL=OFF `
    -D WITH_CLP:BOOL=OFF `
    -D WITH_CUDA:BOOL=OFF `
    -D WITH_EIGEN:BOOL=ON `
    -D WITH_FFMPEG:BOOL=OFF `
    -D WITH_GDAL:BOOL=OFF `
    -D WITH_GDCM:BOOL=OFF `
    -D WITH_GIGEAPI:BOOL=OFF `
    -D WITH_GPHOTO2:BOOL=OFF `
    -D WITH_GSTREAMER:BOOL=OFF `
    -D WITH_GSTREAMER_0_10:BOOL=OFF `
    -D WITH_GTK:BOOL=OFF `
    -D WITH_GTK_2_X:BOOL=OFF `
    -D WITH_HALIDE:BOOL=OFF `
    -D WITH_IMGCODEC_HDR:BOOL=OFF `
    -D WITH_IMGCODEC_PXM:BOOL=OFF `
    -D WITH_IMGCODEC_SUNRASTER:BOOL=OFF `
    -D WITH_INF_ENGINE:BOOL=OFF `
    -D WITH_IPP:BOOL=OFF `
    -D WITH_ITT:BOOL=OFF `
    -D WITH_JASPER:BOOL=OFF `
    -D WITH_JPEG:BOOL=OFF `
    -D WITH_LAPACK:BOOL=ON `
    -D WITH_LIBV4L:BOOL=OFF `
    -D WITH_MFX:BOOL=OFF `
    -D WITH_OPENCL:BOOL=OFF `
    -D WITH_OPENCLAMDBLAS:BOOL=OFF `
    -D WITH_OPENCLAMDFFT:BOOL=OFF `
    -D WITH_OPENCL_SVM:BOOL=OFF `
    -D WITH_OPENEXR:BOOL=OFF `
    -D WITH_OPENGL:BOOL=OFF `
    -D WITH_OPENMP:BOOL=OFF `
    -D WITH_OPENNI:BOOL=OFF `
    -D WITH_OPENNI2:BOOL=OFF `
    -D WITH_OPENVX:BOOL=OFF `
    -D WITH_PNG:BOOL=OFF `
    -D WITH_PROTOBUF:BOOL=OFF `
    -D WITH_PTHREADS_PF:BOOL=ON `
    -D WITH_PVAPI:BOOL=OFF `
    -D WITH_QT:BOOL=OFF `
    -D WITH_QUIRC:BOOL=OFF `
    -D WITH_TBB:BOOL=OFF `
    -D WITH_TIFF:BOOL=OFF `
    -D WITH_UNICAP:BOOL=OFF `
    -D WITH_V4L:BOOL=OFF `
    -D WITH_VA:BOOL=OFF `
    -D WITH_VA_INTEL:BOOL=OFF `
    -D WITH_VTK:BOOL=OFF `
    -D WITH_WEBP:BOOL=OFF `
    -D WITH_XIMEA:BOOL=OFF `
    -D WITH_XINE:BOOL=OFF `
    -D BUILD_EXAMPLES=OFF `
    -D BUILD_TESTS:BOOL=OFF `
    -D BUILD_TIFF:BOOL=OFF `
    -D BUILD_USE_SYMLINKS:BOOL=OFF `
    -D BUILD_WEBP:BOOL=OFF `
    -D BUILD_WITH_DEBUG_INFO:BOOL=OFF `
    -D BUILD_WITH_DYNAMIC_IPP:BOOL=OFF `
    -D BUILD_ZLIB:BOOL=OFF `
    -D BUILD_opencv_apps:BOOL=OFF `
    -D BUILD_opencv_dnn:BOOL=OFF `
    -D BUILD_opencv_highgui:BOOL=OFF `
    -D BUILD_opencv_imgcodecs:BOOL=OFF `
    -D BUILD_opencv_java_bindings_generator:BOOL=OFF `
    -D BUILD_opencv_js:BOOL=OFF `
    -D BUILD_opencv_ml:BOOL=OFF `
    -D BUILD_opencv_objdetect:BOOL=OFF `
    -D BUILD_opencv_photo:BOOL=OFF `
    -D BUILD_opencv_python_bindings_generator:BOOL=OFF `
    -D BUILD_opencv_shape:BOOL=OFF `
    -D BUILD_opencv_stitching:BOOL=OFF `
    -D BUILD_opencv_superres:BOOL=OFF `
    -D BUILD_opencv_ts:BOOL=OFF `
    -D BUILD_opencv_video:BOOL=OFF `
    -D BUILD_opencv_videoio:BOOL=OFF `
    -D BUILD_opencv_videostab:BOOL=OFF `
    -D BUILD_opencv_world:BOOL=OFF `
    -D BUILD_opencv_calib3d:BOOL=ON `
    -D BUILD_opencv_features2d:BOOL=ON `
    -D BUILD_opencv_imgproc:BOOL=ON `
    -D BUILD_opencv_core:BOOL=ON `
    -D BUILD_opencv_flann:BOOL=ON ..

ninja
