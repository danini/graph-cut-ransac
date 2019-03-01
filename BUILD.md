GraphCut_RANSAC
===============

Build instructions
------------------

Required tools:

- CMake
- Git
- C/C++ compiler (GCC, Visual Studio or Clang)

Required libraries:

- OpenCV with modules

Note:

- CMAKE variables you can configure:

  - USE_OPENMP (ON(default)/OFF)
      - Parallelize using OpenMP
	  
Compiling
---------

```shell
$ mkdir build
$ cd build
$ cmake-gui ..
```

- Configure + Generate
- Set the OpenCV_DIR if needed.