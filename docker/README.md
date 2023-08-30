# Manylinux Wheels for Graph-Cut-RANSAC with OpenCV and Eigen

This directory is used to create a docker image to support the building manylinux wheels for the project [Graph-Cut-RANSAC](https://github.com/danini/graph-cut-ransac), requiring OpenCV 3.4.5 and Eigen. The build is optimized for OpenCV with the absolute minimum number of modules. This optimization is important because `auditwheel` will attempt to include all modules when they're all built, which could significantly increase the build time and result in an unnecessarily large wheel file.

By minimizing the number of modules in OpenCV, we're able to reduce the build time and create a more efficient wheel file.

## Getting Started

To build this image, navigate to the `docker` directory and use the following command:

```bash
cd docker
docker build -t manylinux2014_x86_64_opencv3:latest .
```

## Details

This repository provides the Dockerfile needed to create a `manylinux2014` docker image that includes the following:

- OpenCV v3.4.5
- Eigen
- gflag

This image is tagged as `manylinux2014_x86_64_opencv3:latest`.


## License

This repository uses the MIT license. Please refer to the original projects for
licensing information:

- [Graph-Cut-RANSAC](https://github.com/danini/graph-cut-ransac)
- [OpenCV](https://github.com/opencv/opencv)
- [Eigen](https://github.com/libigl/eigen)
