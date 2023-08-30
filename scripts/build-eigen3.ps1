# Build eigen v3.4.0 for Windows
git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen
mkdir eigen/build
cd eigen/build

cmake .. -DCMAKE_INSTALL_PREFIX=C:/eigen3
cmake --build . --target install
