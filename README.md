# Jetson_Slam

## how to setup on Jetson Nano

### compile pyrealsense2

    1. download source code https://github.com/IntelRealSense/librealsense.git
    2. add system variables
        2.1 export PATH=/usr/local/cuda/bin:$PATH
            export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    3. compile\
        $ mkdir build
        $ cd build
        $ cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=/usr/bin/python3 -DBUILD_WITH_CUDA=true
        $ sudo make uninstall && make clean
        $ make -j4
        $ sudo make install
    4. edit ~/.bashrc
        export PYTHONPATH=$PYTHONPATH:/usr/local/lib:/usr/local/lib/python3.6/pyrealsense2


### compile open3d

    1. download source code https://github.com/isl-org/Open3D.git
    2. cmake upgrade
        # uninstall cmake
        sudo apt remove --purge cmake


        #download cmake
        wget -c https://github.com/Kitware/CMake/releases/download/v3.25.0-rc4/cmake-3.25.0-rc4-linux-aarch64.sh
        chmod +x ./cmake-3.25.0-rc4-linux-aarch64.sh
        sudo ./cmake-3.25.0-rc4-linux-aarch64.sh \
            --skip-license \
            --exclude-subdir \
            --prefix=/usr/local

        #check cmake version
        $ cmake --version
    3. compile open3d
        3.1 Check the output of uname -p and it should show aarch64.
        3.2 install dependencies
            util/install_deps_ubuntu.sh 
        3.3 option: install ccache if system disk is large enough
        3.4 activate virtual env
            sudo apt-get install -y python3-virtualenv
            virtualenv --python=$(which python3) $/mnt/venv
            source $mnt/venv/bin/activate
        3.5 clone
            git clone --recursive https://github.com/intel-isl/Open3D
            cd Open3D
            git submodule update --init --recursive
            mkdir build
            cd build
        3.6 config 
            see all configurations in cmakefile.txt
            cmake \
                -DCMAKE_BUILD_TYPE=Release \
                -DBUILD_SHARED_LIBS=ON \
                -DBUILD_CUDA_MODULE=ON \
                -DBUILD_GUI=ON \
                -DBUILD_TENSORFLOW_OPS=OFF \
                -DBUILD_PYTORCH_OPS=OFF \
                -DBUILD_UNIT_TESTS=OFF \
                -DBUILD_WEBRTC=OFF \
                -DBUILD_JUPYTER_EXTENSION=OFF \
                -DCMAKE_INSTALL_PREFIX=/mnt/open3d_install \
                -DPYTHON_EXECUTABLE=$(which python) \
                ..
        3.7 make -j$(nproc) &  make install-pip-package -j$(nproc)

        3.8 In package, dependence needs nbformat==5.5.0, but nbformat's version in Ubuntu18.04 is only up to 5.1.3. 
            Since nbformat is just used for Juypter visualization, if we don't need it, we can manually install open3d using pip install --no-deps open3d-0.16.1+7338831c-cp36-cp36m-manylinux_2_27_aarch64.whl from build/lib/python_package/pip _package

        3.9 test
            python -c "import open3d; print(open3d.__version__)"
            python -c "import open3d; print(open3d.core.cuda.is_available())"
            PS: Install every package that is missing to import open3d.

### Illegal instruction(core dumped) error on Jetson Nano
    nano ~/.bashrc
    export OPENBLAS_CORETYPE=ARMV8

    Or downgrade numpy from 1.19.5 to 1.19.4 (not validated)

### out of space when compiling

    sudo apt-get autoclean

    delete all *.gz && *.1 backup logs in /var/log/
    clear current log files in /var/log
        1、sudo -i
        2、echo > /var/log/syslog
        3、echo > /var/log/kern.log






