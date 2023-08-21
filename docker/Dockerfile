# Start with the base image
FROM quay.io/pypa/manylinux2014_x86_64:2023-07-14-55e4124

# Copy the shell script from your host to the container
COPY install-opencv-3.4.5-in-centos-7.sh /root/install-opencv-3.4.5-in-centos-7.sh

# Execute the script
RUN /root/install-opencv-3.4.5-in-centos-7.sh