#### CENTOS ####
# Install Rockey Linux image from docker hub.
FROM rockylinux/rockylinux:8 as rockylinux

# TODO Load this from an environmental/configuration file.
ENV PROJECT_HOME=projectmetis-rc


# Just get the latest/updated libs for the current distribution.
# TODO clean at the end of the build process, because this one populates the cache.
RUN yum -y update
RUN yum -y groupinstall "Development Tools"

# Downaload and Set Devtoolset-9 for Palisades compilation.
RUN yum -y install gcc-toolset-9-gcc gcc-toolset-9-gcc-c++
RUN echo "source /opt/rh/gcc-toolset-9/enable" >> /etc/bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Download .repo file for bazel installation.
RUN cd /etc/yum.repos.d/ && { curl -O https://copr.fedorainfracloud.org/coprs/vbatts/bazel/repo/epel-7/vbatts-bazel-epel-7.repo; cd -; }
RUN yum -y install bazel4

# Install Python3 and create ProjectMetis virtual environment.
RUN yum -y install python3-devel
RUN yum -y module enable python36
RUN yum -y install python36
RUN alternatives --set python /usr/bin/python3
RUN python3 -m venv $PROJECT_HOME/python/metisvenv

# Final stage - yum clean up.
RUN yum clean all && rm -rf /var/cache/yum

# Copy ProjectMetis content.
WORKDIR $PROJECT_HOME
COPY . .