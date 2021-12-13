# Install Rocky Linux image from docker hub.
FROM rockylinux/rockylinux:8 as rockylinux

# TODO Load this from an environmental/configuration file.
ENV PROJECT_HOME=/projectmetis-rc

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

WORKDIR $PROJECT_HOME
COPY . .

# Since we copy the entire project, we need to clear any pre-existing
# python configuration inside python/metisvenv (default working dir: $PROJECT_HOME)
# else python will not be properly configured inside the image. Also need to install
# the python development tools using python2*/build or python3*/build based on version.
# Copy all ProjectMetis content.
RUN rm -rf python/metisvenv
RUN yum -y module install python38/build
RUN alternatives --set python /usr/bin/python3.8
RUN /usr/bin/python3.8 -m venv python/metisvenv

# Final stage - yum clean up.
RUN yum clean all && rm -rf /var/cache/yum

# Build all core projectmetis modules before publishing the docker image.
##RUN bazel query //... | grep -i -e "//encryption" | xargs bazel build
#RUN bazel query //... | grep -i -e "//projectmetis" | xargs bazel build
#RUN bazel --output_user_root=/tmp/metis/bazel build //...
RUN bazel --output_user_root=/tmp/metis/bazel run //projectmetis/python/driver:initialize_controller
#RUN bazel --output_user_root=/tmp/metis/bazel run //projectmetis/python/driver:initialize_learner
#RUN bazel --output_user_root=/tmp/metis/bazel build --disk_cache=/tmp/metis/bazel //projectmetis/python/driver:initialize_learner
#RUN cd /$PROJECT_HOME
#RUN bazel --output_user_root=/tmp/metis/bazel build //projectmetis/python/driver:initialize_learner
# bazel --output_base=/tmp/metis/bazel
# build --disk_cache=/tmp/metis/bazel

#FROM l.gcr.io/google/bazel:latest
#RUN apt update && apt upgrade -y && apt clean
#RUN apt-get install -y python3-venv
#WORKDIR $PROJECT_HOME
#COPY . .
#RUN rm -rf python/metisvenv
#RUN /usr/bin/python3 -m venv python/metisvenv
