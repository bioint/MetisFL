import argparse
import errno
import os
import platform
import shutil


def is_windows():
    return platform.system() == 'Windows'


def is_linux():
    return platform.system() == 'Linux'


def is_macos():
    return platform.system() == 'Darwin'


def is_ppc64le():
    return platform.machine() == 'ppc64le'


def is_cygwin():
    return platform.system().startswith('CYGWIN_NT')


def symlink_force(target, link_name):
    """Force symlink, equivalent of 'ln -sf'.
    Args:
      target: items to link to.
      link_name: name of the link.
    """
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def setup_python(build_conda_cuda_env):
    # TODO By Default python version is 3.8.8. We might need to make this
    #  user specific. Conda downloads the python interpreter.

    if not build_conda_cuda_env or any([is_macos(), is_windows(), is_cygwin()]):
        src = os.path.join(os.getcwd(), "python/py38_condaenv.yaml")
    else:
        src = os.path.join(os.getcwd(), "python/py38_condaenvcuda.yaml")
    dst = os.path.join(os.getcwd(), "python/conda_env.yaml")
    if os.path.isfile(dst):
        os.remove(dst)
    # We do not create a symlink, because when copying the files inside the image,
    # the content of symlinked files are not transferred. Hence, the hard copy.
    # symlink_force(source, target)
    shutil.copy(src, dst)


def main(build_conda_cuda_env):
    setup_python(build_conda_cuda_env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='MetisFL Python environment configurator.',
        description='Configures whether python conda should be built against a cuda or a non-cuda environment.')
    parser.add_argument("--build_conda_cuda_env", type=int, required=True)
    args = parser.parse_args()
    main(args.build_conda_cuda_env)
