import errno
import os
import platform
import subprocess


def is_nvidia_installed():
    installed = True
    try:
        subprocess.check_output('nvidia-smi')
    except Exception:
        installed = False
    return installed


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


def setup_python():
    # TODO By Default python version is 3.8.8. We might need to make this
    #  user specific. Conda downloads the python interpreter.

    if not is_nvidia_installed() or is_macos() or is_windows() or is_cygwin():
        target = os.path.join(os.getcwd(), "python/py38_condaenv.yaml")
    else:
        target = os.path.join(os.getcwd(), "python/py38_condaenvcuda.yaml")
    link_name = os.path.join(os.getcwd(), "python/conda_env.yaml")
    symlink_force(target, link_name)


def main():
    setup_python()


if __name__ == '__main__':
    main()
