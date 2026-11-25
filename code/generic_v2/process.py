"""Linux parent-death protection, armed before exec in model-free parents."""
import ctypes
import errno
from functools import partial
import os
import signal
import sys


def arm_parent_death(expected_parent, *, platform=None, libc=None,
                     getppid=os.getppid, getpid=os.getpid, kill=os.kill):
    """Fail closed; close the fork/prctl race by checking the original parent."""
    if (sys.platform if platform is None else platform) != 'linux':
        raise RuntimeError('Parent-death protection requires Linux')
    libc = ctypes.CDLL(None, use_errno=True) if libc is None else libc
    if libc.prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        code = ctypes.get_errno() or errno.EINVAL
        raise OSError(code, 'Failed to arm Linux parent-death signal')
    if getppid() != expected_parent:
        kill(getpid(), signal.SIGTERM)
        raise RuntimeError('Parent exited before parent-death protection was armed')


def parent_death_options():
    # These launchers have no model/thread pools. The child callback does only
    # libc prctl and PID checks, before Python exec resets signal handlers.
    if sys.platform != 'linux':
        return {}
    return {'preexec_fn': partial(arm_parent_death, os.getpid())}
