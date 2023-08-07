""" Backend function (Now, it only supports TF.)

"""

from __future__ import absolute_import
from __future__ import print_function
import os
import json
import sys
import importlib

# Default backend: tf.Keras
_BACKEND = 'tensorflow'

if 'BESPOKE_HOME' in os.environ:
    _bespoke_dir = os.environ.get('BESPOKE_HOME')
else:
    _bespoke_base_dir = os.path.expanduser('~')
    if not os.access(_bespoke_base_dir, os.W_OK):
        _bespoke_base_dir = '/tmp'
    _bespoke_dir = os.path.join(_bespoke_base_dir, '.nncps')

# Attempt to read bespoke config file.
_config_path = os.path.expanduser(os.path.join(_bespoke_dir, 'bespoke.json'))
if os.path.exists(_config_path):
    try:
        with open(_config_path) as f:
            _config = json.load(f)
    except ValueError:
        _config = {}
    _BACKEND = _config.get('backend', _BACKEND)

# Save config file, if possible.
if not os.path.exists(_bespoke_dir):
    try:
        os.makedirs(_bespoke_dir)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass

if not os.path.exists(_config_path):
    _config = {
        'backend': _BACKEND,
    }
    try:
        with open(_config_path, 'w') as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        # Except permission denied.
        pass

# Set backend based on BESPOKE_BACKEND flag, if applicable.
if 'BESPOKE_BACKEND' in os.environ:
    _backend = os.environ['BESPOKE_BACKEND']
    if _backend:
        _BACKEND = _backend

# Import backend functions.
if _BACKEND == 'tensorflow':
    sys.stderr.write('Using TensorFlow backend\n')
    from .tensorflow_ import *
else:
    raise ValueError('Unable to import backend : ' + str(_BACKEND))


def backend():
    """Returns the name of the current backend (e.g. "tensorflow").

    # Returns
        String, the name of the backend is currently using.

    # Example
    ```python
        >>> bespoke.backend.backend()
        'tensorflow'
    ```
    """
    return _BACKEND
