Installation
============

Requirements
------------

* Python 3.10 or 3.11 (TensorFlow 2.14.1 compatibility requirement)
* TensorFlow 2.14.1 (automatically installed)

.. note::
   Python 3.12+ is not currently supported due to TensorFlow availability constraints.

Standard Installation
---------------------

We strongly recommend installing pranaam inside a Python virtual environment. (see `venv documentation <https://docs.python.org/3/library/venv.html#creating-virtual-environments>`_)

Install pranaam using pip:

.. code-block:: bash

   pip install pranaam

This installs TensorFlow 2.14.1, which is known to work correctly with the models.

Installation Options
--------------------

For development work:

.. code-block:: bash

   pip install -e .[dev]

For testing:

.. code-block:: bash

   pip install -e .[test]

For documentation building:

.. code-block:: bash

   pip install -e .[docs]

For all optional dependencies:

.. code-block:: bash

   pip install -e .[all]

TensorFlow Compatibility
------------------------

The package requires TensorFlow 2.14.1 with Keras 2.14.0 for model compatibility. If you encounter compatibility issues:

.. code-block:: bash

   pip install 'pranaam[tensorflow-compat]'

Model Downloads
---------------

Models are automatically downloaded from Harvard Dataverse (306MB) and cached locally on first use. Ensure you have:

* Stable internet connection
* At least 500MB free disk space
* Unrestricted access to dataverse.harvard.edu

Verification
------------

Test your installation:

.. code-block:: python

   import pranaam
   result = pranaam.pred_rel("Shah Rukh Khan")
   print(result)

If successful, you should see a pandas DataFrame with prediction results.

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**TensorFlow/Keras Compatibility Errors**

Error: ``"Keras 3 only supports V3 .keras files and legacy H5 format files"``

Solution: Install with ``pip install 'pranaam[tensorflow-compat]'``

**Model Download Issues**

Error: Network timeouts or download failures

Solution: Check internet connection, models are large (306MB)

**Import Errors**

Error: ``pkg_resources`` deprecation warnings

Solution: Already fixed in v0.1.0 (uses ``importlib.resources``)

For additional help, please check our `GitHub Issues <https://github.com/appeler/pranaam/issues>`_.