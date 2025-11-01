Installation
============

Basic Installation
------------------

Install polystore with pip:

.. code-block:: bash

   pip install polystore

This will install the core package with NumPy and basic storage backends.

Optional Dependencies
---------------------

Polystore supports optional dependencies for additional functionality:

Framework Support
~~~~~~~~~~~~~~~~~

Install support for specific ML frameworks:

.. code-block:: bash

   # PyTorch support
   pip install polystore[torch]
   
   # JAX support
   pip install polystore[jax]
   
   # TensorFlow support
   pip install polystore[tensorflow]
   
   # CuPy support (GPU arrays)
   pip install polystore[cupy]

Zarr Support
~~~~~~~~~~~~

For Zarr and OME-Zarr array storage:

.. code-block:: bash

   pip install polystore[zarr]

Streaming Support
~~~~~~~~~~~~~~~~~

For ZeroMQ streaming backends:

.. code-block:: bash

   pip install polystore[streaming]

All Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

Install all optional dependencies at once:

.. code-block:: bash

   pip install polystore[all]

Development Installation
------------------------

To install for development with testing and documentation tools:

.. code-block:: bash

   git clone https://github.com/trissim/polystore.git
   cd polystore
   pip install -e ".[dev,docs]"

Requirements
------------

* **Python**: 3.11 or later
* **NumPy**: 1.26.0 or later
* **portalocker**: 2.8.0 or later (for atomic file operations)
* **metaclass-registry**: From GitHub (automatic backend registration)

Verifying Installation
----------------------

Verify your installation by importing polystore:

.. code-block:: python

   import polystore
   print(polystore.__version__)
   
   # Check available backends
   from polystore import BackendRegistry
   registry = BackendRegistry()
   print("Available backends:", list(registry.keys()))

You should see output similar to:

.. code-block:: text

   0.1.0
   Available backends: ['memory', 'disk']

If you installed optional dependencies, you may see additional backends like ``zarr``.
