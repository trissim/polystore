FileManager
===========

The ``FileManager`` class is the main interface for interacting with polystore.
It provides a unified API for saving and loading data across different storage backends.

Class Reference
---------------

.. autoclass:: polystore.FileManager
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

``FileManager`` acts as a coordinator between your application and storage backends.
It handles:

* Routing operations to the appropriate backend
* Managing backend instances
* Providing a consistent API across all backends
* Supporting batch operations for efficiency

Constructor
-----------

.. code-block:: python

   FileManager(registry)

Parameters:
  * ``registry`` - A ``BackendRegistry`` or dict mapping backend names to backend instances

Example:

.. code-block:: python

   from polystore import FileManager, BackendRegistry
   
   registry = BackendRegistry()
   fm = FileManager(registry)

Methods
-------

save()
~~~~~~

Save data to a file using the specified backend.

.. code-block:: python

   fm.save(data, output_path, backend, **kwargs)

Parameters:
  * ``data`` - The data to save (NumPy array, dict, list, etc.)
  * ``output_path`` - Path where data should be saved
  * ``backend`` - Backend name ('disk', 'memory', 'zarr')
  * ``**kwargs`` - Backend-specific arguments

Example:

.. code-block:: python

   import numpy as np
   data = np.array([1, 2, 3])
   fm.save(data, "output.npy", backend="disk")

load()
~~~~~~

Load data from a file using the specified backend.

.. code-block:: python

   data = fm.load(file_path, backend, **kwargs)

Parameters:
  * ``file_path`` - Path to the file to load
  * ``backend`` - Backend name ('disk', 'memory', 'zarr')
  * ``**kwargs`` - Backend-specific arguments

Returns:
  The loaded data

Example:

.. code-block:: python

   data = fm.load("output.npy", backend="disk")

save_batch()
~~~~~~~~~~~~

Save multiple data objects in a single operation.

.. code-block:: python

   fm.save_batch(data_list, output_paths, backend, **kwargs)

Parameters:
  * ``data_list`` - List of data objects to save
  * ``output_paths`` - List of output paths (must match length of data_list)
  * ``backend`` - Backend name
  * ``**kwargs`` - Backend-specific arguments

Example:

.. code-block:: python

   data_list = [np.array([1, 2]), np.array([3, 4])]
   paths = ["data1.npy", "data2.npy"]
   fm.save_batch(data_list, paths, backend="disk")

load_batch()
~~~~~~~~~~~~

Load multiple files in a single operation.

.. code-block:: python

   data_list = fm.load_batch(file_paths, backend, **kwargs)

Parameters:
  * ``file_paths`` - List of file paths to load
  * ``backend`` - Backend name
  * ``**kwargs`` - Backend-specific arguments

Returns:
  List of loaded data objects in the same order as file_paths

Example:

.. code-block:: python

   paths = ["data1.npy", "data2.npy"]
   data_list = fm.load_batch(paths, backend="disk")

Directory Operations
--------------------

list_files()
~~~~~~~~~~~~

List files in a directory.

.. code-block:: python

   files = fm.list_files(directory, backend, pattern=None, 
                        extensions=None, recursive=False)

Parameters:
  * ``directory`` - Directory to search
  * ``backend`` - Backend name
  * ``pattern`` - Optional glob pattern (e.g., "*.npy")
  * ``extensions`` - Optional set of extensions to filter (e.g., {'.npy', '.npz'})
  * ``recursive`` - Whether to search recursively

Returns:
  List of file paths

Example:

.. code-block:: python

   # List all .npy files recursively
   files = fm.list_files("data", backend="disk", 
                        extensions={'.npy'}, recursive=True)

ensure_directory()
~~~~~~~~~~~~~~~~~~

Create a directory if it doesn't exist.

.. code-block:: python

   path = fm.ensure_directory(directory, backend)

Parameters:
  * ``directory`` - Directory path to create
  * ``backend`` - Backend name

Returns:
  String path to the directory

Example:

.. code-block:: python

   fm.ensure_directory("data/experiment1", backend="disk")

exists()
~~~~~~~~

Check if a path exists.

.. code-block:: python

   exists = fm.exists(path, backend)

Parameters:
  * ``path`` - Path to check
  * ``backend`` - Backend name

Returns:
  True if path exists, False otherwise

is_file()
~~~~~~~~~

Check if a path is a file.

.. code-block:: python

   is_file = fm.is_file(path, backend)

is_dir()
~~~~~~~~

Check if a path is a directory.

.. code-block:: python

   is_dir = fm.is_dir(path, backend)

Thread Safety
-------------

Each ``FileManager`` instance should be scoped to a single execution context.
Do not share ``FileManager`` instances across threads.

For multi-threaded applications, create a separate ``FileManager`` instance
for each thread, optionally sharing the same registry if backends are thread-safe.

Backend-Specific Features
--------------------------

Some backends support additional features accessible via kwargs:

Disk Backend
~~~~~~~~~~~~

.. code-block:: python

   # Save with metadata
   fm.save(data, "output.npy", backend="disk", metadata={"key": "value"})

Memory Backend
~~~~~~~~~~~~~~

.. code-block:: python

   # Use shared dictionary for multiprocessing
   from multiprocessing import Manager
   manager = Manager()
   shared_dict = manager.dict()
   
   backend = MemoryBackend(shared_dict=shared_dict)
   registry = {"memory": backend}
   fm = FileManager(registry)

See Also
--------

* :doc:`backends` - Documentation for specific backends
* :doc:`registry` - Backend registration system
* :doc:`../quickstart` - Quick start guide with examples
