Storage Backends
================

Polystore provides multiple storage backend implementations, each optimized for different use cases.

.. toctree::
   :maxdepth: 2

Available Backends
------------------

DiskBackend
~~~~~~~~~~~

.. autoclass:: polystore.DiskBackend
   :members:
   :undoc-members:
   :show-inheritance:

The ``DiskBackend`` provides persistent storage on the local filesystem.

**Features:**
  * Supports multiple file formats (NumPy, TIFF, CSV, JSON, PyTorch, etc.)
  * Automatic format detection based on file extension
  * Multi-framework support (NumPy, PyTorch, JAX, TensorFlow, CuPy)
  * Atomic file writes for safe concurrent access

**Supported Formats:**

=================  ========================  ================================
Format             Extensions                Frameworks
=================  ========================  ================================
NumPy              .npy, .npz                NumPy, PyTorch, JAX, TF, CuPy
TIFF               .tif, .tiff               NumPy, PyTorch, JAX, TF, CuPy
PyTorch            .pt, .pth                 PyTorch
CSV                .csv                      NumPy, pandas
JSON               .json                     Python dicts
Text               .txt                      Strings
=================  ========================  ================================

**Example:**

.. code-block:: python

   from polystore import DiskBackend
   import numpy as np
   
   backend = DiskBackend()
   
   # Save NumPy array
   data = np.array([1, 2, 3])
   backend.save(data, "output.npy")
   
   # Load data
   loaded = backend.load("output.npy")

MemoryBackend
~~~~~~~~~~~~~

.. autoclass:: polystore.MemoryBackend
   :members:
   :undoc-members:
   :show-inheritance:

The ``MemoryBackend`` provides volatile in-memory storage.

**Features:**
  * Fast access without disk I/O
  * Perfect for testing
  * Supports directory structure
  * Can use shared dictionaries for multiprocessing

**Use Cases:**
  * Unit testing
  * Caching intermediate results
  * Development and debugging
  * Multiprocessing with shared memory

**Example:**

.. code-block:: python

   from polystore import MemoryBackend
   import numpy as np
   
   backend = MemoryBackend()
   
   # Create directory structure
   backend.ensure_directory("/test")
   
   # Save to memory
   data = np.array([1, 2, 3])
   backend.save(data, "/test/data.npy")
   
   # Load from memory
   loaded = backend.load("/test/data.npy")

**Multiprocessing Example:**

.. code-block:: python

   from multiprocessing import Manager, Process
   from polystore import MemoryBackend
   
   def worker(shared_dict):
       backend = MemoryBackend(shared_dict=shared_dict)
       backend.ensure_directory("/shared")
       backend.save(np.array([1, 2, 3]), "/shared/data.npy")
   
   manager = Manager()
   shared = manager.dict()
   
   p = Process(target=worker, args=(shared,))
   p.start()
   p.join()
   
   # Access data from main process
   backend = MemoryBackend(shared_dict=shared)
   data = backend.load("/shared/data.npy")

ZarrBackend
~~~~~~~~~~~

.. autoclass:: polystore.ZarrBackend
   :members:
   :undoc-members:
   :show-inheritance:

The ``ZarrBackend`` provides chunked array storage using Zarr.

**Features:**
  * Efficient storage of large arrays
  * Compression support
  * OME-Zarr format for microscopy images
  * Cloud storage compatible
  * Parallel I/O

**Requires:**
  Install with: ``pip install polystore[zarr]``

**Example:**

.. code-block:: python

   from polystore import ZarrBackend
   import numpy as np
   
   backend = ZarrBackend()
   
   # Save large array with chunking
   large_data = np.random.rand(1000, 1000, 1000)
   backend.save(large_data, "data.zarr", chunks=(100, 100, 100))
   
   # Load array
   loaded = backend.load("data.zarr")

StreamingBackend
~~~~~~~~~~~~~~~~

The ``StreamingBackend`` provides real-time data streaming via ZeroMQ.

**Features:**
  * Real-time data transmission
  * Push/pull patterns
  * Network transparency
  * Minimal latency

**Requires:**
  Install with: ``pip install polystore[streaming]``

**Example:**

.. code-block:: python

   from polystore import StreamingBackend
   import numpy as np
   
   # Sender
   backend = StreamingBackend(mode="push", port=5555)
   data = np.array([1, 2, 3])
   backend.save(data, "stream")
   
   # Receiver
   backend = StreamingBackend(mode="pull", port=5555)
   received = backend.load("stream")

Backend Interface
-----------------

All backends implement the same interface defined by abstract base classes.

DataSink
~~~~~~~~

.. autoclass:: polystore.DataSink
   :members:
   :undoc-members:
   :show-inheritance:

Abstract interface for write operations.

**Methods:**
  * ``save(data, identifier, **kwargs)`` - Save data
  * ``save_batch(data_list, identifiers, **kwargs)`` - Save multiple items

DataSource
~~~~~~~~~~

.. autoclass:: polystore.DataSource
   :members:
   :undoc-members:
   :show-inheritance:

Abstract interface for read operations.

**Methods:**
  * ``load(file_path, **kwargs)`` - Load data
  * ``load_batch(file_paths, **kwargs)`` - Load multiple items
  * ``list_files(directory, **kwargs)`` - List files
  * ``exists(path)`` - Check existence
  * ``is_file(path)`` - Check if file
  * ``is_dir(path)`` - Check if directory
  * ``list_dir(path)`` - List directory entries

StorageBackend
~~~~~~~~~~~~~~

.. autoclass:: polystore.StorageBackend
   :members:
   :undoc-members:
   :show-inheritance:

Base class for read-write storage backends.
Combines ``DataSink`` and ``DataSource`` interfaces.

Creating Custom Backends
-------------------------

You can create custom backends by inheriting from ``StorageBackend``:

.. code-block:: python

   from polystore import StorageBackend
   
   class MyBackend(StorageBackend):
       _backend_type = 'my_backend'  # Auto-registers
       
       def save(self, data, file_path, **kwargs):
           # Your save logic
           pass
       
       def load(self, file_path, **kwargs):
           # Your load logic
           pass
       
       def list_files(self, directory, **kwargs):
           # Your list logic
           pass
       
       # Implement other required methods...

The backend will be automatically registered and available via the registry:

.. code-block:: python

   from polystore import BackendRegistry, FileManager
   
   registry = BackendRegistry()
   # 'my_backend' is now available
   fm = FileManager(registry)
   fm.save(data, "output.dat", backend="my_backend")

See Also
--------

* :doc:`../guides/custom_backends` - Detailed guide for creating custom backends
* :doc:`filemanager` - Using backends via FileManager
* :doc:`registry` - Backend registration system
