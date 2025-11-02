Quick Start Guide
=================

This guide will help you get started with polystore in just a few minutes.

Basic Usage
-----------

Save and Load with Disk Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to use polystore is with the disk backend:

.. code-block:: python

   import numpy as np
   from polystore import FileManager, BackendRegistry
   
   # Create a backend registry and file manager
   registry = BackendRegistry()
   fm = FileManager(registry)
   
   # Create some data
   data = np.array([[1, 2, 3], [4, 5, 6]])
   
   # Save to disk
   fm.save(data, "output.npy", backend="disk")
   
   # Load from disk
   loaded = fm.load("output.npy", backend="disk")
   
   print(loaded)
   # Output: [[1 2 3]
   #          [4 5 6]]

Memory Backend for Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The memory backend is perfect for testing without touching the filesystem:

.. code-block:: python

   import numpy as np
   from polystore import FileManager, BackendRegistry
   
   registry = BackendRegistry()
   fm = FileManager(registry)
   
   # Create directory structure in memory
   fm.ensure_directory("/test", backend="memory")
   
   # Save to memory
   data = np.array([1, 2, 3])
   fm.save(data, "/test/data.npy", backend="memory")
   
   # Load from memory
   loaded = fm.load("/test/data.npy", backend="memory")
   
   print(loaded)  # Output: [1 2 3]

Batch Operations
----------------

Save and load multiple files efficiently:

.. code-block:: python

   import numpy as np
   from polystore import FileManager, BackendRegistry
   
   registry = BackendRegistry()
   fm = FileManager(registry)
   
   # Create multiple arrays
   data_list = [
       np.array([1, 2, 3]),
       np.array([4, 5, 6]),
       np.array([7, 8, 9])
   ]
   
   paths = ["data1.npy", "data2.npy", "data3.npy"]
   
   # Save all at once
   fm.save_batch(data_list, paths, backend="disk")
   
   # Load all at once
   loaded_list = fm.load_batch(paths, backend="disk")
   
   for i, arr in enumerate(loaded_list):
       print(f"Array {i}: {arr}")

Multi-Framework Support
-----------------------

Polystore works seamlessly with multiple ML frameworks:

PyTorch Tensors
~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from polystore import FileManager, BackendRegistry
   
   registry = BackendRegistry()
   fm = FileManager(registry)
   
   # Create a PyTorch tensor
   tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
   
   # Save as PyTorch format
   fm.save(tensor, "model.pt", backend="disk")
   
   # Load back as PyTorch tensor
   loaded = fm.load("model.pt", backend="disk")
   print(type(loaded))  # <class 'torch.Tensor'>

JAX Arrays
~~~~~~~~~~

.. code-block:: python

   import jax.numpy as jnp
   from polystore import FileManager, BackendRegistry
   
   registry = BackendRegistry()
   fm = FileManager(registry)
   
   # Create a JAX array
   arr = jnp.array([1, 2, 3, 4, 5])
   
   # Save (automatically converts to NumPy for storage)
   fm.save(arr, "jax_data.npy", backend="disk")
   
   # Load back
   loaded = fm.load("jax_data.npy", backend="disk")

File Formats
------------

Polystore supports multiple file formats with automatic detection:

NumPy Arrays
~~~~~~~~~~~~

.. code-block:: python

   # .npy format (single array)
   fm.save(data, "array.npy", backend="disk")
   
   # .npz format (multiple arrays)
   fm.save({"x": x_data, "y": y_data}, "arrays.npz", backend="disk")

TIFF Images
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   
   # Save as TIFF
   image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
   fm.save(image, "image.tif", backend="disk")

CSV and JSON
~~~~~~~~~~~~

.. code-block:: python

   # Save CSV data
   csv_data = [
       {"name": "Alice", "age": 30},
       {"name": "Bob", "age": 25}
   ]
   fm.save(csv_data, "data.csv", backend="disk")
   
   # Save JSON data
   json_data = {"config": "value", "setting": 42}
   fm.save(json_data, "config.json", backend="disk")

Atomic Operations
-----------------

Polystore provides atomic file operations for safe concurrent access:

Atomic Write
~~~~~~~~~~~~

.. code-block:: python

   from polystore import atomic_write
   
   # Write atomically (all or nothing)
   with atomic_write("output.txt") as f:
       f.write("Important data\n")
       f.write("More important data\n")

Atomic JSON Write
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from polystore import atomic_write_json
   
   data = {"key": "value", "count": 42}
   atomic_write_json(data, "config.json")

Directory Operations
--------------------

Polystore provides filesystem-like operations across all backends:

.. code-block:: python

   from polystore import FileManager, BackendRegistry
   
   registry = BackendRegistry()
   fm = FileManager(registry)
   
   # Create directory
   fm.ensure_directory("data/experiment1", backend="disk")
   
   # List files
   files = fm.list_files("data", backend="disk", recursive=True)
   
   # Check existence
   exists = fm.exists("data/experiment1/result.npy", backend="disk")
   
   # Check if path is file or directory
   is_file = fm.is_file("data/experiment1/result.npy", backend="disk")
   is_dir = fm.is_dir("data/experiment1", backend="disk")

Next Steps
----------

* Read the :doc:`api/index` for detailed API documentation
* Learn about :doc:`guides/custom_backends` to create your own backends
* Explore :doc:`examples/index` for more advanced usage patterns
