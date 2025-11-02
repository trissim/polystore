Exceptions
==========

Polystore defines a hierarchy of exceptions for storage operations.

Exception Hierarchy
-------------------

.. code-block:: text

   Exception
   └── StorageError (base exception)
       ├── StorageResolutionError
       ├── BackendNotFoundError
       ├── UnsupportedFormatError
       └── ... (runtime errors)
           ├── ImageLoadError
           ├── ImageSaveError
           └── StorageWriteError

Base Exception
--------------

StorageError
~~~~~~~~~~~~

.. autoexception:: polystore.StorageError
   :members:
   :show-inheritance:

Base exception for all polystore storage operations.

**When to Catch:**
  Catch this to handle any polystore-related error.

**Example:**

.. code-block:: python

   from polystore import FileManager, BackendRegistry, StorageError
   
   registry = BackendRegistry()
   fm = FileManager(registry)
   
   try:
       data = fm.load("data.npy", backend="disk")
   except StorageError as e:
       print(f"Storage operation failed: {e}")

Storage Exceptions
------------------

StorageResolutionError
~~~~~~~~~~~~~~~~~~~~~~

.. autoexception:: polystore.StorageResolutionError
   :members:
   :show-inheritance:

Raised when a storage path or backend cannot be resolved.

**Common Causes:**
  * Invalid backend name
  * Path format not compatible with backend
  * Backend configuration error

**Example:**

.. code-block:: python

   from polystore import FileManager, BackendRegistry
   from polystore import StorageResolutionError
   
   registry = BackendRegistry()
   fm = FileManager(registry)
   
   try:
       # Invalid backend name
       fm.save(data, "output.npy", backend="invalid")
   except StorageResolutionError as e:
       print(f"Could not resolve backend: {e}")

BackendNotFoundError
~~~~~~~~~~~~~~~~~~~~

.. autoexception:: polystore.BackendNotFoundError
   :members:
   :show-inheritance:

Raised when a requested backend is not available.

**Common Causes:**
  * Backend not registered
  * Optional backend not installed
  * Typo in backend name

**Example:**

.. code-block:: python

   from polystore import BackendRegistry, BackendNotFoundError
   
   registry = BackendRegistry()
   
   try:
       backend = registry['nonexistent']
   except BackendNotFoundError as e:
       print(f"Backend not found: {e}")
       print(f"Available: {list(registry.keys())}")

UnsupportedFormatError
~~~~~~~~~~~~~~~~~~~~~~

.. autoexception:: polystore.UnsupportedFormatError
   :members:
   :show-inheritance:

Raised when a file format is not supported by the backend.

**Common Causes:**
  * Unknown file extension
  * Format not registered
  * Missing optional dependency

**Example:**

.. code-block:: python

   from polystore import FileManager, BackendRegistry
   from polystore import UnsupportedFormatError
   
   registry = BackendRegistry()
   fm = FileManager(registry)
   
   try:
       # Unsupported extension
       data = fm.load("data.xyz", backend="disk")
   except UnsupportedFormatError as e:
       print(f"Format not supported: {e}")

Runtime Exceptions
------------------

ImageLoadError
~~~~~~~~~~~~~~

.. autoexception:: polystore.ImageLoadError
   :members:
   :show-inheritance:

Raised when image loading fails.

**Common Causes:**
  * Corrupted image file
  * Invalid image format
  * Missing image library (tifffile, PIL)

**Example:**

.. code-block:: python

   from polystore import FileManager, BackendRegistry
   from polystore import ImageLoadError
   
   registry = BackendRegistry()
   fm = FileManager(registry)
   
   try:
       image = fm.load("corrupted.tif", backend="disk")
   except ImageLoadError as e:
       print(f"Failed to load image: {e}")

ImageSaveError
~~~~~~~~~~~~~~

.. autoexception:: polystore.ImageSaveError
   :members:
   :show-inheritance:

Raised when image saving fails.

**Common Causes:**
  * Invalid image data
  * Disk space issues
  * Permission errors

**Example:**

.. code-block:: python

   from polystore import FileManager, BackendRegistry
   from polystore import ImageSaveError
   import numpy as np
   
   registry = BackendRegistry()
   fm = FileManager(registry)
   
   try:
       # Invalid image data
       invalid_data = np.array([1, 2, 3])  # 1D array, not 2D image
       fm.save(invalid_data, "image.tif", backend="disk")
   except ImageSaveError as e:
       print(f"Failed to save image: {e}")

StorageWriteError
~~~~~~~~~~~~~~~~~

.. autoexception:: polystore.StorageWriteError
   :members:
   :show-inheritance:

Raised when writing to storage fails.

**Common Causes:**
  * Disk full
  * Permission denied
  * Read-only filesystem

**Example:**

.. code-block:: python

   from polystore import FileManager, BackendRegistry
   from polystore import StorageWriteError
   
   registry = BackendRegistry()
   fm = FileManager(registry)
   
   try:
       fm.save(data, "/read-only-path/output.npy", backend="disk")
   except StorageWriteError as e:
       print(f"Write failed: {e}")

VFS Exceptions
--------------

PathMismatchError
~~~~~~~~~~~~~~~~~

.. autoexception:: polystore.PathMismatchError
   :members:
   :show-inheritance:

Raised when a path scheme doesn't match the expected scheme for a backend.

**Common Causes:**
  * Using absolute paths with memory backend expecting relative
  * Using relative paths when absolute required
  * Platform-specific path issues

**Example:**

.. code-block:: python

   from polystore import FileManager, BackendRegistry
   from polystore import PathMismatchError
   
   registry = BackendRegistry()
   fm = FileManager(registry)
   
   try:
       # Path format mismatch
       fm.load("C:\\windows\\path", backend="memory")
   except PathMismatchError as e:
       print(f"Path format error: {e}")

VFSTypeError
~~~~~~~~~~~~

.. autoexception:: polystore.VFSTypeError
   :members:
   :show-inheritance:

Raised when a type error occurs in the VFS boundary.

**Common Causes:**
  * Invalid path type (not str or Path)
  * Wrong data type for operation

**Example:**

.. code-block:: python

   from polystore import FileManager, BackendRegistry
   from polystore import VFSTypeError
   
   registry = BackendRegistry()
   fm = FileManager(registry)
   
   try:
       # Invalid path type
       fm.load(123, backend="disk")  # Should be str or Path
   except VFSTypeError as e:
       print(f"Type error: {e}")

Error Handling Best Practices
------------------------------

Specific Exceptions
~~~~~~~~~~~~~~~~~~~

Catch specific exceptions when you can handle them:

.. code-block:: python

   from polystore import (
       FileManager, BackendRegistry,
       BackendNotFoundError, UnsupportedFormatError
   )
   
   registry = BackendRegistry()
   fm = FileManager(registry)
   
   try:
       data = fm.load("data.xyz", backend="custom")
   except BackendNotFoundError:
       print("Backend not available, using fallback")
       data = fm.load("data.xyz", backend="disk")
   except UnsupportedFormatError:
       print("Format not supported, trying different format")
       data = fm.load("data.npy", backend="disk")

Broad Exception Handling
~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``StorageError`` for general error handling:

.. code-block:: python

   from polystore import FileManager, BackendRegistry, StorageError
   
   registry = BackendRegistry()
   fm = FileManager(registry)
   
   try:
       result = process_data(fm)
   except StorageError as e:
       logger.error(f"Storage operation failed: {e}")
       # Cleanup or fallback logic
       return default_result()

Logging
~~~~~~~

Always log exceptions for debugging:

.. code-block:: python

   import logging
   from polystore import StorageError
   
   logger = logging.getLogger(__name__)
   
   try:
       fm.save(data, "output.npy", backend="disk")
   except StorageError as e:
       logger.exception("Failed to save data")
       raise

Re-raising
~~~~~~~~~~

Add context when re-raising:

.. code-block:: python

   from polystore import StorageError
   
   def save_results(results, path):
       try:
           fm.save(results, path, backend="disk")
       except StorageError as e:
           raise StorageError(
               f"Failed to save results to {path}: {e}"
           ) from e

See Also
--------

* :doc:`filemanager` - FileManager API that raises these exceptions
* :doc:`backends` - Backend implementations and their errors
* Python's built-in exceptions - Many operations also raise ``FileNotFoundError``, ``PermissionError``, etc.
