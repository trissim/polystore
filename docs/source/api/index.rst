API Reference
=============

This section documents the complete API for polystore.

.. toctree::
   :maxdepth: 2
   :caption: API Documentation:

   filemanager
   backends
   registry
   atomic
   exceptions

Core Components
---------------

FileManager
~~~~~~~~~~~

.. autoclass:: polystore.FileManager
   :members:
   :undoc-members:
   :show-inheritance:

BackendRegistry
~~~~~~~~~~~~~~~

.. autoclass:: polystore.BackendRegistry
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The polystore API is organized around several key concepts:

**FileManager**
  The main interface for saving and loading data. Coordinates between different storage backends.

**Storage Backends**
  Pluggable storage implementations (disk, memory, zarr, streaming). Each backend implements the same interface.

**Backend Registry**
  Auto-discovers and manages available storage backends using metaclass registration.

**Atomic Operations**
  Utilities for atomic file writes with automatic locking.

**Exceptions**
  Custom exception hierarchy for storage operations.

Quick Reference
---------------

Common Imports
~~~~~~~~~~~~~~

.. code-block:: python

   from polystore import (
       FileManager,
       BackendRegistry,
       MemoryBackend,
       DiskBackend,
       ZarrBackend,
       atomic_write,
       atomic_write_json,
   )

Main Classes
~~~~~~~~~~~~

========================  =========================================================
Class                     Description
========================  =========================================================
``FileManager``           High-level API for storage operations
``BackendRegistry``       Registry of available storage backends
``MemoryBackend``         In-memory storage backend
``DiskBackend``           Local filesystem storage backend
``ZarrBackend``           Zarr/OME-Zarr array storage backend
``StreamingBackend``      ZeroMQ streaming backend (optional)
========================  =========================================================

Main Functions
~~~~~~~~~~~~~~

=========================  ========================================================
Function                   Description
=========================  ========================================================
``atomic_write()``         Context manager for atomic file writes
``atomic_write_json()``    Atomically write JSON data to file
=========================  ========================================================

Exceptions
~~~~~~~~~~

===========================  ====================================================
Exception                    Description
===========================  ====================================================
``StorageError``             Base exception for storage operations
``StorageResolutionError``   Failed to resolve storage path or backend
``BackendNotFoundError``     Requested backend not available
``UnsupportedFormatError``   File format not supported
``ImageLoadError``           Failed to load image
``ImageSaveError``           Failed to save image
===========================  ====================================================
