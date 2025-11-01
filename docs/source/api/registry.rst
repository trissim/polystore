Backend Registry
================

The backend registry system provides automatic discovery and registration of storage backends
using metaclass-based registration.

BackendRegistry
---------------

.. autoclass:: polystore.BackendRegistry
   :members:
   :undoc-members:
   :show-inheritance:

The ``BackendRegistry`` class manages available storage backends and provides access to backend instances.

**Example:**

.. code-block:: python

   from polystore import BackendRegistry
   
   # Create registry (auto-discovers backends)
   registry = BackendRegistry()
   
   # List available backends
   print(list(registry.keys()))
   # Output: ['memory', 'disk', 'zarr']
   
   # Access a backend
   disk_backend = registry['disk']
   
   # Use with FileManager
   from polystore import FileManager
   fm = FileManager(registry)

Auto-Registration
-----------------

Backends are automatically registered when their classes are defined, using the ``AutoRegisterMeta`` metaclass.

How It Works
~~~~~~~~~~~~

1. Backends inherit from ``StorageBackend`` or ``ReadOnlyBackend``
2. They define a ``_backend_type`` class attribute
3. The metaclass automatically registers them in the global registry
4. ``BackendRegistry`` discovers and instantiates them on first access

**Example:**

.. code-block:: python

   from polystore import StorageBackend
   
   class MyBackend(StorageBackend):
       _backend_type = 'my_backend'  # Registration key
       
       def save(self, data, file_path, **kwargs):
           # Implementation
           pass
       
       def load(self, file_path, **kwargs):
           # Implementation
           pass
       
       # ... other required methods

Once defined, the backend is automatically available:

.. code-block:: python

   from polystore import BackendRegistry
   
   registry = BackendRegistry()
   assert 'my_backend' in registry

AutoRegisterMeta
----------------

The registration system uses ``AutoRegisterMeta`` from the external ``metaclass-registry`` package.

**Key Features:**
  * Automatic class registration at definition time
  * Support for abstract base classes (not registered)
  * Clean separation of concerns
  * No manual registration required

**Registry Key:**
  Backends use ``_backend_type`` as the registry key:

.. code-block:: python

   class DiskBackend(StorageBackend):
       _backend_type = 'disk'  # Registered as 'disk'

**Abstract Classes:**
  Base classes with abstract methods are not registered:

.. code-block:: python

   from abc import abstractmethod
   
   class BaseBackend(StorageBackend):
       _backend_type = 'base'  # Not registered (abstract)
       
       @abstractmethod
       def custom_method(self):
           pass

Backend Discovery
-----------------

The registry discovers backends through import-time registration:

1. **Module Import**: When a module containing a backend class is imported, the class is defined
2. **Metaclass Hook**: ``AutoRegisterMeta.__new__()`` is called during class creation
3. **Registration**: The backend is added to the global registry
4. **Discovery**: ``BackendRegistry`` accesses the registry to find all registered backends

**Import Trigger:**

Backends must be imported before they can be discovered. The core backends
(disk, memory, zarr) are imported automatically by ``BackendRegistry``.

For custom backends, ensure they are imported:

.. code-block:: python

   # Import your custom backend module
   import my_package.my_backend
   
   # Now it's available in the registry
   from polystore import BackendRegistry
   registry = BackendRegistry()
   assert 'my_backend' in registry

Lazy Instantiation
------------------

The registry uses lazy instantiation for efficiency:

1. **Discovery**: All backend classes are discovered at registry creation
2. **Instantiation**: Backends are instantiated only when first accessed
3. **Caching**: Instances are cached for reuse

**Example:**

.. code-block:: python

   from polystore import BackendRegistry
   
   # Registry created, backends discovered but not instantiated
   registry = BackendRegistry()
   
   # First access instantiates the backend
   disk = registry['disk']  # DiskBackend() called here
   
   # Subsequent access returns cached instance
   disk2 = registry['disk']  # Returns same instance
   assert disk is disk2

Manual Registration
-------------------

While auto-registration is preferred, you can manually register backends:

.. code-block:: python

   from polystore import BackendRegistry, MemoryBackend
   
   # Create custom registry
   registry = {}
   
   # Manually register backends
   registry['memory'] = MemoryBackend()
   registry['custom'] = MyCustomBackend()
   
   # Use with FileManager
   from polystore import FileManager
   fm = FileManager(registry)

Thread Safety
-------------

Backend instances are shared across the registry by default. If your backends
maintain mutable state, consider thread-safety implications:

**Shared State:**

.. code-block:: python

   # Same backend instance used by all FileManagers
   registry = BackendRegistry()
   fm1 = FileManager(registry)
   fm2 = FileManager(registry)
   
   # fm1 and fm2 share the same backend instances

**Isolated State:**

.. code-block:: python

   # Create separate backend instances per FileManager
   from polystore import MemoryBackend, DiskBackend
   
   registry1 = {'memory': MemoryBackend(), 'disk': DiskBackend()}
   registry2 = {'memory': MemoryBackend(), 'disk': DiskBackend()}
   
   fm1 = FileManager(registry1)
   fm2 = FileManager(registry2)

Best Practices
--------------

1. **Use Auto-Registration**: Prefer metaclass registration over manual registration
2. **Import Early**: Import backend modules early to ensure discovery
3. **Single Registry**: Use a single ``BackendRegistry`` instance per application
4. **Thread Safety**: Be aware of shared backend instances in multi-threaded applications
5. **Naming**: Use lowercase, descriptive names for ``_backend_type``

See Also
--------

* :doc:`backends` - Available storage backends
* :doc:`../guides/custom_backends` - Creating custom backends
* `metaclass-registry <https://github.com/trissim/metaclass-registry>`_ - External registration package
