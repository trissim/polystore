Atomic Operations
=================

Polystore provides atomic file operations with automatic locking for safe concurrent access.

Functions
---------

atomic_write()
~~~~~~~~~~~~~~

.. autofunction:: polystore.atomic_write

Context manager for atomic file writes.

**Features:**
  * Writes to temporary file first
  * Atomically renames to target (all-or-nothing)
  * Automatic cleanup on error
  * Cross-platform compatible
  * Supports text and binary modes

**Example - Text Mode:**

.. code-block:: python

   from polystore import atomic_write
   
   # Write text atomically
   with atomic_write("output.txt") as f:
       f.write("Line 1\n")
       f.write("Line 2\n")
       f.write("Line 3\n")

**Example - Binary Mode:**

.. code-block:: python

   import pickle
   from polystore import atomic_write
   
   data = {"key": "value", "count": 42}
   
   with atomic_write("data.pkl", mode="wb") as f:
       pickle.dump(data, f)

**Parameters:**
  * ``file_path`` (str | Path) - Target file path
  * ``mode`` (str) - File mode ('w' for text, 'wb' for binary). Default: 'w'
  * ``ensure_directory`` (bool) - Create parent directory if missing. Default: True

**Raises:**
  * ``FileLockError`` - If atomic write fails

atomic_write_json()
~~~~~~~~~~~~~~~~~~~

.. autofunction:: polystore.atomic_write_json

Atomically write JSON data to a file.

**Features:**
  * Writes to temporary file with formatting
  * Atomically renames to target
  * Automatic indentation for readability
  * fsync() before rename for durability

**Example:**

.. code-block:: python

   from polystore import atomic_write_json
   
   config = {
       "setting1": "value1",
       "setting2": 42,
       "nested": {
           "key": "value"
       }
   }
   
   atomic_write_json(config, "config.json")

**Parameters:**
  * ``file_path`` (str | Path) - Target file path
  * ``data`` (dict) - Dictionary to write as JSON
  * ``indent`` (int) - JSON indentation level. Default: 2
  * ``ensure_directory`` (bool) - Create parent directory. Default: True

**Output Format:**

The JSON is pretty-printed with indentation:

.. code-block:: json

   {
     "setting1": "value1",
     "setting2": 42,
     "nested": {
       "key": "value"
     }
   }

File Locking
------------

Polystore uses cross-platform file locking via the ``portalocker`` library.

file_lock()
~~~~~~~~~~~

Context manager for file locking.

**Example:**

.. code-block:: python

   from polystore.atomic import file_lock
   from pathlib import Path
   
   lock_path = Path("data.lock")
   
   with file_lock(lock_path, timeout=30.0):
       # Critical section - only one process can enter
       with open("shared_data.txt", "r+") as f:
           data = f.read()
           # Process data
           f.seek(0)
           f.write(updated_data)

**Parameters:**
  * ``lock_path`` (str | Path) - Path to lock file
  * ``timeout`` (float) - Maximum time to wait for lock. Default: 30.0 seconds
  * ``poll_interval`` (float) - Polling interval. Default: 0.1 seconds

**Raises:**
  * ``FileLockTimeoutError`` - If lock cannot be acquired within timeout
  * ``FileLockError`` - For other locking errors

atomic_update_json()
~~~~~~~~~~~~~~~~~~~~

Atomically update a JSON file using read-modify-write with locking.

**Example:**

.. code-block:: python

   from polystore.atomic import atomic_update_json
   
   def increment_counter(data):
       """Update function receives current data, returns updated data."""
       if data is None:
           data = {"counter": 0}
       data["counter"] += 1
       return data
   
   # Multiple processes can safely call this
   atomic_update_json("counter.json", increment_counter)

**Parameters:**
  * ``file_path`` (str | Path) - JSON file path
  * ``update_func`` (Callable) - Function that receives current data and returns updated data
  * ``lock_timeout`` (float) - Lock timeout in seconds. Default: 30.0
  * ``default_data`` (dict | None) - Default data if file doesn't exist

**Process:**
  1. Acquire file lock
  2. Read current JSON (or use default)
  3. Call update function
  4. Write updated JSON atomically
  5. Release lock

Use Cases
---------

Configuration Files
~~~~~~~~~~~~~~~~~~~

Safely update configuration files:

.. code-block:: python

   from polystore import atomic_write_json
   
   # Read config
   import json
   with open("config.json") as f:
       config = json.load(f)
   
   # Modify
   config["last_run"] = datetime.now().isoformat()
   config["version"] = "1.2.0"
   
   # Write atomically
   atomic_write_json(config, "config.json")

Concurrent Access
~~~~~~~~~~~~~~~~~

Multiple processes accessing shared files:

.. code-block:: python

   from polystore.atomic import atomic_update_json
   
   def add_result(data):
       if data is None:
           data = {"results": []}
       data["results"].append({
           "timestamp": time.time(),
           "value": compute_value()
       })
       return data
   
   # Safe for multiple processes
   atomic_update_json("results.json", add_result)

Data Integrity
~~~~~~~~~~~~~~

Ensure complete writes:

.. code-block:: python

   from polystore import atomic_write
   import pandas as pd
   
   # Save DataFrame atomically
   with atomic_write("data.csv") as f:
       df.to_csv(f, index=False)
   
   # File is either complete or not written at all
   # No partial writes

Error Handling
--------------

Atomic operations handle errors gracefully:

.. code-block:: python

   from polystore import atomic_write
   
   try:
       with atomic_write("output.txt") as f:
           f.write("data\n")
           raise ValueError("Something went wrong")
   except ValueError:
       pass
   
   # Temporary file is cleaned up
   # Target file is unchanged

Platform Support
----------------

Atomic operations work on all platforms:

* **Unix/Linux**: Uses ``os.rename()`` which is atomic
* **Windows**: Uses ``os.replace()`` which atomically replaces files
* **File Locking**: ``portalocker`` provides cross-platform locking

Performance
-----------

Atomic operations have minimal overhead:

* **Write Performance**: ~5-10% slower than direct writes (due to temp file + rename)
* **Lock Overhead**: Minimal when uncontested (<1ms)
* **Lock Contention**: Configurable timeout and polling

Best Practices
--------------

1. **Use for Critical Data**: Apply atomic writes to configuration, state, and results
2. **Reasonable Timeouts**: Set appropriate lock timeouts based on operation duration
3. **Small Files**: Best for files under 100MB (larger files increase rename time)
4. **Error Handling**: Always handle ``FileLockError`` and ``FileLockTimeoutError``
5. **Cleanup**: Context managers handle cleanup automatically

See Also
--------

* :doc:`filemanager` - High-level API with atomic operations built-in
* :doc:`backends` - Backend implementations use atomic operations
