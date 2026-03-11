.. _setup-xpu:

Intel XPU Setup Guide
=====================

This guide explains how to build, install, and test **Nunchaku** on Intel XPU
(Data Center GPU Max Series, Arc Series, and integrated GPUs that expose
``torch.xpu``).

.. contents:: Table of Contents
   :depth: 2
   :local:


Prerequisites
-------------

Hardware
^^^^^^^^

Any Intel GPU that is supported by the `Intel Extension for PyTorch (IPEX)
<https://intel.github.io/intel-extension-for-pytorch/>`_ XPU backend:

- Intel Data Center GPU Max Series (e.g. Max 1550, Max 1100)
- Intel Arc A-Series (e.g. Arc A770, A750)
- Intel Core Ultra Series integrated GPUs

Software
^^^^^^^^

+-------------------+-------------------------------+
| Dependency        | Minimum Version               |
+===================+===============================+
| OS                | Linux (Ubuntu 22.04+)         |
+-------------------+-------------------------------+
| Intel GPU Driver  | Latest stable                 |
+-------------------+-------------------------------+
| Intel oneAPI      | 2024.0+ (for SYCL/DPC++)      |
+-------------------+-------------------------------+
| Python            | 3.10+                         |
+-------------------+-------------------------------+
| PyTorch           | 2.7+ (with XPU support)       |
+-------------------+-------------------------------+


Step 1: Install the Intel GPU Driver and oneAPI Toolkit
-------------------------------------------------------

Install the Intel GPU driver and oneAPI Base Toolkit by following the official
Intel documentation:

- `Intel GPU Driver Installation <https://dgpu-docs.intel.com/driver/installation.html>`_
- `oneAPI Base Toolkit <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>`_

After installation, source the oneAPI environment in every new shell session:

.. code-block:: shell

    source /opt/intel/oneapi/setvars.sh

.. tip::

   Add the above line to your ``~/.bashrc`` so that it runs automatically.


Step 2: Set Up a Python Environment
------------------------------------

.. code-block:: shell

    conda create -n nunchaku-xpu python=3.11
    conda activate nunchaku-xpu


Step 3: Install PyTorch with XPU Support
-----------------------------------------

Install a PyTorch build that includes the ``torch.xpu`` backend.
The recommended approach is to use the official PyTorch XPU wheels:

.. code-block:: shell

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

Verify that ``torch.xpu`` is available:

.. code-block:: shell

    python -c "import torch; print('XPU available:', torch.xpu.is_available())"

You should see:

.. code-block:: text

    XPU available: True

.. important::

   If ``torch.xpu.is_available()`` returns ``False``, check that:

   1. The Intel GPU driver is installed and loaded (``sycl-ls`` should list
      your GPU).
   2. You are using a PyTorch build that includes XPU support.
   3. ``source /opt/intel/oneapi/setvars.sh`` has been run in the current
      shell.


Step 4: Install Nunchaku
-------------------------

Option A: Python-only Install (Recommended for XPU)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The native CUDA C++ extension (``nunchaku._C``) cannot be compiled on a
machine without an NVIDIA GPU and CUDA toolkit.  On Intel XPU, Nunchaku
automatically falls back to **PyTorch fallback** or **Triton** backends that
do not require the extension.

.. code-block:: shell

    git clone --recurse-submodules https://github.com/nunchaku-tech/nunchaku.git
    cd nunchaku
    pip install -e ".[dev]" --no-build-isolation || pip install -e "."

.. note::

   The ``pip install`` may print warnings about the missing CUDA extension—
   this is expected.  All operations that are listed as ✅ in the
   :doc:`XPU Support Checklist </developer/xpu_support_checklist>` work
   without the extension.

Option B: Full Build from Source (Requires CUDA Toolkit)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you also have an NVIDIA GPU and a CUDA toolkit available, follow the
standard :ref:`build-from-source` instructions.  The resulting build will
include the native CUDA extension **and** the XPU fallback/Triton paths.


Step 5: Install Triton (Optional, Recommended)
------------------------------------------------

Triton provides higher performance than the pure-PyTorch fallback on XPU.
Intel maintains an XPU-compatible fork of Triton:

.. code-block:: shell

    pip install triton

Verify:

.. code-block:: python

    from nunchaku.ops.triton_kernels import is_triton_available
    print("Triton available:", is_triton_available())


Step 6: Configure the Backend
------------------------------

Nunchaku selects the best available backend automatically:

1. **CUDA native** (``nunchaku._C``) — used on NVIDIA GPUs when the
   extension is compiled.
2. **Triton** — portable high-performance kernels; works on both CUDA and
   Intel XPU when Triton is installed.
3. **PyTorch fallback** — pure-PyTorch implementation; works on any
   PyTorch-supported device (CPU, CUDA, XPU).

On Intel XPU, the CUDA native extension is not available, so Nunchaku will
use **Triton** (if installed) or the **PyTorch fallback** (always available).

You can force a specific backend via the ``NUNCHAKU_BACKEND`` environment
variable:

.. code-block:: shell

    # Use Triton kernels explicitly
    NUNCHAKU_BACKEND=triton python my_script.py

    # Use pure-PyTorch fallback explicitly
    NUNCHAKU_BACKEND=torch python my_script.py


Running Tests
-------------

Nunchaku ships with two XPU-focused test suites that validate device-agnostic
operations on **any** machine (no GPU required):

.. code-block:: shell

    # Run device abstraction tests
    python -m pytest tests/test_device_abstraction.py -v

    # Run comprehensive XPU ops tests (helper ops, GEMM, quantize, AWQ, dispatch, linear layers)
    python -m pytest tests/test_xpu_ops.py -v

    # Run both together
    python -m pytest tests/test_device_abstraction.py tests/test_xpu_ops.py -v

These tests use small tensors and run on CPU, so they complete quickly (< 10 s)
and verify that:

- INT4 unpacking/dequantization produces correct results
- W4A4 GEMM and activation quantization fallbacks are numerically correct
- AWQ W4A16 GEMV/GEMM fallbacks work correctly
- Backend dispatch selects the correct backend
- ``NUNCHAKU_BACKEND`` environment variable override works
- ``SVDQW4A4Linear`` and ``AWQW4A16Linear`` modules function end-to-end
- Device abstraction APIs (streams, events, synchronization) are functional

If Triton is installed, two additional Triton-specific tests run automatically;
otherwise, they are skipped.

.. tip::

   On an Intel XPU machine you can additionally run the tests with
   ``torch.xpu`` as the device to validate the full device path:

   .. code-block:: shell

       python -c "
       import torch
       assert torch.xpu.is_available(), 'XPU not available'
       x = torch.randn(4, 64, device='xpu', dtype=torch.bfloat16)
       print('XPU tensor created:', x.shape, x.device)
       "


Supported Operations
--------------------

See :doc:`/developer/xpu_support_checklist` for the full list of APIs and
their XPU support status.

Quick summary:

.. list-table::
   :header-rows: 1
   :widths: 50 20 30

   * - Category
     - Status
     - Notes
   * - Core Quantized Ops (W4A4 GEMM, quantization, AWQ)
     - ✅
     - PyTorch fallback + Triton
   * - Fused Ops (GELU MLP, QKV norm rotary, SiLU)
     - ✅
     - Via GEMM dispatch
   * - Quantized Linear Layers (SVDQW4A4, AWQW4A16)
     - ✅
     - Full forward pass
   * - Helper Ops (INT4 pack/unpack, pad, ceil_divide)
     - ✅
     - Pure PyTorch
   * - Device Abstraction (streams, events, memory)
     - ✅
     - XPU-native
   * - Model Utilities (precision, compatibility checks)
     - ✅
     - Device-agnostic
   * - Triton Kernels (W4A4 GEMM, quantization)
     - ✅
     - When Triton installed
   * - Fused Attention (``attention_fp16``)
     - ❌
     - Planned


Troubleshooting
---------------

``torch.xpu.is_available()`` returns ``False``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Check the driver**: Run ``sycl-ls`` and verify your GPU appears.
2. **Source oneAPI**: ``source /opt/intel/oneapi/setvars.sh``
3. **Check PyTorch build**: Ensure you installed the ``xpu`` variant of
   PyTorch (e.g. ``--index-url https://download.pytorch.org/whl/xpu``).

``nunchaku._C`` import fails
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is expected when the CUDA extension was not compiled. On XPU, Nunchaku
uses the PyTorch fallback or Triton backend automatically. No action needed.

Tests fail with ``ModuleNotFoundError``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Make sure all dependencies are installed:

.. code-block:: shell

    pip install -e ".[dev]"

Slow performance
^^^^^^^^^^^^^^^^

If Triton is not installed, Nunchaku uses the pure-PyTorch fallback which is
significantly slower.  Install Triton for better performance:

.. code-block:: shell

    pip install triton

And set the backend explicitly:

.. code-block:: shell

    NUNCHAKU_BACKEND=triton python my_script.py


End-to-End Example
------------------

Below is a minimal example that runs a quantized linear layer on the
available device (XPU if available, otherwise CPU):

.. code-block:: python

    import os
    import torch
    from nunchaku.models.linear import SVDQW4A4Linear

    # Use PyTorch fallback (works on any device)
    os.environ["NUNCHAKU_BACKEND"] = "torch"

    device = "xpu" if torch.xpu.is_available() else "cpu"
    dtype = torch.bfloat16

    # Create a small quantized linear layer
    linear = SVDQW4A4Linear(
        in_features=64,
        out_features=32,
        rank=8,
        bias=True,
        precision="int4",
        torch_dtype=dtype,
        device=device,
    )

    # Initialize with random data
    with torch.no_grad():
        linear.qweight.copy_(torch.randint(-128, 128, linear.qweight.shape, dtype=torch.int8))
        linear.wscales.normal_()
        linear.smooth_factor.fill_(1.0)
        linear.proj_down.normal_()
        linear.proj_up.normal_()

    # Run forward pass
    x = torch.randn(1, 2, 64, dtype=dtype, device=device)
    output = linear(x)
    print(f"Input:  {x.shape} on {x.device}")
    print(f"Output: {output.shape} on {output.device}")
