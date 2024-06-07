# Package Structure

[TOC]

## Overview

### Terminology

#### Python Module

A Python module is a file containing Python definitions and statements. See
[modules](https://docs.python.org/3/tutorial/modules.html#modules) for more
information.

#### Python Package

Python packages are a way of structuring Python modules. See
[packages](https://docs.python.org/3/tutorial/modules.html#packages) for more
information.

#### Public TFF API

The TFF API that is exposed by the
[TFF API documentation](https://www.tensorflow.org/federated/api_docs/python/tff);
this documentation is generated with
[TensorFlow Docs](https://github.com/tensorflow/docs) using the logic defined by
the
[explicit_package_contents_filter](https://github.com/tensorflow/docs/blob/master/tools/tensorflow_docs/api_generator/public_api.py#L214)

#### Private TFF API

The TFF API that that is *not* exposed in the TFF
[TFF API documentation](https://www.tensorflow.org/federated/api_docs/python/tff).

#### TFF Python package

The Python [package](https://pypi.org/project/tensorflow-federated/) distributed
on https://pypi.org.

Please be aware, the Python package contains both
[public TFF API](#public-tff-api) and [private TFF API](#private-tff-api) and it
is not obvious *by inspecting the package* which API is intended to be public
and which is intended to be private, for example:

```python
import tensorflow_federated as tff

tff.Computation  # Public TFF API
tff.proto.v0.computation_pb2.Computation  # Private TFF API
```

Therefore, it is useful to keep the
[TFF API documentation](https://www.tensorflow.org/federated/api_docs/python/tff)
in mind when using TFF.

### Diagrams

#### Overview

```dot
<!--#include file="package_structure_overview.dot"-->
```

#### Simulation

```dot
<!--#include file="package_structure_simulation.dot"-->
```

#### Learning

```dot
<!--#include file="package_structure_learning.dot"-->
```

#### Analytics

```dot
<!--#include file="package_structure_analytics.dot"-->
```

#### Core

```dot
<!--#include file="package_structure_core.dot"-->
```
