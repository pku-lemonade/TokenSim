use pyo3::exceptions::{PyRecursionError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyList, PyLong, PyString, PyTuple};

#[pyclass]
pub struct Fingerprints {
    #[pyo3(get)]
    values: Py<PyDict>,
    #[pyo3(get)]
    capacity: usize,
    layouts: Py<PyDict>,
    layout: PyObject,
    mapping: PyObject,
    frozen: PyObject,
}

fn raw_scalar(value: &PyAny) -> bool {
    let py = value.py();
    value.is_none()
        || value.get_type().is(py.get_type::<PyString>())
        || value.get_type().is(py.get_type::<PyLong>())
}

impl Fingerprints {
    fn visit(&self, value: &PyAny, depth: usize) -> PyResult<(PyObject, bool)> {
        let py = value.py();
        if raw_scalar(value) {
            return Ok((value.into_py(py), true));
        }
        let kind = value.get_type();
        if kind.is(py.get_type::<PyBool>()) || kind.is(py.get_type::<PyFloat>()) {
            return Ok(((kind, value).into_py(py), true));
        }
        if depth > 256 {
            return Err(PyRecursionError::new_err(
                "structural key is too deeply nested",
            ));
        }
        let memo = self.values.as_ref(py);
        let position = value.as_ptr() as usize;
        if let Some(previous) = memo.get_item(position)? {
            let pair = previous.downcast::<PyTuple>()?;
            if pair.get_item(0)?.is(value) {
                return Ok((pair.get_item(1)?.into_py(py), true));
            }
        }
        let layouts = self.layouts.as_ref(py);
        let layout = match layouts.get_item(kind)? {
            Some(layout) => layout,
            None => {
                let layout = self.layout.call1(py, (kind,))?;
                if layouts.len() >= 128 {
                    let oldest = layouts.iter().next().unwrap().0.into_py(py);
                    layouts.del_item(oldest)?;
                }
                layouts.set_item(kind, &layout)?;
                layout.into_ref(py)
            }
        };
        let mut children = Vec::new();
        let mut safe;
        if !layout.is_none() {
            let layout = layout.downcast::<PyTuple>()?;
            safe = layout.get_item(1)?.is_true()?;
            for name in layout.get_item(0)?.iter()? {
                let name = name?;
                let (child, immutable) =
                    self.visit(value.getattr(name.downcast::<PyString>()?)?, depth + 1)?;
                children.push((name, child).into_py(py));
                safe &= immutable;
            }
        } else if value.is_instance(self.mapping.as_ref(py))? {
            safe = value.is_instance(self.frozen.as_ref(py))?;
            let items = value.call_method0("items")?;
            for pair in items.iter()? {
                let pair = pair?;
                let (child, immutable) = self.visit(pair.get_item(1)?, depth + 1)?;
                children.push((pair.get_item(0)?, child).into_py(py));
                safe &= immutable;
            }
            let sorted = PyList::new(py, &children);
            sorted.sort()?;
            children = sorted.iter().map(|item| item.into_py(py)).collect();
        } else if value.is_instance_of::<PyTuple>() || value.is_instance_of::<PyList>() {
            if kind.is(py.get_type::<PyTuple>())
                && value.downcast::<PyTuple>()?.iter().all(raw_scalar)
            {
                return Ok((value.into_py(py), true));
            }
            safe = value.is_instance_of::<PyTuple>();
            for item in value.iter()? {
                let (child, immutable) = self.visit(item?, depth + 1)?;
                children.push(child);
                safe &= immutable;
            }
        } else {
            return Ok((value.into_py(py), false));
        }
        let key = PyTuple::new(py, children).into_py(py);
        if safe {
            if memo.len() >= self.capacity {
                let oldest = memo.iter().next().unwrap().0.into_py(py);
                memo.del_item(oldest)?;
            }
            memo.set_item(position, (value, &key))?;
        }
        Ok((key, safe))
    }
}

#[pymethods]
impl Fingerprints {
    #[new]
    #[pyo3(signature = (layout, mapping, frozen, capacity=4096))]
    fn new(
        py: Python<'_>,
        layout: PyObject,
        mapping: PyObject,
        frozen: PyObject,
        capacity: usize,
    ) -> PyResult<Self> {
        if capacity == 0 {
            return Err(PyValueError::new_err("capacity must be positive"));
        }
        Ok(Self {
            values: PyDict::new(py).into(),
            layouts: PyDict::new(py).into(),
            capacity,
            layout,
            mapping,
            frozen,
        })
    }

    fn __call__(&self, value: &PyAny) -> PyResult<PyObject> {
        Ok(self.visit(value, 0)?.0)
    }

    fn normalize_many(&self, values: &PyTuple) -> PyResult<PyObject> {
        let children = values
            .iter()
            .map(|value| self.visit(value, 0).map(|pair| pair.0));
        Ok(PyTuple::new(values.py(), children.collect::<PyResult<Vec<_>>>()?).into())
    }
}
