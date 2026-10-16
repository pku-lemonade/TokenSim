use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyLong, PyTuple};

// Keep SimPy's queue, event objects, callbacks and priorities unchanged.
// Only the dispatch loop moves across the Python/Rust boundary.
#[pyclass]
pub struct EventLoop {
    pop: PyObject,
    event: PyObject,
    stop: PyObject,
    empty: PyObject,
    stop_callback: PyObject,
    float: PyObject,
    subtract: PyObject,
}

impl EventLoop {
    fn step(&self, py: Python<'_>, env: &PyAny) -> PyResult<()> {
        let row = match self.pop.call1(py, (env.getattr("_queue")?,)) {
            Ok(row) => row,
            Err(error) if error.is_instance_of::<PyIndexError>(py) => {
                return Err(PyErr::from_value(self.empty.call0(py)?.as_ref(py)));
            }
            Err(error) => return Err(error),
        };
        let row = row.as_ref(py).downcast::<PyTuple>()?;
        env.setattr("_now", row.get_item(0)?)?;
        let event = row.get_item(3)?;
        let callbacks = event.getattr("callbacks")?.downcast::<PyList>()?;
        event.setattr("callbacks", py.None())?;
        let mut index = 0;
        while index < callbacks.len() {
            let callback = callbacks.get_item(index)?;
            index += 1;
            if let Err(error) = callback.call1((event,)) {
                if error.is_instance(py, self.stop.as_ref(py)) {
                    let index: usize = callbacks.call_method1("index", (callback,))?.extract()?;
                    event.setattr("callbacks", callbacks.get_slice(index + 1, callbacks.len()))?;
                    env.call_method1("schedule", (event, -1))?;
                }
                return Err(error);
            }
        }
        if !event.getattr("_ok")?.is_true()? && !event.hasattr("_defused")? {
            let original = event.getattr("_value")?;
            let error = original
                .get_type()
                .call(original.getattr("args")?.downcast::<PyTuple>()?, None)?;
            error.setattr("__cause__", original)?;
            return Err(PyErr::from_value(error));
        }
        Ok(())
    }
}

#[pymethods]
impl EventLoop {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        let core = py.import("simpy.core")?;
        let stop = core.getattr("StopSimulation")?;
        Ok(Self {
            pop: py.import("heapq")?.getattr("heappop")?.into_py(py),
            event: core.getattr("Event")?.into_py(py),
            stop: stop.into_py(py),
            empty: core.getattr("EmptySchedule")?.into_py(py),
            stop_callback: stop.getattr("callback")?.into_py(py),
            float: py.import("builtins")?.getattr("float")?.into_py(py),
            subtract: py.import("operator")?.getattr("sub")?.into_py(py),
        })
    }

    #[pyo3(signature = (env, until=None))]
    fn run(&self, py: Python<'_>, env: &PyAny, until: Option<&PyAny>) -> PyResult<PyObject> {
        let target = match until {
            None => None,
            Some(until) if !until.is_instance(self.event.as_ref(py))? => {
                let at = if until.is_instance_of::<PyLong>() {
                    until.into_py(py)
                } else {
                    self.float.call1(py, (until,))?
                };
                let now = env.getattr("now")?;
                if at.as_ref(py).le(now)? {
                    return Err(PyValueError::new_err(format!(
                        "until ({}) must be greater than the current simulation time",
                        at.as_ref(py).str()?
                    )));
                }
                let event = self.event.call1(py, (env,))?;
                event.as_ref(py).setattr("_ok", true)?;
                event.as_ref(py).setattr("_value", py.None())?;
                let delay = self.subtract.call1(py, (&at, now))?;
                env.call_method1("schedule", (&event, 0, delay))?;
                Some(event)
            }
            Some(until) if until.getattr("callbacks")?.is_none() => {
                return Ok(until.getattr("value")?.into_py(py));
            }
            Some(until) => Some(until.into_py(py)),
        };
        if let Some(target) = &target {
            target
                .as_ref(py)
                .getattr("callbacks")?
                .call_method1("append", (&self.stop_callback,))?;
        }
        let environment = env.into_py(py);
        loop {
            // Release every event's temporary Python references at this boundary.
            // Only owned Py<T> handles and PyErr values escape this safe scope.
            let result = py.with_pool(|scope| self.step(scope, environment.as_ref(scope)));
            match result {
                Ok(()) => (),
                Err(error) if error.is_instance(py, self.stop.as_ref(py)) => {
                    return Ok(error.value(py).getattr("args")?.get_item(0)?.into_py(py));
                }
                Err(error) if error.is_instance(py, self.empty.as_ref(py)) => {
                    if let Some(target) = &target {
                        return Err(PyRuntimeError::new_err(format!(
                            "No scheduled events left but \"until\" event was not triggered: {}",
                            target.as_ref(py).str()?
                        )));
                    }
                    return Ok(py.None());
                }
                Err(error) => return Err(error),
            }
        }
    }
}
