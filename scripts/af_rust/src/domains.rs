use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyLong, PyString, PyTuple};
use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet};
use std::sync::Arc;

#[derive(Eq, PartialEq, Ord, PartialOrd)]
enum IdentityPart {
    Integer(i64),
    Text(String),
}

struct Priority {
    time: f64,
    identity: Vec<IdentityPart>,
}

impl PartialEq for Priority {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.identity == other.identity
    }
}
impl Eq for Priority {}
impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        // submit rejects NaN; ordinary comparison preserves Python's signed zero ties.
        self.time
            .partial_cmp(&other.time)
            .unwrap()
            .then_with(|| self.identity.cmp(&other.identity))
    }
}

#[derive(Eq, PartialEq, Ord, PartialOrd)]
struct Entry {
    priority: Arc<Priority>,
    serial: usize,
}

struct Request {
    value: PyObject,
    priority: Arc<Priority>,
    keys: Vec<usize>,
    acquired: bool,
}

#[derive(Default)]
struct Resource {
    owner: Option<usize>,
    queue: BinaryHeap<Reverse<Entry>>,
}

// This object owns the complete queue state; arbitration never calls Python.
// SimPy events and modeling rules remain in the Python adapter.
#[pyclass]
#[derive(Default)]
pub struct DomainQueues {
    names: HashMap<String, usize>,
    resources: Vec<Resource>,
    requests: HashMap<usize, Request>,
    priorities: BTreeSet<Arc<Priority>>,
    dirty: HashSet<usize>,
    serial: usize,
}

impl DomainQueues {
    fn front(&mut self, key: usize) -> Option<usize> {
        let resource = &mut self.resources[key];
        while let Some(Reverse(entry)) = resource.queue.peek() {
            if self.requests.contains_key(&entry.serial) {
                return Some(entry.serial);
            }
            resource.queue.pop();
        }
        None
    }
}

#[pymethods]
impl DomainQueues {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn submit(
        &mut self,
        value: PyObject,
        keys: Vec<String>,
        time: &PyAny,
        identity: &PyTuple,
    ) -> PyResult<usize> {
        let at: f64 = time.extract()?;
        let large_integer = time.is_instance_of::<PyLong>()
            && time.extract::<i64>()?.unsigned_abs() > 9007199254740992;
        if !at.is_finite() || large_integer {
            return Err(PyValueError::new_err(
                "Rust domain probe requires a finite exactly representable timestamp",
            ));
        }
        let mut parts = Vec::with_capacity(identity.len());
        for part in identity {
            parts.push(if part.is_instance_of::<PyLong>() {
                IdentityPart::Integer(part.extract()?)
            } else if part.is_instance_of::<PyString>() {
                IdentityPart::Text(part.extract()?)
            } else {
                return Err(PyValueError::new_err("unsupported domain identity"));
            });
        }
        let priority = Arc::new(Priority {
            time: at,
            identity: parts,
        });
        if self.priorities.contains(&priority) {
            return Err(PyValueError::new_err(
                "concurrent domain requests need unique stable logical identities",
            ));
        }
        self.serial += 1;
        let serial = self.serial;
        let mut resolved = Vec::with_capacity(keys.len());
        for key in keys {
            let index = *self.names.entry(key).or_insert_with(|| {
                self.resources.push(Resource::default());
                self.resources.len() - 1
            });
            self.resources[index].queue.push(Reverse(Entry {
                priority: priority.clone(),
                serial,
            }));
            self.dirty.insert(index);
            resolved.push(index);
        }
        self.priorities.insert(priority.clone());
        self.requests.insert(
            serial,
            Request {
                value,
                priority,
                keys: resolved,
                acquired: false,
            },
        );
        Ok(serial)
    }

    fn arbitrate(&mut self, py: Python<'_>) -> Vec<PyObject> {
        let changed = std::mem::take(&mut self.dirty);
        let mut candidates = BTreeSet::new();
        for key in changed {
            if let Some(serial) = self.front(key) {
                candidates.insert(Entry {
                    priority: self.requests[&serial].priority.clone(),
                    serial,
                });
            }
        }
        let mut granted = Vec::new();
        for entry in candidates {
            let serial = entry.serial;
            let keys = self.requests[&serial].keys.clone();
            if keys
                .iter()
                .any(|&key| self.resources[key].owner.is_some() || self.front(key) != Some(serial))
            {
                continue;
            }
            let request = self.requests.get_mut(&serial).unwrap();
            request.acquired = true;
            self.priorities.remove(&request.priority);
            granted.push(request.value.clone_ref(py));
            for key in keys {
                self.resources[key].owner = Some(serial);
                self.resources[key].queue.pop();
            }
        }
        granted
    }

    fn release(&mut self, serial: usize) {
        if let Some(request) = self.requests.remove(&serial) {
            if !request.acquired {
                self.priorities.remove(&request.priority);
            }
            for key in request.keys {
                if self.resources[key].owner == Some(serial) {
                    self.resources[key].owner = None;
                }
                self.dirty.insert(key);
            }
        }
    }

    fn peek(&mut self, py: Python<'_>, name: &str) -> Option<PyObject> {
        let key = *self.names.get(name)?;
        self.front(key)
            .map(|serial| self.requests[&serial].value.clone_ref(py))
    }

    #[getter]
    fn waiters(&self, py: Python<'_>) -> Vec<PyObject> {
        let mut waiting: Vec<_> = self.requests.values().filter(|r| !r.acquired).collect();
        waiting.sort_by(|a, b| a.priority.cmp(&b.priority));
        waiting.iter().map(|r| r.value.clone_ref(py)).collect()
    }

    #[getter]
    fn owners(&self, py: Python<'_>) -> PyResult<PyObject> {
        let owners = PyDict::new(py);
        for (name, &index) in &self.names {
            if let Some(serial) = self.resources[index].owner {
                owners.set_item(name, &self.requests[&serial].value)?;
            }
        }
        Ok(owners.into_py(py))
    }

    #[getter]
    fn storage(&self) -> (usize, usize, usize) {
        (
            self.requests.len(),
            self.resources.iter().map(|r| r.queue.len()).sum(),
            self.resources.len(),
        )
    }
}
