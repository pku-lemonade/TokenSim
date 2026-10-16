use pyo3::exceptions::{PyRecursionError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::collections::HashMap;

enum Recipe {
    Static(PyObject),
    Prefix(String),
    Token(usize, String),
    Edge(usize),
    Tuple(Vec<usize>),
    Mapping(Vec<(PyObject, usize)>),
    Record(PyObject, Py<PyDict>, Vec<(PyObject, usize)>),
}

#[pyclass]
pub struct Template {
    recipes: Vec<Recipe>,
    nodes: Vec<usize>,
    object_new: PyObject,
    frozen: PyObject,
}

fn compile(
    value: &PyAny,
    recipes: &mut Vec<Recipe>,
    memo: &mut HashMap<usize, usize>,
    depth: usize,
) -> PyResult<usize> {
    if depth > 256 {
        return Err(PyRecursionError::new_err(
            "DAG template is too deeply nested",
        ));
    }
    let position = value.as_ptr() as usize;
    if let Some(index) = memo.get(&position) {
        return Ok(*index);
    }
    let py = value.py();
    let kind: &str = value.getattr("kind")?.extract()?;
    let data = value.getattr("value")?;
    let recipe = match kind {
        "static" => Recipe::Static(data.into_py(py)),
        "prefix" => Recipe::Prefix(data.extract()?),
        "token" => Recipe::Token(data.get_item(0)?.extract()?, data.get_item(1)?.extract()?),
        "edge" => Recipe::Edge(data.extract()?),
        "tuple" => {
            let mut children = Vec::new();
            for child in value.getattr("children")?.iter()? {
                children.push(compile(child?, recipes, memo, depth + 1)?);
            }
            Recipe::Tuple(children)
        }
        "mapping" | "record" => {
            let mut children = Vec::new();
            for pair in value.getattr("children")?.iter()? {
                let pair = pair?;
                children.push((
                    pair.get_item(0)?.into_py(py),
                    compile(pair.get_item(1)?, recipes, memo, depth + 1)?,
                ));
            }
            if kind == "mapping" {
                Recipe::Mapping(children)
            } else {
                Recipe::Record(
                    data.get_item(0)?.into_py(py),
                    data.get_item(1)?.downcast::<PyDict>()?.into(),
                    children,
                )
            }
        }
        _ => return Err(PyValueError::new_err("unknown DAG template recipe")),
    };
    let index = recipes.len();
    recipes.push(recipe);
    memo.insert(position, index);
    Ok(index)
}

#[pymethods]
impl Template {
    #[new]
    fn new(nodes: &PyTuple, frozen: PyObject) -> PyResult<Self> {
        let py = nodes.py();
        let mut recipes = Vec::new();
        let mut memo = HashMap::new();
        let mut roots = Vec::new();
        for node in nodes.iter() {
            roots.push(compile(node, &mut recipes, &mut memo, 0)?);
        }
        Ok(Self {
            recipes,
            nodes: roots,
            object_new: py
                .import("builtins")?
                .getattr("object")?
                .getattr("__new__")?
                .into_py(py),
            frozen,
        })
    }

    fn bind(
        &self,
        py: Python<'_>,
        prefix: &str,
        tokens: &PyTuple,
        edges: &PyTuple,
    ) -> PyResult<PyObject> {
        let mut bound: Vec<PyObject> = Vec::with_capacity(self.recipes.len());
        for recipe in &self.recipes {
            let value = match recipe {
                Recipe::Static(value) => value.clone_ref(py),
                Recipe::Prefix(suffix) => format!("{prefix}{suffix}").into_py(py),
                Recipe::Token(index, suffix) => {
                    let pair = tokens.get_item(*index)?;
                    let request = pair.get_item(0)?.str()?;
                    let token = pair.get_item(1)?.str()?;
                    format!("request:{request}:token:{token}{suffix}").into_py(py)
                }
                Recipe::Edge(index) => edges.get_item(*index)?.into_py(py),
                Recipe::Tuple(children) => {
                    PyTuple::new(py, children.iter().map(|index| &bound[*index])).into_py(py)
                }
                Recipe::Mapping(children) => {
                    let result = PyDict::new(py);
                    for (key, index) in children {
                        result.set_item(key, &bound[*index])?;
                    }
                    self.frozen.call1(py, (result,))?
                }
                Recipe::Record(kind, static_fields, children) => {
                    let result = self.object_new.call1(py, (kind,))?;
                    let fields = result
                        .as_ref(py)
                        .getattr("__dict__")?
                        .downcast::<PyDict>()?;
                    for (key, value) in static_fields.as_ref(py) {
                        fields.set_item(key, value)?;
                    }
                    for (key, index) in children {
                        fields.set_item(key, &bound[*index])?;
                    }
                    result
                }
            };
            bound.push(value);
        }
        Ok(PyTuple::new(py, self.nodes.iter().map(|index| &bound[*index])).into_py(py))
    }
}
