#![allow(non_local_definitions)]

mod domains;
mod events;
mod fingerprints;
mod templates;

use pyo3::prelude::*;

#[pymodule]
fn tokensim_rust_probe(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<fingerprints::Fingerprints>()?;
    module.add_class::<events::EventLoop>()?;
    module.add_class::<domains::DomainQueues>()?;
    module.add_class::<templates::Template>()?;
    Ok(())
}
