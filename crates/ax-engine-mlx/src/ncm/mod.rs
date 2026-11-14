//! Experimental native unconditional TinyStories inference. No normal model admission.
pub mod artifact;
#[doc(hidden)]
pub mod codec;
#[doc(hidden)]
pub mod generator;
mod runner;
pub use runner::NcmRunner;
