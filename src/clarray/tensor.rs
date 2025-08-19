use std::rc::Rc;

use num_traits::Float;
use ocl::{Buffer, OclPrm};

use crate::clarray::env::{DTypeSpec, Env, env};

pub trait TensorType: DTypeSpec + OclPrm + Float {}
impl TensorType for f32 {}
impl TensorType for f64 {}

#[derive(Clone)]
pub struct Tensor<T>
where
  T: TensorType,
{
  pub buffer: Buffer<T>,
  pub shape: Vec<usize>,
  pub strides: Vec<usize>,
  pub env: &'static Env,
}

impl<T> Tensor<T>
where
  T: TensorType,
{
  pub fn zeros(shape: Vec<usize>) -> Self {
    let strides = shape
      .iter()
      .scan(1, |acc, &x| {
        let old = *acc;
        *acc *= x;
        Some(old)
      })
      .collect::<Vec<_>>();

    let size: usize = shape.iter().product();
    let buffer = Buffer::<T>::builder()
      .queue(env().proque().queue().clone())
      .len(size)
      .fill_val(T::zero())
      .build()
      .expect("Failed to create buffer");

    Self {
      buffer,
      shape,
      strides,
      env: env(),
    }
  }

  pub fn from_array(shape: Vec<usize>, array: Vec<T>) -> Self {
    let strides = shape
      .iter()
      .scan(1, |acc, &x| {
        let old = *acc;
        *acc *= x;
        Some(old)
      })
      .collect::<Vec<_>>();

    let size: usize = shape.iter().product();
    assert_eq!(array.len(), size, "Array length does not match shape");

    let buffer = Buffer::<T>::builder()
      .queue(env().proque().queue().clone())
      .len(size)
      .copy_host_slice(&array)
      .build()
      .expect("Failed to create buffer");

    Self {
      buffer,
      shape,
      strides,
      env: env(),
    }
  }

  pub fn scalar(value: T) -> Self {
    let shape = vec![];
    let strides = vec![0];

    let buffer = Buffer::<T>::builder()
      .queue(env().proque().queue().clone())
      .len(1)
      .fill_val(value)
      .build()
      .expect("Failed to create buffer");

    Self {
      buffer,
      shape,
      strides,
      env: env(),
    }
  }

  pub fn len(&self) -> usize {
    if self.shape.is_empty() {
      return 1;
    }
    self.shape.iter().product()
  }

  pub fn dim(&self) -> usize {
    self.shape.len()
  }
}
