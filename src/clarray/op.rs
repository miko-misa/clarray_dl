use core::str;

use num_traits::Float;
use ocl::{Buffer, Error, Event, EventList, OclPrm};

use crate::clarray::env::env;
use crate::clarray::tensor::{Tensor, TensorType};

pub fn array_2_buffer(array: Vec<usize>) -> Buffer<u32> {
  let array_i32: Vec<u32> = array.into_iter().map(|x| x as u32).collect();
  Buffer::<u32>::builder()
    .queue(env().proque().queue().clone())
    .len(array_i32.len())
    .copy_host_slice(&array_i32)
    .build()
    .expect("Failed to create buffer")
}

fn events_to_list(events: Vec<&Event>) -> EventList {
  let mut el = EventList::new();
  for e in events {
    el.push(e.clone());
  }
  el
}

fn enq_kernel(kernel: &ocl::Kernel, events: Vec<&Event>) -> ocl::Result<Event> {
  let mut event = Event::empty();
  let el = events_to_list(events);
  unsafe {
    kernel.cmd().ewait(&el).enew(&mut event).enq()?;
  }
  Ok(event)
}

pub trait NodeOp<T>
where
  T: TensorType,
{
  fn name(&self) -> &str;
  fn forward(
    &self,
    parents: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Tensor<T>, Option<Event>), Error>;
  fn backward(
    &self,
    childrens: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Vec<&Tensor<T>>, Event), Error>;
}

pub struct Input {
  pub(crate) name: String,
}

impl<T> NodeOp<T> for Input
where
  T: TensorType,
{
  fn name(&self) -> &str {
    &self.name
  }

  fn forward(
    &self,
    parents: Vec<&Tensor<T>>,
    _: Vec<&Event>,
  ) -> Result<(Tensor<T>, Option<Event>), Error> {
    Ok((parents[0].clone(), None))
  }

  fn backward(
    &self,
    _: Vec<&Tensor<T>>,
    _: Vec<&Event>,
  ) -> Result<(Vec<&Tensor<T>>, Event), ocl::Error> {
    let empty_tensor: Vec<&Tensor<T>> = vec![];
    let dummy_event = Event::empty();
    Ok((empty_tensor, dummy_event))
  }
}

pub struct EwAddOp {
  pub(crate) name: String,
}

impl<T> NodeOp<T> for EwAddOp
where
  T: TensorType,
{
  fn name(&self) -> &str {
    &self.name
  }

  fn forward(
    &self,
    parents: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Tensor<T>, Option<Event>), Error> {
    if parents.len() != 2 {
      return Err(Error::from("EwAddOp requires exactly two parent tensors"));
    }
    let lhs = parents[0];
    let rhs = parents[1];
    if lhs.shape != rhs.shape {
      return Err(Error::from("EwAddOp requires tensors of the same shape"));
    }
    let ws_size = lhs.len();
    let ndim = lhs.dim();
    let output = Tensor::zeros(lhs.shape.clone());
    let kernel = ocl::Kernel::builder()
      .program(lhs.env.get_program::<T>()?.as_ref())
      .name("ew_add_fwd")
      .queue(lhs.env.proque().queue().clone())
      .arg(&lhs.buffer)
      .global_work_size(ws_size)
      .arg(array_2_buffer(lhs.strides.clone()))
      .arg(&rhs.buffer)
      .arg(array_2_buffer(rhs.strides.clone()))
      .arg(&output.buffer)
      .arg(array_2_buffer(output.strides.clone()))
      .arg(ndim as u32)
      .build()?;

    let event = enq_kernel(&kernel, events)?;
    Ok((output, Some(event)))
  }

  fn backward(
    &self,
    childrens: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Vec<&Tensor<T>>, Event), Error> {
    // Implement the backward operation logic here
    unimplemented!()
  }
}

pub struct EwSubOp {
  pub(crate) name: String,
}

impl<T> NodeOp<T> for EwSubOp
where
  T: TensorType,
{
  fn name(&self) -> &str {
    &self.name
  }

  fn forward(
    &self,
    parents: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Tensor<T>, Option<Event>), Error> {
    if parents.len() != 2 {
      return Err(Error::from("EwSubOp requires exactly two parent tensors"));
    }
    let lhs = parents[0];
    let rhs = parents[1];
    if lhs.shape != rhs.shape {
      return Err(Error::from("EwSubOp requires tensors of the same shape"));
    }
    let ws_size = lhs.len();
    let ndim = lhs.dim();
    let output = Tensor::zeros(lhs.shape.clone());
    let kernel = ocl::Kernel::builder()
      .program(lhs.env.get_program::<T>()?.as_ref())
      .name("ew_sub_fwd")
      .queue(lhs.env.proque().queue().clone())
      .arg(&lhs.buffer)
      .global_work_size(ws_size)
      .arg(array_2_buffer(lhs.strides.clone()))
      .arg(&rhs.buffer)
      .arg(array_2_buffer(rhs.strides.clone()))
      .arg(&output.buffer)
      .arg(array_2_buffer(output.strides.clone()))
      .arg(ndim as u32)
      .build()?;

    let event = enq_kernel(&kernel, events)?;
    Ok((output, Some(event)))
  }

  fn backward(
    &self,
    childrens: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Vec<&Tensor<T>>, Event), Error> {
    // Implement the backward operation logic here
    unimplemented!()
  }
}

pub struct EwMulOp {
  pub(crate) name: String,
}

impl<T> NodeOp<T> for EwMulOp
where
  T: TensorType,
{
  fn name(&self) -> &str {
    &self.name
  }

  fn forward(
    &self,
    parents: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Tensor<T>, Option<Event>), Error> {
    if parents.len() != 2 {
      return Err(Error::from("EwMulOp requires exactly two parent tensors"));
    }
    let lhs = parents[0];
    let rhs = parents[1];
    if lhs.shape != rhs.shape {
      return Err(Error::from("EwMulOp requires tensors of the same shape"));
    }
    let ws_size = lhs.len();
    let ndim = lhs.dim();
    let output = Tensor::zeros(lhs.shape.clone());
    let kernel = ocl::Kernel::builder()
      .program(lhs.env.get_program::<T>()?.as_ref())
      .name("ew_mul_fwd")
      .queue(lhs.env.proque().queue().clone())
      .arg(&lhs.buffer)
      .global_work_size(ws_size)
      .arg(array_2_buffer(lhs.strides.clone()))
      .arg(&rhs.buffer)
      .arg(array_2_buffer(rhs.strides.clone()))
      .arg(&output.buffer)
      .arg(array_2_buffer(output.strides.clone()))
      .arg(ndim as u32)
      .build()?;

    let event = enq_kernel(&kernel, events)?;
    Ok((output, Some(event)))
  }

  fn backward(
    &self,
    childrens: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Vec<&Tensor<T>>, Event), Error> {
    // Implement the backward operation logic here
    unimplemented!()
  }
}

pub struct EwDivOp {
  pub(crate) name: String,
}

impl<T> NodeOp<T> for EwDivOp
where
  T: TensorType,
{
  fn name(&self) -> &str {
    &self.name
  }

  fn forward(
    &self,
    parents: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Tensor<T>, Option<Event>), Error> {
    if parents.len() != 2 {
      return Err(Error::from("EwDivOp requires exactly two parent tensors"));
    }
    let lhs = parents[0];
    let rhs = parents[1];
    if lhs.shape != rhs.shape {
      return Err(Error::from("EwDivOp requires tensors of the same shape"));
    }
    let ws_size = lhs.len();
    let ndim = lhs.dim();
    let output = Tensor::zeros(lhs.shape.clone());
    let kernel = ocl::Kernel::builder()
      .program(lhs.env.get_program::<T>()?.as_ref())
      .name("ew_div_fwd")
      .queue(lhs.env.proque().queue().clone())
      .arg(&lhs.buffer)
      .global_work_size(ws_size)
      .arg(array_2_buffer(lhs.strides.clone()))
      .arg(&rhs.buffer)
      .arg(array_2_buffer(rhs.strides.clone()))
      .arg(&output.buffer)
      .arg(array_2_buffer(output.strides.clone()))
      .arg(ndim as u32)
      .build()?;

    let event = enq_kernel(&kernel, events)?;
    Ok((output, Some(event)))
  }

  fn backward(
    &self,
    childrens: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Vec<&Tensor<T>>, Event), Error> {
    // Implement the backward operation logic here
    unimplemented!()
  }
}

struct AbsOp {
  pub(crate) name: String,
}

impl<T> NodeOp<T> for AbsOp
where
  T: TensorType,
{
  fn name(&self) -> &str {
    &self.name
  }

  fn forward(
    &self,
    parents: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Tensor<T>, Option<Event>), Error> {
    if parents.len() != 1 {
      return Err(Error::from("AbsOp requires exactly one parent tensor"));
    }
    let input = parents[0];
    let ws_size = input.len();
    let ndim = input.dim();
    let output = Tensor::zeros(input.shape.clone());
    let kernel = ocl::Kernel::builder()
      .program(input.env.get_program::<T>()?.as_ref())
      .name("abs_fwd")
      .queue(input.env.proque().queue().clone())
      .global_work_size(ws_size)
      .arg(&input.buffer)
      .arg(array_2_buffer(input.strides.clone()))
      .arg(&output.buffer)
      .arg(array_2_buffer(output.strides.clone()))
      .arg(ndim as u32)
      .build()?;

    let event = enq_kernel(&kernel, events)?;
    Ok((output, Some(event)))
  }

  fn backward(
    &self,
    childrens: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Vec<&Tensor<T>>, Event), Error> {
    // Implement the backward operation logic here
    unimplemented!()
  }
}

pub struct PowOp {
  pub(crate) name: String,
}

impl<T> NodeOp<T> for PowOp
where
  T: TensorType,
{
  fn name(&self) -> &str {
    &self.name
  }

  fn forward(
    &self,
    parents: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Tensor<T>, Option<Event>), Error> {
    if parents.len() != 2 {
      return Err(Error::from("PowOp requires exactly two parent tensors"));
    }
    let base = parents[0];
    let exponent = parents[1];
    if exponent.dim() != 0 {
      return Err(Error::from("Exponent must be a scalar tensor"));
    }
    let ws_size = base.len();
    let ndim = base.dim();
    let output = Tensor::zeros(base.shape.clone());
    let kernel = ocl::Kernel::builder()
      .program(base.env.get_program::<T>()?.as_ref())
      .name("pow_fwd")
      .queue(base.env.proque().queue().clone())
      .global_work_size(ws_size)
      .arg(&base.buffer)
      .arg(array_2_buffer(base.strides.clone()))
      .arg(&exponent.buffer)
      .arg(&output.buffer)
      .arg(array_2_buffer(output.strides.clone()))
      .arg(ndim as u32)
      .build()?;

    let event = enq_kernel(&kernel, events)?;
    Ok((output, Some(event)))
  }

  fn backward(
    &self,
    childrens: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Vec<&Tensor<T>>, Event), Error> {
    // Implement the backward operation logic here
    unimplemented!()
  }
}

struct SqrtOp {
  pub(crate) name: String,
}

impl<T> NodeOp<T> for SqrtOp
where
  T: TensorType,
{
  fn name(&self) -> &str {
    &self.name
  }

  fn forward(
    &self,
    parents: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Tensor<T>, Option<Event>), Error> {
    if parents.len() != 1 {
      return Err(Error::from("SqrtOp requires exactly one parent tensor"));
    }
    let input = parents[0];
    let ws_size = input.len();
    let ndim = input.dim();
    let output = Tensor::zeros(input.shape.clone());
    let kernel = ocl::Kernel::builder()
      .program(input.env.get_program::<T>()?.as_ref())
      .name("sqrt_fwd")
      .queue(input.env.proque().queue().clone())
      .global_work_size(ws_size)
      .arg(&input.buffer)
      .arg(array_2_buffer(input.strides.clone()))
      .arg(&output.buffer)
      .arg(array_2_buffer(output.strides.clone()))
      .arg(ndim as u32)
      .build()?;

    let event = enq_kernel(&kernel, events)?;
    Ok((output, Some(event)))
  }

  fn backward(
    &self,
    childrens: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Vec<&Tensor<T>>, Event), Error> {
    // Implement the backward operation logic here
    unimplemented!()
  }
}

struct ExpOp {
  pub(crate) name: String,
}

impl<T> NodeOp<T> for ExpOp
where
  T: TensorType,
{
  fn name(&self) -> &str {
    &self.name
  }

  fn forward(
    &self,
    parents: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Tensor<T>, Option<Event>), Error> {
    if parents.len() != 1 {
      return Err(Error::from("ExpOp requires exactly one parent tensor"));
    }
    let input = parents[0];
    let ws_size = input.len();
    let ndim = input.dim();
    let output = Tensor::zeros(input.shape.clone());
    let kernel = ocl::Kernel::builder()
      .program(input.env.get_program::<T>()?.as_ref())
      .name("exp_fwd")
      .queue(input.env.proque().queue().clone())
      .global_work_size(ws_size)
      .arg(&input.buffer)
      .arg(array_2_buffer(input.strides.clone()))
      .arg(&output.buffer)
      .arg(array_2_buffer(output.strides.clone()))
      .arg(ndim as u32)
      .build()?;

    let event = enq_kernel(&kernel, events)?;
    Ok((output, Some(event)))
  }

  fn backward(
    &self,
    childrens: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Vec<&Tensor<T>>, Event), Error> {
    // Implement the backward operation logic here
    unimplemented!()
  }
}

pub struct LnOp {
  pub(crate) name: String,
}

impl<T> NodeOp<T> for LnOp
where
  T: TensorType,
{
  fn name(&self) -> &str {
    &self.name
  }

  fn forward(
    &self,
    parents: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Tensor<T>, Option<Event>), Error> {
    if parents.len() != 1 {
      return Err(Error::from("LnOp requires exactly one parent tensor"));
    }
    let input = parents[0];
    let ws_size = input.len();
    let ndim = input.dim();
    let output = Tensor::zeros(input.shape.clone());
    let kernel = ocl::Kernel::builder()
      .program(input.env.get_program::<T>()?.as_ref())
      .name("ln_fwd")
      .queue(input.env.proque().queue().clone())
      .global_work_size(ws_size)
      .arg(&input.buffer)
      .arg(array_2_buffer(input.strides.clone()))
      .arg(&output.buffer)
      .arg(array_2_buffer(output.strides.clone()))
      .arg(ndim as u32)
      .build()?;

    let event = enq_kernel(&kernel, events)?;
    Ok((output, Some(event)))
  }

  fn backward(
    &self,
    childrens: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Vec<&Tensor<T>>, Event), Error> {
    // Implement the backward operation logic here
    unimplemented!()
  }
}

pub struct LogOp {
  pub(crate) name: String,
}

impl<T> NodeOp<T> for LogOp
where
  T: TensorType,
{
  fn name(&self) -> &str {
    &self.name
  }

  fn forward(
    &self,
    parents: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Tensor<T>, Option<Event>), Error> {
    if parents.len() != 2 {
      return Err(Error::from("LogOp requires exactly two parent tensors"));
    }
    let argument = parents[0];
    let base = parents[1];
    if base.dim() != 0 {
      return Err(Error::from("Base must be a scalar tensor"));
    }
    let ws_size = argument.len();
    let ndim = argument.dim();
    let output = Tensor::zeros(argument.shape.clone());
    let kernel = ocl::Kernel::builder()
      .program(argument.env.get_program::<T>()?.as_ref())
      .name("log_fwd")
      .queue(argument.env.proque().queue().clone())
      .global_work_size(ws_size)
      .arg(&argument.buffer)
      .arg(array_2_buffer(argument.strides.clone()))
      .arg(&base.buffer)
      .arg(&output.buffer)
      .arg(array_2_buffer(output.strides.clone()))
      .arg(ndim as u32)
      .build()?;
    let event = enq_kernel(&kernel, events)?;
    Ok((output, Some(event)))
  }

  fn backward(
    &self,
    childrens: Vec<&Tensor<T>>,
    events: Vec<&Event>,
  ) -> Result<(Vec<&Tensor<T>>, Event), Error> {
    // Implement the backward operation logic here
    unimplemented!()
  }
}
