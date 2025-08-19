use std::{cell::RefCell, rc::Rc};

use num_traits::Float;
use ocl::OclPrm;

use crate::clarray::{
  node::{Node, NodeRef, NodeWeak},
  tensor::TensorType,
};

pub struct Tape<T>
where
  T: TensorType,
{
  nodes: RefCell<Vec<NodeWeak<T>>>,
}

impl<T> Tape<T>
where
  T: TensorType,
{
  pub fn new() -> Self {
    Self {
      nodes: RefCell::new(vec![]),
    }
  }

  pub fn push_node(&mut self, node: &NodeWeak<T>) {
    self.nodes.borrow_mut().push(node.clone());
  }
}
