use std::{
  cell::RefCell,
  rc::{Rc, Weak},
};

use num_traits::Float;
use ocl::{Event, OclPrm};

use crate::clarray::{
  op::{EwAddOp, EwDivOp, EwMulOp, EwSubOp, Input, NodeOp},
  tape::Tape,
  tensor::{Tensor, TensorType},
};

#[derive(Clone)]
pub struct Node<T>
where
  T: TensorType,
{
  pub id: usize,
  pub name: Option<String>,
  parents: RefCell<Vec<NodeRef<T>>>,
  children: RefCell<Vec<NodeWeak<T>>>,
  node_type: Rc<dyn NodeOp<T>>,
  node_event: Option<Event>,
  pub tape: Rc<RefCell<Tape<T>>>,
  pub output: Tensor<T>,
  pub grad: Tensor<T>,
}

pub type NodeWeak<T> = Weak<RefCell<Node<T>>>;
pub type NodeRef<T> = Rc<RefCell<Node<T>>>;

impl<T> Node<T>
where
  T: TensorType,
{
  pub fn add_children(&mut self, child: &NodeRef<T>) {
    self.children.borrow_mut().push(Rc::downgrade(child));
  }

  pub fn push_unary_op(parent: &Node<T>, node_type: Rc<dyn NodeOp<T>>) -> Self {
    let (tensor, event) = node_type
      .forward(
        vec![&parent.output],
        [parent.node_event.clone()].iter().flatten().collect(),
      )
      .unwrap();

    let res = Self {
      id: 0,
      name: Some(node_type.name().to_string()),
      parents: RefCell::new(vec![Rc::new(RefCell::new(parent.clone()))]),
      children: RefCell::new(vec![]),
      node_type: node_type.clone(),
      node_event: event.clone(),
      tape: parent.tape.clone(),
      output: tensor.clone(),
      grad: Tensor::zeros(tensor.shape.clone()),
    };
    parent
      .tape
      .borrow_mut()
      .push_node(&Rc::downgrade(&Rc::new(RefCell::new(res.clone()))));
    res
  }

  pub fn push_binary_op(
    parent1: &Node<T>,
    parent2: &Node<T>,
    node_type: Rc<dyn NodeOp<T>>,
  ) -> Self {
    let (tensor, event) = node_type
      .forward(
        vec![&parent1.output, &parent2.output],
        [parent1.node_event.clone(), parent2.node_event.clone()]
          .iter()
          .flatten()
          .collect(),
      )
      .unwrap();

    Self {
      id: 0,
      name: Some("test".to_string()),
      parents: RefCell::new(vec![
        Rc::new(RefCell::new(parent1.clone())),
        Rc::new(RefCell::new(parent2.clone())),
      ]),
      children: RefCell::new(vec![]),
      node_type: node_type.clone(),
      node_event: event.clone(),
      tape: parent1.tape.clone(),
      output: tensor.clone(),
      grad: Tensor::zeros(tensor.shape.clone()),
    }
  }

  pub fn push_input(
    id: usize,
    name: String,
    output: Tensor<T>,
    tape: Rc<RefCell<Tape<T>>>,
  ) -> Self {
    let res = Self {
      id,
      name: Some(name),
      parents: RefCell::new(vec![]),
      children: RefCell::new(vec![]),
      node_type: Rc::new(Input {
        name: "input".to_string(),
      }),
      node_event: None,
      tape,
      output: output.clone(),
      grad: Tensor::zeros(output.shape.clone()),
    };
    res
      .tape
      .borrow_mut()
      .push_node(&Rc::downgrade(&Rc::new(RefCell::new(res.clone()))));
    res
  }

  pub fn to_cpu(&self) -> Vec<T> {
    let mut host_buffer = vec![T::default(); self.output.buffer.len()];
    self.output.buffer.read(&mut host_buffer).enq().unwrap();
    host_buffer
  }

  pub fn zeros(shape: Vec<usize>, tape: Rc<RefCell<Tape<T>>>) -> Self {
    let input_tensor = Tensor::<T>::zeros(shape);
    Self::push_input(0, "zeros".to_string(), input_tensor, tape)
  }

  pub fn from_array(shape: Vec<usize>, array: Vec<T>, tape: Rc<RefCell<Tape<T>>>) -> Self {
    let input_tensor = Tensor::<T>::from_array(shape, array);
    Self::push_input(0, "from_array".to_string(), input_tensor, tape)
  }

  pub fn scalar(value: T, tape: Rc<RefCell<Tape<T>>>) -> Self {
    let input_tensor = Tensor::<T>::scalar(value);
    Self::push_input(0, "scalar".to_string(), input_tensor, tape)
  }
}

// し則演算を実装
impl<T> std::ops::Add for &Node<T>
where
  T: TensorType + Float,
{
  type Output = Node<T>;

  fn add(self, other: Self) -> Self::Output {
    let node_type = Rc::new(EwAddOp {
      name: "add".to_string(),
    });
    Node::push_binary_op(&self, &other, node_type)
  }
}

impl<T> std::ops::Sub for &Node<T>
where
  T: TensorType + Float,
{
  type Output = Node<T>;

  fn sub(self, other: Self) -> Self::Output {
    let node_type = Rc::new(EwSubOp {
      name: "sub".to_string(),
    });
    Node::push_binary_op(&self, &other, node_type)
  }
}

impl<T> std::ops::Mul for &Node<T>
where
  T: TensorType + Float,
{
  type Output = Node<T>;

  fn mul(self, other: Self) -> Self::Output {
    let node_type = Rc::new(EwMulOp {
      name: "mul".to_string(),
    });
    Node::push_binary_op(&self, &other, node_type)
  }
}

impl<T> std::ops::Div for &Node<T>
where
  T: TensorType + Float,
{
  type Output = Node<T>;

  fn div(self, other: Self) -> Self::Output {
    let node_type = Rc::new(EwDivOp {
      name: "div".to_string(),
    });
    Node::push_binary_op(&self, &other, node_type)
  }
}
