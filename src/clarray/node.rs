use std::{
  cell::RefCell,
  rc::{Rc, Weak},
};

use num_traits::Float;
use ocl::{Event, OclPrm};
use uuid::Uuid;

use crate::clarray::{
  op::{
    AbsOp, CosOp, DotBaseOp, DotOp, EwAddOp, EwDivOp, EwMulOp, EwSubOp, ExpOp, Input, LnOp, LogOp,
    NegOp, NodeOp, PowOp, SinOp, SqrtOp, TanOp,
  },
  tape::Tape,
  tensor::{Tensor, TensorType},
};

#[derive(Clone)]
pub struct Node<T>
where
  T: TensorType,
{
  pub id: Uuid,
  pub parents: RefCell<Vec<NodeRef<T>>>,
  children: RefCell<Vec<NodeWeak<T>>>,
  pub node_type: Rc<dyn NodeOp<T>>,
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
  pub fn name(&self) -> String {
    self.node_type.name().to_owned()
  }

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
      id: Uuid::new_v4(),
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
      .push_node(&Rc::new(RefCell::new(res.clone())));
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

    let res = Self {
      id: Uuid::new_v4(),
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
    };
    parent1
      .tape
      .borrow_mut()
      .push_node(&Rc::new(RefCell::new(res.clone())));
    res
  }

  pub fn push_input(name: String, output: Tensor<T>, tape: Rc<RefCell<Tape<T>>>) -> Self {
    let res = Self {
      id: Uuid::new_v4(),
      parents: RefCell::new(vec![]),
      children: RefCell::new(vec![]),
      node_type: Rc::new(Input { name }),
      node_event: None,
      tape,
      output: output.clone(),
      grad: Tensor::zeros(output.shape.clone()),
    };
    res
      .tape
      .borrow_mut()
      .push_node(&Rc::new(RefCell::new(res.clone())));
    res
  }

  pub fn to_cpu(&self) -> Vec<T> {
    let mut host_buffer = vec![T::default(); self.output.buffer.len()];
    self.output.buffer.read(&mut host_buffer).enq().unwrap();
    host_buffer
  }

  pub fn zeros(shape: Vec<usize>, tape: Rc<RefCell<Tape<T>>>) -> Self {
    let input_tensor = Tensor::<T>::zeros(shape);
    Self::push_input("zeros".to_string(), input_tensor, tape)
  }

  pub fn from_array(shape: Vec<usize>, array: Vec<T>, tape: Rc<RefCell<Tape<T>>>) -> Self {
    let input_tensor = Tensor::<T>::from_array(shape, array);
    Self::push_input("from_array".to_string(), input_tensor, tape)
  }

  pub fn scalar(value: T, tape: Rc<RefCell<Tape<T>>>) -> Self {
    let input_tensor = Tensor::<T>::scalar(value);
    Self::push_input("scalar".to_string(), input_tensor, tape)
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

impl<T> std::ops::Neg for &Node<T>
where
  T: TensorType + Float,
{
  type Output = Node<T>;

  fn neg(self) -> Self::Output {
    let node_type = Rc::new(NegOp {
      name: "neg".to_string(),
    });
    Node::push_unary_op(self, node_type)
  }
}

impl<T> Node<T>
where
  T: TensorType + Float,
{
  pub fn abs(&self) -> Self {
    let node_type = Rc::new(AbsOp {
      name: "abs".to_string(),
    });
    Node::push_unary_op(self, node_type)
  }

  pub fn pow(&self, exponent: &Node<T>) -> Self {
    let node_type = Rc::new(PowOp {
      name: "pow".to_string(),
    });
    Node::push_binary_op(self, exponent, node_type)
  }

  pub fn sqrt(&self) -> Self {
    let node_type = Rc::new(SqrtOp {
      name: "sqrt".to_string(),
    });
    Node::push_unary_op(self, node_type)
  }

  pub fn exp(&self) -> Self {
    let node_type = Rc::new(ExpOp {
      name: "exp".to_string(),
    });
    Node::push_unary_op(self, node_type)
  }

  pub fn ln(&self) -> Self {
    let node_type = Rc::new(LnOp {
      name: "ln".to_string(),
    });
    Node::push_unary_op(self, node_type)
  }

  pub fn log(&self, base: &Node<T>) -> Self {
    let node_type = Rc::new(LogOp {
      name: "log".to_string(),
    });
    Node::push_binary_op(self, base, node_type)
  }

  pub fn sin(&self) -> Self {
    let node_type = Rc::new(SinOp {
      name: "sin".to_string(),
    });
    Node::push_unary_op(self, node_type)
  }

  pub fn cos(&self) -> Self {
    let node_type = Rc::new(CosOp {
      name: "cos".to_string(),
    });
    Node::push_unary_op(self, node_type)
  }

  pub fn tan(&self) -> Self {
    let node_type = Rc::new(TanOp {
      name: "tan".to_string(),
    });
    Node::push_unary_op(self, node_type)
  }

  pub fn dot_base(&self, other: &Node<T>) -> Self {
    let node_type = Rc::new(DotBaseOp {
      name: "dot_base".to_string(),
    });
    Node::push_binary_op(self, other, node_type)
  }

  pub fn dot(&self, other: &Node<T>) -> Self {
    let node_type = Rc::new(DotOp {
      name: "dot".to_string(),
    });
    Node::push_binary_op(self, other, node_type)
  }
}
