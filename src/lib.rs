mod clarray;
#[cfg(test)]
mod tests {
  use std::{cell::RefCell, rc::Rc};

  use crate::clarray::{node::Node, tape::Tape, tensor::Tensor};

  #[test]
  fn forward_test() {
    let tape = Rc::new(RefCell::new(Tape::<f32>::new()));
    let a = Node::from_array(vec![2, 2], vec![2.0, 1.0, 5.0, 3.0], tape.clone());
    let b = Node::from_array(vec![2, 2], vec![0.2, 8.0, 2.0, 9.0], tape.clone());
    let c = &a * &b;

    let out = c.to_cpu();
    println!("c_cpu: {:?}", out);
  }
}
