mod clarray;
#[cfg(test)]
mod tests {
  use std::{cell::RefCell, rc::Rc};

  use crate::clarray::{
    node::Node,
    tape::{Direction, LabelKind, MermaidOpts, Tape, to_mermaid_flowchart},
  };

  #[test]
  fn forward_test() {
    let tape = Rc::new(RefCell::new(Tape::<f32>::new()));
    let a = Node::from_array(vec![2, 2], vec![2.0, 1.0, 5.0, 3.0], tape.clone());
    let b = Node::from_array(vec![2, 2], vec![0.2, 8.0, 2.0, 9.0], tape.clone());
    let x = Node::scalar(2.0, tape.clone());
    let c = &a.exp() / &b.pow(&x);

    let out = c.to_cpu();
    println!("c_cpu: {:?}", out);
    let output = to_mermaid_flowchart(
      &tape,
      &MermaidOpts {
        direction: Direction::LR,
        label_kind: LabelKind::NameAndShortUuid,
        show_layers: false,
      },
    );
    println!("Mermaid output:\n{}", output);
  }
}
