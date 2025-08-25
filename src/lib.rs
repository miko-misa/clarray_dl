mod clarray;
#[cfg(test)]

mod tests {
  use std::{cell::RefCell, rc::Rc};

  use crate::clarray::{
    env::env,
    node::Node,
    tape::{Direction, LabelKind, MermaidOpts, Tape, to_mermaid_flowchart},
  };

  #[test]
  fn forward_test() {
    let tape = Rc::new(RefCell::new(Tape::<f32>::new()));
    let a = Node::from_array(vec![2, 2], vec![2.0, 1.0, 5.0, 3.0], tape.clone());
    let b = Node::from_array(vec![2, 2], vec![0.2, 8.0, 2.0, 9.0], tape.clone());
    let x = Node::scalar(2.0, tape.clone());
    let c = &a.exp().sin() / &b.pow(&x);

    let out = c.to_cpu();
    println!("c_cpu: {:?}", out);
    let output = to_mermaid_flowchart(
      &tape,
      &MermaidOpts {
        direction: Direction::LR,
        label_kind: LabelKind::NameAndShortUuid,
        show_layers: true,
      },
    );
    println!("Mermaid output:\n{}", output);
  }

  #[test]
  fn dot_spead_test() {
    // huge matrix multiplication test
    let a_vec = (0..1000 * 1000)
      .map(|x| ((x % 256) / 256) as f32)
      .collect::<Vec<_>>();
    let b_vec = (0..1000 * 1000)
      .map(|x| (((x + 128) % 256) / 256) as f32)
      .collect::<Vec<_>>();
    let tape = Rc::new(RefCell::new(Tape::<f32>::new()));
    let a = Node::from_array(vec![1000, 1000], a_vec, tape.clone());
    let b = Node::from_array(vec![1000, 1000], b_vec, tape.clone());
    let start = std::time::Instant::now();
    let c = a.dot_base(&b);
    env().proque().queue().finish().unwrap();
    let _ = c.to_cpu();
    let duration = start.elapsed();
    println!("Time elapsed in dot_base() is: {:?}", duration);
    tape.borrow_mut().reset();
    let start = std::time::Instant::now();
    let c = a.dot(&b);
    env().proque().queue().finish().unwrap();
    let _ = c.to_cpu();
    let duration = start.elapsed();
    println!("Time elapsed in dot() is: {:?}", duration);
  }
}
