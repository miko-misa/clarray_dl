use std::{
  cell::RefCell,
  collections::{BTreeMap, HashMap, HashSet, VecDeque},
  rc::Rc,
};

use num_traits::Float;
use ocl::OclPrm;
use uuid::Uuid;

use crate::clarray::{
  node::{Node, NodeRef, NodeWeak},
  tensor::TensorType,
};

pub struct Tape<T>
where
  T: TensorType,
{
  pub nodes: RefCell<Vec<NodeRef<T>>>,
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

  pub fn push_node(&mut self, node: &NodeRef<T>) {
    self.nodes.borrow_mut().push(node.clone());
  }
}

#[derive(Clone, Copy)]
pub enum Direction {
  TD,
  LR,
  BT,
  RL,
}

pub enum LabelKind {
  NameOnly,
  ShortUuidOnly,
  NameAndShortUuid,
}

pub struct MermaidOpts {
  pub direction: Direction,
  pub label_kind: LabelKind,
  pub show_layers: bool,
}
impl Default for MermaidOpts {
  fn default() -> Self {
    Self {
      direction: Direction::TD,
      label_kind: LabelKind::NameAndShortUuid,
      show_layers: true,
    }
  }
}

fn direction_str(d: Direction) -> &'static str {
  match d {
    Direction::TD => "TD",
    Direction::LR => "LR",
    Direction::BT => "BT",
    Direction::RL => "RL",
  }
}

fn short_uuid(u: &Uuid) -> String {
  u.to_string().split('-').next().unwrap().to_string()
}
fn node_sym(u: &Uuid) -> String {
  format!("n_{}", short_uuid(u))
}
fn fmt_label(raw: &str) -> String {
  raw
    .replace('\\', "\\\\")
    .replace('"', "\\\"")
    .replace('\n', "\\n")
}

fn display_of(id: &Uuid, name: &str, kind: &LabelKind) -> String {
  match kind {
    LabelKind::NameOnly => {
      if name.is_empty() {
        short_uuid(id)
      } else {
        name.to_string()
      }
    }
    LabelKind::ShortUuidOnly => short_uuid(id),
    LabelKind::NameAndShortUuid => {
      if name.is_empty() {
        short_uuid(id)
      } else {
        format!("{name} ({})", short_uuid(id))
      }
    }
  }
}

// トポ順から O(N) で depth を計算
fn compute_depths<T: TensorType>(
  topo: &[NodeRef<T>],
) -> (
  HashMap<Uuid, usize>,
  HashMap<Uuid, Vec<Uuid>>,
  HashMap<Uuid, String>,
) {
  let mut depth = HashMap::with_capacity(topo.len());
  let mut parents = HashMap::with_capacity(topo.len());
  let mut names = HashMap::with_capacity(topo.len());

  for nrc in topo {
    let n = nrc.borrow();
    let ps: Vec<Uuid> = n.parents.borrow().iter().map(|p| p.borrow().id).collect();
    let d = if ps.is_empty() {
      0
    } else {
      ps.iter().map(|pid| depth[pid]).max().unwrap() + 1
    };
    depth.insert(n.id, d);
    parents.insert(n.id, ps);
    names.insert(n.id, n.name().to_string());
  }
  (depth, parents, names)
}

/* ========== Mermaid 生成本体 ========== */

/// 仕様:
/// - 親0: 角丸四角 ( ... )
/// - 親1: 四角 [ ... ]
/// - 親2: ヘキサゴン {{ ... }}
/// - 親2のエッジに "lhs"/"rhs" ラベルを付与
/// - show_layers=true なら各 depth を subgraph にまとめる（同層は横並び）
pub fn to_mermaid_flowchart<T: TensorType>(
  tape: &Rc<RefCell<Tape<T>>>,
  opts: &MermaidOpts,
) -> String {
  let topo = tape.borrow().nodes.borrow().clone(); // トポ順
  let (depth, parents, names) = compute_depths(&topo);

  // レイヤ仕分け（出力順はトポ順を尊重）
  let mut by_layer: BTreeMap<usize, Vec<Uuid>> = BTreeMap::new();
  for nrc in topo {
    let id = nrc.borrow().id;
    by_layer.entry(depth[&id]).or_default().push(id);
  }

  let mut out = String::new();
  out.push_str(&format!("flowchart {}\n", direction_str(opts.direction)));

  // ノード宣言（サブグラフ：任意）
  if opts.show_layers {
    for (l, ids) in &by_layer {
      out.push_str(&format!("  subgraph L{l}[\"Layer {l}\"]\n"));
      out.push_str("    direction LR\n"); // 層内は横並び
      for id in ids {
        let sym = node_sym(id);
        let label = fmt_label(&display_of(id, &names[id], &opts.label_kind));
        let pcount = parents[id].len();
        match pcount {
          0 => out.push_str(&format!("    {sym}([\"{label}\"])\n")), // 角丸四角
          2 => out.push_str(&format!("    {sym}[\"{label}\"]\n")),   // 四角
          1 => out.push_str(&format!("    {sym}{{{{\"{label}\"}}}}\n")), // ヘキサゴン（{{..}}）※format!で{{/}}はエスケープ要
          _ => out.push_str(&format!("    {sym}[\"{label}\"]\n")),
        }
      }
      out.push_str("  end\n");
    }
  } else {
    for (_l, ids) in &by_layer {
      for id in ids {
        let sym = node_sym(id);
        let label = fmt_label(&display_of(id, &names[id], &opts.label_kind));
        let pcount = parents[id].len();
        match pcount {
          0 => out.push_str(&format!("  {sym}([\"{label}\"])\n")),
          2 => out.push_str(&format!("  {sym}[\"{label}\"]\n")),
          1 => out.push_str(&format!("  {sym}{{{{\"{label}\"}}}}\n")),
          _ => out.push_str(&format!("  {sym}[\"{label}\"]\n")),
        }
      }
    }
  }

  // エッジ（親→子）。2親は lhs / rhs ラベル
  for (_l, ids) in &by_layer {
    for id in ids {
      let csym = node_sym(id);
      match parents[id].as_slice() {
        [] => {}
        [p0] => {
          out.push_str(&format!("  {} --> {csym}\n", node_sym(p0)));
        }
        [p0, p1] => {
          out.push_str(&format!("  {} -- lhs --> {csym}\n", node_sym(p0)));
          out.push_str(&format!("  {} -- rhs --> {csym}\n", node_sym(p1)));
        }
        _ => {}
      }
    }
  }

  out
}
