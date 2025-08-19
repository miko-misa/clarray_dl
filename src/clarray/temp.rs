// Cargo.toml（抜粋）
// [dependencies]
// ocl = "0.19"
// anyhow = "1.0"
// once_cell = "1.19"

use anyhow::{Result, anyhow};
use ocl::{Buffer, Context, Device, Event, Kernel, Platform, Program, Queue, flags::MemFlags};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::{Rc, Weak};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// ==================== グローバルID ====================
static NODE_COUNTER: AtomicUsize = AtomicUsize::new(0);
fn next_node_id() -> usize {
  NODE_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// ==================== OpenCL カーネル ====================
const KERNELS: &str = r#"
__kernel void add_fwd(__global const float* a, __global const float* b, __global float* c, uint n){
  uint i=get_global_id(0); if(i<n) c[i]=a[i]+b[i];
}
__kernel void add_bwd(__global const float* upstream, __global float* da, __global float* db, uint n){
  uint i=get_global_id(0); if(i<n){ da[i]+=upstream[i]; db[i]+=upstream[i]; }
}

__kernel void mul_fwd(__global const float* a, __global const float* b, __global float* c, uint n){
  uint i=get_global_id(0); if(i<n) c[i]=a[i]*b[i];
}
__kernel void mul_bwd(__global const float* a, __global const float* b, __global const float* upstream,
                      __global float* da, __global float* db, uint n){
  uint i=get_global_id(0); if(i<n){ da[i]+=upstream[i]*b[i]; db[i]+=upstream[i]*a[i]; }
}

__kernel void pow_fwd(__global const float* a, float e, __global float* c, uint n){
  uint i=get_global_id(0); if(i<n) c[i]=pow(a[i], e);
}
__kernel void pow_bwd(__global const float* a, float e, __global const float* upstream,
                      __global float* da, uint n){
  uint i=get_global_id(0);
  if(i<n){ float v=a[i]; float g=(e==0.0f)?0.0f:e*pow(v,e-1.0f); da[i]+=upstream[i]*g; }
}

// C[M,N] = A[M,K] * B[K,N]  (素朴GEMM)
__kernel void dot_fwd(__global const float* A, __global const float* B, __global float* C,
                      uint M, uint K, uint N){
  uint r=get_global_id(0), c=get_global_id(1);
  if(r<M && c<N){
    float s=0.0f;
    for(uint t=0;t<K;++t){ s+=A[r*K+t]*B[t*N+c]; }
    C[r*N+c]=s;
  }
}

// dA = dC * B^T
__kernel void dot_bwd_a(__global const float* dC, __global const float* B, __global float* dA,
                        uint M, uint K, uint N){
  uint r=get_global_id(0), c=get_global_id(1);
  if(r<M && c<K){
    float s=0.0f;
    for(uint t=0;t<N;++t){ s+=dC[r*N+t]*B[c*N+t]; }
    dA[r*K+c]+=s;
  }
}

// dB = A^T * dC
__kernel void dot_bwd_b(__global const float* A, __global const float* dC, __global float* dB,
                        uint M, uint K, uint N){
  uint r=get_global_id(0), c=get_global_id(1);
  if(r<K && c<N){
    float s=0.0f;
    for(uint t=0;t<M;++t){ s+=A[t*K+r]*dC[t*N+c]; }
    dB[r*N+c]+=s;
  }
}
"#;

/// ==================== GPU 環境 ====================
#[derive(Clone)]
pub struct GPUEnv {
  pub context: Context,
  pub device: Device,
  pub queue: Queue,
  pub program: Program,
}
impl GPUEnv {
  pub fn new() -> Result<Self> {
    let platform = Platform::default();
    let device = Device::first(platform)?;
    let context = Context::builder()
      .platform(platform)
      .devices(device.clone())
      .build()?;
    let queue = Queue::new(&context, device.clone(), None)?;
    let program = Program::builder()
      .devices(device)
      .src(KERNELS)
      .build(&context)?;
    Ok(Self {
      context,
      device,
      queue,
      program,
    })
  }
}

/// ==================== Tensor ====================
#[derive(Clone)]
pub struct Tensor {
  pub buf: Buffer<f32>,
  pub shape: Vec<usize>,
  pub len: usize,
}
impl Tensor {
  pub fn new_uninit(env: &GPUEnv, shape: &[usize]) -> Result<Self> {
    let len = shape.iter().product();
    let buf = Buffer::<f32>::builder()
      .queue(env.queue.clone())
      .flags(MemFlags::READ_WRITE)
      .len(len)
      .build()?;
    Ok(Self {
      buf,
      shape: shape.to_vec(),
      len,
    })
  }
  pub fn from_slice(env: &GPUEnv, shape: &[usize], host: &[f32]) -> Result<Self> {
    let mut t = Self::new_uninit(env, shape)?;
    if host.len() != t.len {
      return Err(anyhow!("host len {} != {}", host.len(), t.len));
    }
    t.buf.write(host).enq()?;
    Ok(t)
  }
  pub fn zeros(env: &GPUEnv, shape: &[usize]) -> Result<Self> {
    let t = Self::new_uninit(env, shape)?;
    t.buf.write(&vec![0.0f32; t.len]).enq()?;
    Ok(t)
  }
  pub fn is_scalar(&self) -> bool {
    self.len == 1
  }
  pub fn same_shape(&self, rhs: &Tensor) -> bool {
    self.shape == rhs.shape
  }
  pub fn shape1(n: usize) -> Vec<usize> {
    vec![n]
  }
  pub fn shape2(m: usize, n: usize) -> Vec<usize> {
    vec![m, n]
  }
  pub fn scalar() -> Vec<usize> {
    vec![1]
  }
}

/// ==================== Tape（内部） ====================
type NodeRef = Rc<RefCell<Node>>;
type NodeWeak = Weak<RefCell<Node>>;

pub struct Tape {
  pub env: Rc<GPUEnv>,
  pub program: Program,
  pub queue: Queue,
  pub nodes: Vec<NodeRef>,
  pub parents: HashMap<usize, Vec<usize>>,
  pub children: HashMap<usize, Vec<usize>>,
  dead: bool,
}
impl Tape {
  pub fn new(env: Rc<GPUEnv>) -> Result<Rc<RefCell<Self>>> {
    Ok(Rc::new(RefCell::new(Self {
      queue: env.queue.clone(),
      program: env.program.clone(),
      env,
      nodes: vec![],
      parents: HashMap::new(),
      children: HashMap::new(),
      dead: false,
    })))
  }

  fn register_node(&mut self, n: &NodeRef) {
    let id = n.borrow().id;
    let pids = n
      .borrow()
      .parents
      .iter()
      .map(|p| p.borrow().id)
      .collect::<Vec<_>>();
    self.parents.insert(id, pids.clone());
    for pid in pids {
      self.children.entry(pid).or_default().push(id);
    }
    self.nodes.push(n.clone());
  }

  fn reachable_from(&self, end_id: usize) -> HashSet<usize> {
    let mut vis = HashSet::new();
    let mut st = vec![end_id];
    while let Some(u) = st.pop() {
      if !vis.insert(u) {
        continue;
      }
      if let Some(ps) = self.parents.get(&u) {
        for &p in ps {
          st.push(p);
        }
      }
    }
    vis
  }

  fn topo_on(&self, subset: &HashSet<usize>) -> Vec<usize> {
    let mut indeg: HashMap<usize, usize> = HashMap::new();
    for &u in subset {
      let d = self
        .parents
        .get(&u)
        .map(|ps| ps.iter().filter(|pid| subset.contains(pid)).count())
        .unwrap_or(0);
      indeg.insert(u, d);
    }
    let mut q: VecDeque<usize> = indeg
      .iter()
      .filter(|(_, &d)| d == 0)
      .map(|(id, _)| *id)
      .collect();
    let mut out = vec![];
    while let Some(u) = q.pop_front() {
      out.push(u);
      if let Some(ch) = self.children.get(&u) {
        for &v in ch {
          if !subset.contains(&v) {
            continue;
          }
          if let Some(d) = indeg.get_mut(&v) {
            *d -= 1;
            if *d == 0 {
              q.push_back(v);
            }
          }
        }
      }
    }
    out
  }

  fn backward_from(&mut self, end: &NodeRef) -> Result<()> {
    // 1) 終端の grad = 1 をセット（他ノードは子の backward で確保される）
    let ones = vec![1.0f32; end.borrow().out.len];
    let g = Tensor::from_slice(&self.env, &end.borrow().out.shape, &ones)?;
    end.borrow_mut().grad = Some(g);

    // 2) 到達部分グラフの逆トポ順
    let reach = self.reachable_from(end.borrow().id);
    let mut topo = self.topo_on(&reach);
    topo.reverse();

    // 3) 子→親に伝える backward 完了イベント束
    let mut child_bwd: HashMap<usize, Vec<Event>> = HashMap::new();

    for id in topo {
      let node = self
        .nodes
        .iter()
        .find(|n| n.borrow().id == id)
        .unwrap()
        .clone();
      let deps = child_bwd.remove(&id).unwrap_or_default();
      let evts = node.borrow().op.backward(self, &node, &deps)?;
      node.borrow_mut().bwd_events = evts.clone();
      for p in node.borrow().parents.iter() {
        child_bwd
          .entry(p.borrow().id)
          .or_default()
          .extend(evts.iter().cloned());
      }
    }
    Ok(())
  }

  /// lhs の tape に rhs をマージして rhs を無効化
  fn merge_tapes(lhs: &Rc<RefCell<Tape>>, rhs: &Rc<RefCell<Tape>>) -> Result<()> {
    if Rc::ptr_eq(lhs, rhs) {
      return Ok(());
    }
    let mut l = lhs.borrow_mut();
    let mut r = rhs.borrow_mut();
    if r.dead {
      return Ok(());
    }

    for n in r.nodes.iter() {
      n.borrow_mut().tape = Rc::downgrade(lhs);
    }

    let lset: HashSet<_> = l.nodes.iter().map(|n| n.borrow().id).collect();
    for n in r.nodes.drain(..) {
      if !lset.contains(&n.borrow().id) {
        l.nodes.push(n);
      }
    }
    for (k, v) in r.parents.drain() {
      l.parents.insert(k, v);
    }
    for (k, v) in r.children.drain() {
      l.children.insert(k, v);
    }
    r.dead = true;
    Ok(())
  }
}

/// ==================== Node / Op ====================
pub struct Node {
  pub id: usize,
  pub name: String,
  pub tape: Weak<RefCell<Tape>>,
  pub op: Arc<dyn NodeOp>, // プラグイン演算
  pub parents: Vec<NodeRef>,
  pub children: Vec<NodeWeak>,
  pub out: Tensor,
  pub grad: Option<Tensor>, // ★ 各ノードが自分の勾配を保持（∂L/∂out(node)）
  pub fwd_event: Option<Event>,
  pub bwd_events: Vec<Event>,
}
impl Node {
  fn new_input(tape: &Rc<RefCell<Tape>>, name: &str, out: Tensor) -> NodeRef {
    let id = next_node_id();
    let op: Arc<dyn NodeOp> = Arc::new(InputOp);
    let n = Rc::new(RefCell::new(Node {
      id,
      name: name.to_string(),
      tape: Rc::downgrade(tape),
      op,
      parents: vec![],
      children: vec![],
      out,
      grad: None,
      fwd_event: None,
      bwd_events: vec![],
    }));
    tape.borrow_mut().register_node(&n);
    n
  }
}

/// ノード勾配の遅延確保（0 初期化）。戻り値はクローン（Buffer は参照カウント型）。
fn ensure_node_grad(n: &NodeRef) -> Result<Tensor> {
  let tape = n
    .borrow()
    .tape
    .upgrade()
    .ok_or_else(|| anyhow!("tape gone"))?;
  if n.borrow().grad.is_none() {
    let g = Tensor::zeros(&tape.borrow().env, &n.borrow().out.shape)?;
    n.borrow_mut().grad = Some(g);
  }
  Ok(n.borrow().grad.as_ref().unwrap().clone())
}

/// 各演算が実装（forward/backward を一体運用）
pub trait NodeOp: Send + Sync {
  fn infer_shape(&self, parents: &[NodeRef]) -> Result<Vec<usize>>;
  fn enqueue_forward(
    &self,
    tape: &Rc<RefCell<Tape>>,
    node: &NodeRef,
    deps: &[Event],
  ) -> Result<Event>;
  fn backward(&self, tape: &mut Tape, node: &NodeRef, deps: &[Event]) -> Result<Vec<Event>>;
}

struct InputOp;
impl NodeOp for InputOp {
  fn infer_shape(&self, _: &[NodeRef]) -> Result<Vec<usize>> {
    Ok(vec![])
  }
  fn enqueue_forward(&self, _: &Rc<RefCell<Tape>>, _: &NodeRef, _: &[Event]) -> Result<Event> {
    Ok(Event::empty())
  }
  fn backward(&self, _: &mut Tape, _: &NodeRef, _: &[Event]) -> Result<Vec<Event>> {
    Ok(vec![])
  }
}

/// 共通：新ノードを作って forward を流す
fn make_node_and_fwd(
  lhs: &NodeRef,
  rhs: Option<&NodeRef>,
  name: &str,
  op: Arc<dyn NodeOp>,
) -> Result<NodeRef> {
  let tape = ensure_same_tape(lhs, rhs)?;
  let parents: Vec<NodeRef> = match rhs {
    Some(r) => vec![lhs.clone(), r.clone()],
    None => vec![lhs.clone()],
  };
  let out_shape = op.infer_shape(&parents)?;
  let out = Tensor::new_uninit(&tape.borrow().env, &out_shape)?;

  let id = next_node_id();
  let n = Rc::new(RefCell::new(Node {
    id,
    name: name.to_string(),
    tape: Rc::downgrade(&tape),
    op: op.clone(),
    parents: parents.clone(),
    children: vec![],
    out,
    grad: None,
    fwd_event: None,
    bwd_events: vec![],
  }));
  for p in &parents {
    p.borrow_mut().children.push(Rc::downgrade(&n));
  }
  tape.borrow_mut().register_node(&n);

  // 依存イベント → forward enqueue
  let deps = parents
    .iter()
    .filter_map(|p| p.borrow().fwd_event.clone())
    .collect::<Vec<_>>();
  let evt = op.enqueue_forward(&tape, &n, &deps)?;
  n.borrow_mut().fwd_event = Some(evt);
  Ok(n)
}

fn ensure_same_tape(lhs: &NodeRef, rhs: Option<&NodeRef>) -> Result<Rc<RefCell<Tape>>> {
  let t1 = lhs
    .borrow()
    .tape
    .upgrade()
    .ok_or_else(|| anyhow!("lhs tape gone"))?;
  if let Some(r) = rhs {
    let t2 = r
      .borrow()
      .tape
      .upgrade()
      .ok_or_else(|| anyhow!("rhs tape gone"))?;
    if !Rc::ptr_eq(&t1, &t2) {
      Tape::merge_tapes(&t1, &t2)?;
    }
  }
  Ok(t1)
}

fn broadcast2(a: &Tensor, b: &Tensor) -> Result<(Vec<usize>, usize)> {
  if a.same_shape(b) {
    Ok((a.shape.clone(), a.len))
  } else if a.is_scalar() {
    Ok((b.shape.clone(), b.len))
  } else if b.is_scalar() {
    Ok((a.shape.clone(), a.len))
  } else {
    Err(anyhow!("shape mismatch (same or scalar-only supported)"))
  }
}

/// ==================== 既存演算（プラグイン） ====================
/// ---- Add ----
pub struct AddOp;
impl NodeOp for AddOp {
  fn infer_shape(&self, parents: &[NodeRef]) -> Result<Vec<usize>> {
    let (s, _) = broadcast2(&parents[0].borrow().out, &parents[1].borrow().out)?;
    Ok(s)
  }
  fn enqueue_forward(
    &self,
    tape: &Rc<RefCell<Tape>>,
    node: &NodeRef,
    deps: &[Event],
  ) -> Result<Event> {
    let t = tape.borrow();
    let a = node.borrow().parents[0].borrow().out.clone();
    let b = node.borrow().parents[1].borrow().out.clone();
    let n = node.borrow().out.len;
    let mut k = Kernel::builder()
      .program(&t.program)
      .name("add_fwd")
      .queue(t.queue.clone())
      .global_work_size(n)
      .arg(&a.buf)
      .arg(&b.buf)
      .arg(&node.borrow().out.buf)
      .arg(&(n as u32))
      .build()?;
    let mut evt = Event::empty();
    unsafe {
      k.cmd().ewait(deps).enew(&mut evt).enq()?;
    }
    Ok(evt)
  }
  fn backward(&self, tape: &mut Tape, node: &NodeRef, deps: &[Event]) -> Result<Vec<Event>> {
    let parents = node.borrow().parents.clone();
    let a = parents[0].clone();
    let b = parents[1].clone();
    let g_c = node
      .borrow()
      .grad
      .as_ref()
      .ok_or_else(|| anyhow!("grad(node) missing"))?
      .clone();

    let da = ensure_node_grad(&a)?;
    let db = ensure_node_grad(&b)?;
    let n = node.borrow().out.len;

    let mut k = Kernel::builder()
      .program(&tape.program)
      .name("add_bwd")
      .queue(tape.queue.clone())
      .global_work_size(n)
      .arg(&g_c.buf)
      .arg(&da.buf)
      .arg(&db.buf)
      .arg(&(n as u32))
      .build()?;
    let mut evt = Event::empty();
    unsafe {
      k.cmd().ewait(deps).enew(&mut evt).enq()?;
    }
    Ok(vec![evt])
  }
}

/// ---- Mul ----
pub struct MulOp;
impl NodeOp for MulOp {
  fn infer_shape(&self, parents: &[NodeRef]) -> Result<Vec<usize>> {
    let (s, _) = broadcast2(&parents[0].borrow().out, &parents[1].borrow().out)?;
    Ok(s)
  }
  fn enqueue_forward(
    &self,
    tape: &Rc<RefCell<Tape>>,
    node: &NodeRef,
    deps: &[Event],
  ) -> Result<Event> {
    let t = tape.borrow();
    let a = node.borrow().parents[0].borrow().out.clone();
    let b = node.borrow().parents[1].borrow().out.clone();
    let n = node.borrow().out.len;
    let mut k = Kernel::builder()
      .program(&t.program)
      .name("mul_fwd")
      .queue(t.queue.clone())
      .global_work_size(n)
      .arg(&a.buf)
      .arg(&b.buf)
      .arg(&node.borrow().out.buf)
      .arg(&(n as u32))
      .build()?;
    let mut evt = Event::empty();
    unsafe {
      k.cmd().ewait(deps).enew(&mut evt).enq()?;
    }
    Ok(evt)
  }
  fn backward(&self, tape: &mut Tape, node: &NodeRef, deps: &[Event]) -> Result<Vec<Event>> {
    let parents = node.borrow().parents.clone();
    let a = parents[0].clone();
    let b = parents[1].clone();
    let g_c = node
      .borrow()
      .grad
      .as_ref()
      .ok_or_else(|| anyhow!("grad(node) missing"))?
      .clone();

    let da = ensure_node_grad(&a)?;
    let db = ensure_node_grad(&b)?;
    let n = node.borrow().out.len;

    let mut k = Kernel::builder()
      .program(&tape.program)
      .name("mul_bwd")
      .queue(tape.queue.clone())
      .global_work_size(n)
      .arg(&a.borrow().out.buf)
      .arg(&b.borrow().out.buf)
      .arg(&g_c.buf)
      .arg(&da.buf)
      .arg(&db.buf)
      .arg(&(n as u32))
      .build()?;
    let mut evt = Event::empty();
    unsafe {
      k.cmd().ewait(deps).enew(&mut evt).enq()?;
    }
    Ok(vec![evt])
  }
}

/// ---- Pow(e) ----
pub struct PowOp {
  pub e: f32,
}
impl NodeOp for PowOp {
  fn infer_shape(&self, parents: &[NodeRef]) -> Result<Vec<usize>> {
    Ok(parents[0].borrow().out.shape.clone())
  }
  fn enqueue_forward(
    &self,
    tape: &Rc<RefCell<Tape>>,
    node: &NodeRef,
    deps: &[Event],
  ) -> Result<Event> {
    let t = tape.borrow();
    let a = node.borrow().parents[0].borrow().out.clone();
    let n = node.borrow().out.len;
    let mut k = Kernel::builder()
      .program(&t.program)
      .name("pow_fwd")
      .queue(t.queue.clone())
      .global_work_size(n)
      .arg(&a.buf)
      .arg(&self.e)
      .arg(&node.borrow().out.buf)
      .arg(&(n as u32))
      .build()?;
    let mut evt = Event::empty();
    unsafe {
      k.cmd().ewait(deps).enew(&mut evt).enq()?;
    }
    Ok(evt)
  }
  fn backward(&self, tape: &mut Tape, node: &NodeRef, deps: &[Event]) -> Result<Vec<Event>> {
    let a = node.borrow().parents[0].clone();
    let g_c = node
      .borrow()
      .grad
      .as_ref()
      .ok_or_else(|| anyhow!("grad(node) missing"))?
      .clone();

    let da = ensure_node_grad(&a)?;
    let n = node.borrow().out.len;

    let mut k = Kernel::builder()
      .program(&tape.program)
      .name("pow_bwd")
      .queue(tape.queue.clone())
      .global_work_size(n)
      .arg(&a.borrow().out.buf)
      .arg(&self.e)
      .arg(&g_c.buf)
      .arg(&da.buf)
      .arg(&(n as u32))
      .build()?;
    let mut evt = Event::empty();
    unsafe {
      k.cmd().ewait(deps).enew(&mut evt).enq()?;
    }
    Ok(vec![evt])
  }
}

/// ---- Dot ----
pub struct DotOp {
  pub m: usize,
  pub k: usize,
  pub n: usize,
}
impl NodeOp for DotOp {
  fn infer_shape(&self, _: &[NodeRef]) -> Result<Vec<usize>> {
    Ok(vec![self.m, self.n])
  }
  fn enqueue_forward(
    &self,
    tape: &Rc<RefCell<Tape>>,
    node: &NodeRef,
    deps: &[Event],
  ) -> Result<Event> {
    let t = tape.borrow();
    let a = node.borrow().parents[0].borrow().out.clone();
    let b = node.borrow().parents[1].borrow().out.clone();
    let mut kf = Kernel::builder()
      .program(&t.program)
      .name("dot_fwd")
      .queue(t.queue.clone())
      .global_work_size([self.m, self.n])
      .arg(&a.buf)
      .arg(&b.buf)
      .arg(&node.borrow().out.buf)
      .arg(&(self.m as u32))
      .arg(&(self.k as u32))
      .arg(&(self.n as u32))
      .build()?;
    let mut evt = Event::empty();
    unsafe {
      kf.cmd().ewait(deps).enew(&mut evt).enq()?;
    }
    Ok(evt)
  }
  fn backward(&self, tape: &mut Tape, node: &NodeRef, deps: &[Event]) -> Result<Vec<Event>> {
    let parents = node.borrow().parents.clone();
    let a = parents[0].clone();
    let b = parents[1].clone();
    let g_c = node
      .borrow()
      .grad
      .as_ref()
      .ok_or_else(|| anyhow!("grad(node) missing"))?
      .clone();

    let dA = ensure_node_grad(&a)?;
    let dB = ensure_node_grad(&b)?;

    let mut evt_a = Event::empty();
    let mut evt_b = Event::empty();

    let mut ka = Kernel::builder()
      .program(&tape.program)
      .name("dot_bwd_a")
      .queue(tape.queue.clone())
      .global_work_size([self.m, self.k])
      .arg(&g_c.buf)
      .arg(&b.borrow().out.buf)
      .arg(&dA.buf)
      .arg(&(self.m as u32))
      .arg(&(self.k as u32))
      .arg(&(self.n as u32))
      .build()?;

    let mut kb = Kernel::builder()
      .program(&tape.program)
      .name("dot_bwd_b")
      .queue(tape.queue.clone())
      .global_work_size([self.k, self.n])
      .arg(&a.borrow().out.buf)
      .arg(&g_c.buf)
      .arg(&dB.buf)
      .arg(&(self.m as u32))
      .arg(&(self.k as u32))
      .arg(&(self.n as u32))
      .build()?;

    unsafe {
      ka.cmd().ewait(deps).enew(&mut evt_a).enq()?;
      kb.cmd().ewait(deps).enew(&mut evt_b).enq()?;
    }
    Ok(vec![evt_a, evt_b])
  }
}

/// ==================== 拡張トレイト（演算ごとに追加） ====================
pub trait AddExt {
  fn add(&self, rhs: &NodeRef) -> Result<NodeRef>;
}
impl AddExt for NodeRef {
  fn add(&self, rhs: &NodeRef) -> Result<NodeRef> {
    make_node_and_fwd(self, Some(rhs), "add", Arc::new(AddOp))
  }
}
pub trait MulExt {
  fn mul(&self, rhs: &NodeRef) -> Result<NodeRef>;
}
impl MulExt for NodeRef {
  fn mul(&self, rhs: &NodeRef) -> Result<NodeRef> {
    make_node_and_fwd(self, Some(rhs), "mul", Arc::new(MulOp))
  }
}
pub trait PowExt {
  fn powf(&self, e: f32) -> Result<NodeRef>;
}
impl PowExt for NodeRef {
  fn powf(&self, e: f32) -> Result<NodeRef> {
    make_node_and_fwd(self, None, "pow", Arc::new(PowOp { e }))
  }
}
pub trait DotExt {
  fn dot(&self, rhs: &NodeRef) -> Result<NodeRef>;
}
impl DotExt for NodeRef {
  fn dot(&self, rhs: &NodeRef) -> Result<NodeRef> {
    let a = self.borrow().out.shape.clone();
    let b = rhs.borrow().out.shape.clone();
    if a.len() != 2 || b.len() != 2 {
      return Err(anyhow!("dot needs 2D"));
    }
    if a[1] != b[0] {
      return Err(anyhow!("A[m,k] dot B[k,n] mismatch"));
    }
    let (m, k, n) = (a[0], a[1], b[1]);
    make_node_and_fwd(self, Some(rhs), "dot", Arc::new(DotOp { m, k, n }))
  }
}

/// ==================== ユーザAPI：backward 起動（自由関数） ====================
pub fn backward(end: &NodeRef) -> Result<()> {
  let tape = end
    .borrow()
    .tape
    .upgrade()
    .ok_or_else(|| anyhow!("tape gone"))?;
  tape.borrow_mut().backward_from(end)
}

/// ==================== デモ & 検証 ====================
fn main() -> Result<()> {
  let env = Rc::new(GPUEnv::new()?);
  let tape = Tape::new(env.clone())?;

  // ベクトル3要素
  let a0 = [1.0, 2.0, 3.0];
  let b0 = [4.0, 5.0, 6.0];
  let a = Node::new_input(
    &tape,
    "a",
    Tensor::from_slice(&env, &Tensor::shape1(3), &a0)?,
  );
  let b = Node::new_input(
    &tape,
    "b",
    Tensor::from_slice(&env, &Tensor::shape1(3), &b0)?,
  );

  // e = ((a+b)*b)^2
  let c = a.add(&b)?; // a + b
  let d = c.mul(&b)?; // (a+b) * b
  let e = d.powf(2.0)?; // ((a+b) * b)^2

  // L = sum(e) に対する逆伝播（終端勾配=1 を投入）
  backward(&e)?;

  // grad をホストへ
  let ga = read_grad(&a)?;
  let gb = read_grad(&b)?;
  println!("grad(a) = {:?}", ga);
  println!("grad(b) = {:?}", gb);

  // --- 数値微分で検証 ---
  let eps = 1e-3f32;
  let l0 = host_loss(&a0, &b0);
  let mut num_ga = [0.0; 3];
  let mut num_gb = [0.0; 3];
  for i in 0..3 {
    let mut ap = a0;
    ap[i] += eps;
    let lp = host_loss(&ap, &b0);
    let mut am = a0;
    am[i] -= eps;
    let lm = host_loss(&am, &b0);
    num_ga[i] = (lp - lm) / (2.0 * eps);

    let mut bp = b0;
    bp[i] += eps;
    let lp2 = host_loss(&a0, &bp);
    let mut bm = b0;
    bm[i] -= eps;
    let lm2 = host_loss(&a0, &bm);
    num_gb[i] = (lp2 - lm2) / (2.0 * eps);
  }
  println!("num_grad(a) = {:?}", num_ga);
  println!("num_grad(b) = {:?}", num_gb);

  let max_err_a = max_abs_diff(&ga, &num_ga);
  let max_err_b = max_abs_diff(&gb, &num_gb);
  println!("max |grad(a)-num| = {}", max_err_a);
  println!("max |grad(b)-num| = {}", max_err_b);

  // 簡易アサート（閾値は素朴GEMM/単純カーネルなので緩め）
  assert!(max_err_a < 2e-2, "grad(a) check failed");
  assert!(max_err_b < 2e-2, "grad(b) check failed");

  // dot（形状・実行のみ検証）
  let A = Node::new_input(
    &tape,
    "A",
    Tensor::from_slice(&env, &Tensor::shape2(2, 3), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?,
  );
  let B = Node::new_input(
    &tape,
    "B",
    Tensor::from_slice(&env, &Tensor::shape2(3, 2), &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0])?,
  );
  let C = A.dot(&B)?;
  backward(&C)?;
  println!("dot backward executed (shape/flow OK).");

  Ok(())
}

/// ======= 検証用ユーティリティ（CPU側の L 計算：L = sum(((a+b)*b)^2)） =======
fn host_loss(a: &[f32; 3], b: &[f32; 3]) -> f32 {
  let mut s = 0.0;
  for i in 0..3 {
    let v = (a[i] + b[i]) * b[i];
    s += v * v;
  }
  s
}
fn max_abs_diff(a: &[f32], b: &[f32; 3]) -> f32 {
  let mut m = 0.0;
  for i in 0..3 {
    m = m.max((a[i] - b[i]).abs());
  }
  m
}
fn read_grad(n: &NodeRef) -> Result<Vec<f32>> {
  let g = n
    .borrow()
    .grad
    .as_ref()
    .ok_or_else(|| anyhow!("grad missing"))?;
  let mut host = vec![0.0f32; g.len];
  g.buf.read(&mut host).enq()?;
  Ok(host)
}
