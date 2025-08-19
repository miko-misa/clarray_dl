use num_traits::Float;
use ocl::{CommandQueueProperties, OclPrm, ProQue, Program};
use once_cell::sync::OnceCell;
use std::{
  any::TypeId,
  collections::HashMap,
  sync::{Arc, RwLock},
};

// ===== デフォルト .cl をハードコーディング =====
// パスはこのファイルからの相対。必要に応じて差し替えてください。
static DEFAULT_CLS: &[&str] = &[include_str!("kernels/ops.cl")];

// ===== 型ごとのヘッダ注入（ビルドオプションなし）=====
pub trait DTypeSpec {
  const HEADER: &'static str;
  fn preflight(_: &ocl::Device) -> ocl::Result<()> {
    Ok(())
  }
}
impl DTypeSpec for f32 {
  const HEADER: &'static str = r#"
        typedef float scalar_t;
    "#;
}
impl DTypeSpec for f64 {
  const HEADER: &'static str = r#"
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        typedef double scalar_t;
    "#;
  fn preflight(dev: &ocl::Device) -> ocl::Result<()> {
    let exts = dev.info(ocl::enums::DeviceInfo::Extensions)?.to_string();
    if !exts.contains("cl_khr_fp64") {
      Err("Device does not support cl_khr_fp64 (f64)".into())
    } else {
      Ok(())
    }
  }
}

// ===== 環境 =====
pub struct Env {
  proque: ProQue,
  // デフォルトは不変で常に読み込む
  defaults: &'static [&'static str],
  // 途中追加される .cl（追加のたびにキャッシュを無効化）
  extras: RwLock<Vec<Arc<str>>>,
  // 型ごとの Program キャッシュ
  cache: RwLock<HashMap<TypeId, Arc<Program>>>,
}

impl Env {
  fn new() -> Self {
    let props_ooo = CommandQueueProperties::new().out_of_order().profiling();

    let proque = ProQue::builder()
      .src("")
      .dims(1)
      .queue_properties(props_ooo)
      .build()
      .unwrap_or_else(|_| {
        ProQue::builder()
          .src("")
          .dims(1)
          .queue_properties(CommandQueueProperties::new().profiling())
          .build()
          .expect("ProQue init failed")
      });

    Self {
      proque,
      defaults: DEFAULT_CLS,
      extras: RwLock::new(Vec::new()),
      cache: RwLock::new(HashMap::new()),
    }
  }

  pub fn proque(&self) -> &ProQue {
    &self.proque
  }

  /// .cl を途中から追加（複数まとめも可）。追加後の最初の get_program で再ビルド。
  pub fn add_cl<S: Into<Arc<str>>>(&self, src: S) {
    self.add_cls([src]);
  }
  pub fn add_cls<I, S>(&self, srcs: I)
  where
    I: IntoIterator<Item = S>,
    S: Into<Arc<str>>,
  {
    let mut w = self.extras.write().unwrap();
    for s in srcs {
      w.push(s.into());
    }
    // 新ソースを反映させるためキャッシュをクリア
    self.cache.write().unwrap().clear();
  }

  /// 型に応じた Program を取得（初回のみビルド＆キャッシュ）
  pub fn get_program<T>(&self) -> ocl::Result<Arc<Program>>
  where
    T: OclPrm + Float + DTypeSpec + 'static,
  {
    let tid = TypeId::of::<T>();
    if let Some(p) = self.cache.read().unwrap().get(&tid) {
      return Ok(p.clone());
    }

    let device = self.proque.queue().device();
    T::preflight(&device)?; // 例：f64 サポート確認

    // ヘッダ + デフォルト + 追加 を連結
    let mut full = String::new();
    full.push_str(T::HEADER);
    for s in self.defaults {
      full.push('\n');
      full.push_str(s);
    }
    for s in self.extras.read().unwrap().iter() {
      full.push('\n');
      full.push_str(s);
    }

    let program = Program::builder()
      .src(full)
      .devices(device)
      .build(self.proque.context())?;

    let arc = Arc::new(program);
    let mut w = self.cache.write().unwrap();
    Ok(w.entry(tid).or_insert_with(|| arc.clone()).clone())
  }
}

// ===== グローバル =====
static GLOBAL: OnceCell<Env> = OnceCell::new();

pub fn init() -> &'static Env {
  GLOBAL.get_or_init(|| Env::new())
}

pub fn env() -> &'static Env {
  // まだなら init() を呼ぶ。初期化済みならそのまま返す。
  if GLOBAL.get().is_none() {
    let _ = init();
  }
  GLOBAL.get().unwrap()
  // もっと短くするなら↓でも同じ意味
  // init()
}
