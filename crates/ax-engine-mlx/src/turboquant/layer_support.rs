#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TurboQuantLayerSupportReason {
    Eligible,
    LinearAttention,
    SlidingWindow,
    KvShared,
    UnsupportedHeadDim,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TurboQuantLayerSupport {
    pub layer_index: usize,
    pub head_dim: usize,
    pub reason: TurboQuantLayerSupportReason,
}

impl TurboQuantLayerSupport {
    pub fn is_eligible(self) -> bool {
        self.reason == TurboQuantLayerSupportReason::Eligible
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TurboQuantSupportReport {
    pub layers: Vec<TurboQuantLayerSupport>,
    pub eligible_layers: usize,
    pub linear_attention_layers: usize,
    pub sliding_window_layers: usize,
    pub kv_shared_layers: usize,
    pub unsupported_head_dim_layers: usize,
}

impl TurboQuantSupportReport {
    pub fn eligible_layer_mask(&self) -> Vec<bool> {
        self.layers
            .iter()
            .map(|layer| layer.is_eligible())
            .collect()
    }
}
