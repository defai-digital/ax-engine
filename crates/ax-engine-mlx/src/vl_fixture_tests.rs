//! Golden fixture driven by shipped VL geometry (WS-V1/V2).

#[cfg(test)]
mod tests {
    use crate::gemma4_vl::{Gemma4VlImageGeometry, plan_image_scatter};
    use crate::qwen3_vl::{Qwen3VlImageGeometry, deepstack_layers, plan_mrope_for_images};

    #[test]
    fn vl_geometry_golden_matches_shipped_functions() {
        let raw = include_str!("../tests/fixtures/vl_geometry_golden.json");
        let v: serde_json::Value = serde_json::from_str(raw).expect("parse golden");

        let g = &v["gemma4_vl"]["image"];
        let geom = Gemma4VlImageGeometry {
            height: g["h"].as_u64().unwrap() as u32,
            width: g["w"].as_u64().unwrap() as u32,
            patch_size: g["patch"].as_u64().unwrap() as u32,
            merge_size: g["merge"].as_u64().unwrap() as u32,
            max_soft_tokens: g["max"].as_u64().unwrap() as u32,
        };
        assert_eq!(
            geom.soft_token_count().unwrap(),
            v["gemma4_vl"]["soft_tokens"].as_u64().unwrap() as u32
        );
        let ph = v["gemma4_vl"]["scatter_placeholder"].as_u64().unwrap() as usize;
        let idx = plan_image_scatter(&[ph], &[geom]).unwrap();
        let fl = &v["gemma4_vl"]["scatter_first_last"];
        assert_eq!(idx[0], fl[0].as_u64().unwrap() as usize);
        assert_eq!(*idx.last().unwrap(), fl[1].as_u64().unwrap() as usize);

        let q = &v["qwen3_vl"]["image"];
        let qg = Qwen3VlImageGeometry {
            height: q["h"].as_u64().unwrap() as u32,
            width: q["w"].as_u64().unwrap() as u32,
            patch_size: q["patch"].as_u64().unwrap() as u32,
            spatial_merge_size: q["merge"].as_u64().unwrap() as u32,
            max_soft_tokens: q["max"].as_u64().unwrap() as u32,
        };
        assert_eq!(
            qg.soft_token_count().unwrap(),
            v["qwen3_vl"]["soft_tokens"].as_u64().unwrap() as u32
        );
        let mrope = plan_mrope_for_images(&[qg]).unwrap();
        assert_eq!(
            mrope.len(),
            v["qwen3_vl"]["mrope_len"].as_u64().unwrap() as usize
        );
        assert_eq!(
            deepstack_layers(3, 36),
            v["qwen3_vl"]["deepstack_layers"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect::<Vec<_>>()
        );
    }
}
