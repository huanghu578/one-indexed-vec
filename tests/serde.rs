//! serde 序列化测试（仅在启用 `serde` feature 时编译，见 Cargo.toml `[[test]]`）。

use one_indexed_vec::{Index1, VecIndexFromOne};

#[test]
fn serde_roundtrip_preserves_elements() {
    let v = VecIndexFromOne::from(vec![1, 2, 3]);

    let json = serde_json::to_string(&v).expect("serialize");
    assert_eq!(json, "[1,2,3]");

    let back: VecIndexFromOne<i32> = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.len(), 3);
    assert_eq!(back[1], 1);
    assert_eq!(back[3], 3);
}

#[test]
fn serde_roundtrip_index1() {
    let i = Index1::new(7);
    let json = serde_json::to_string(&i).expect("serialize");
    assert_eq!(json, "7");
    let back: Index1 = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, Index1::new(7));
}

#[test]
fn serde_roundtrip_composite_struct() {
    #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
    struct Holder {
        items: VecIndexFromOne<String>,
        pos: Index1,
    }

    let h = Holder {
        items: VecIndexFromOne::from(vec!["a".into(), "b".into()]),
        pos: Index1::new(2),
    };

    let json = serde_json::to_string(&h).expect("serialize");
    let back: Holder = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(h, back);
}
