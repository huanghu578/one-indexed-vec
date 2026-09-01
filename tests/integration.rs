//! `one-indexed-vec` 集成测试：从外部使用者视角验证 1-based 语义。

use one_indexed_vec::{Index1, VecIndexFromOne};

#[test]
fn end_to_end_one_based_workflow() {
    // 模拟一个"第 1 项、第 2 项……"的场景
    let mut scores = VecIndexFromOne::new();
    scores.push(95);
    scores.push(87);
    scores.push(92);

    assert_eq!(scores[1], 95);
    assert_eq!(scores[3], 92);
    assert_eq!(scores.len(), 3);

    // 更新第 2 项
    scores[2] = 90;
    assert_eq!(scores.get(2), Some(&90));

    // 删除第 1 项后，后续元素前移但仍从 1 编号
    let removed = scores.remove(1);
    assert_eq!(removed, 95);
    assert_eq!(scores[1], 90);
    assert_eq!(scores[2], 92);
}

#[test]
fn collection_interop() {
    // 从迭代器构建
    let v: VecIndexFromOne<String> = (0..5).map(|i| format!("item-{i}")).collect();
    assert_eq!(v.len(), 5);
    assert_eq!(v[5], "item-4");

    // extend 批量追加
    let mut w = VecIndexFromOne::from(vec![1, 2]);
    w.extend([3, 4]);
    assert_eq!(w[1], 1);
    assert_eq!(w[4], 4);

    // 转回标准 Vec，恢复 0-based 视角
    let plain: Vec<i32> = w.into();
    assert_eq!(plain, vec![1, 2, 3, 4]);
}

#[test]
fn generic_code_works_with_one_based_container() {
    // 泛型函数：对任意可求和容器求和（演示 VecIndexFromOne 可自由参与泛型）
    fn total<I>(iter: I) -> i64
    where
        I: IntoIterator<Item = i64>,
    {
        iter.into_iter().sum()
    }

    let v = VecIndexFromOne::from(vec![10, 20, 30]);
    assert_eq!(total(v.iter().copied()), 60);
    assert_eq!(total(v.iter_indexed().map(|(_, x)| *x)), 60);
}

#[test]
fn index_zero_and_oob_are_rejected_safely() {
    let v = VecIndexFromOne::from(vec![1, 2, 3]);

    // 安全 API：返回 None
    assert_eq!(v.get(0), None);
    assert_eq!(v.get(4), None);

    // 下标语法：panic 并给出清晰的 1-based 范围提示
    let zero = std::panic::catch_unwind(|| v[0]);
    assert!(zero.is_err());

    let oob = std::panic::catch_unwind(|| v[9]);
    assert!(oob.is_err());
}

#[test]
fn one_based_enumeration_matches_positions() {
    let v = VecIndexFromOne::from(vec!["a", "b", "c"]);
    let positions: Vec<(usize, &str)> =
        v.iter_indexed().map(|(i, s)| (i, *s)).collect();
    assert_eq!(positions, vec![(1, "a"), (2, "b"), (3, "c")]);

    // indices() 返回 1..=len
    let all: Vec<usize> = v.indices().collect();
    assert_eq!(all, vec![1, 2, 3]);
}

#[test]
fn type_safe_index1_usage() {
    let mut v = VecIndexFromOne::from(vec![10, 20, 30]);

    // 用 Index1 类型安全索引
    let second = Index1::new(2);
    assert_eq!(v[second], 20);
    v[second] = 200;
    assert_eq!(v[second], 200);

    // Index1 与 usize 双向换算
    assert_eq!(second.get(), 2);
    assert_eq!(second.to_zero_based(), Some(1));
    let raw: usize = second.into();
    assert_eq!(raw, 2);

    // 安全 API
    assert_eq!(v.get_index1(Index1::new(1)), Some(&10));
    assert_eq!(v.get_index1(Index1::new(0)), None);
}

#[test]
fn numeric_algorithms_from_outside() {
    let v = VecIndexFromOne::from(vec![1, 2, 3, 4]);

    let ps = v.prefix_sum();
    assert_eq!(ps.as_slice(), &[1, 3, 6, 10]);
    assert_eq!(v.sum(), 10);

    let w = VecIndexFromOne::from(vec![2, 2, 2, 2]);
    assert_eq!(v.dot_product(&w), 20);

    let neg = VecIndexFromOne::from(vec![-1, -2]);
    assert!(neg.all_non_positive());
    assert!(!v.any_non_positive());
}
