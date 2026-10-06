"""把 MedAD/Brain_AD 改造成脑部管线要求的目录结构（只复制 2D 切片 PNG）。

源结构（Windows 上不区分大小写，Ungood/ungood 是同一个目录）:
    Brain_AD/
      train/good/            *.png                   7500 张，全部正常（424 个病例）
      valid/good/{img,label}  39 + 39                正常切片
      valid/Ungood/{img,label} 44 + 44               异常切片 + 病灶掩码
      test/good/{img,label}   640 + 640              正常切片
      test/Ungood/{img,label} 3075 + 3075            异常切片 + 病灶掩码

目标结构（text_side_anomaly 期望的 normal/abnormal + 同名掩码）:
    brain_mri/{train,val,test}/{normal,abnormal}/*.png
    brain_mri_masks/{train,val,test}/abnormal/*.png

划分口径（可用参数调整）:
    train 池 = 正常(train/good + valid/good) + 异常(valid/Ungood)
    test     = test 全集（640 正常 + 3075 异常，异常都有掩码）
    val      = 从池子里按病例各留出 --val-cases 个正常/异常病例，用于选阈值
               （不建 val 时只有 AUROC/AP 这类不用阈值的指标）

已知数据问题（本脚本会打印出来）:
    BMAD 的 valid 与 test 有 10 个异常病例重叠，其中每个病例各有 1 张切片与
    test 中同名文件逐字节相同（md5 一致）。默认会把那 10 个重复文件从训练池
    剔除（--keep-duplicates 可关掉）；病例级重叠无法在不掏空异常池的前提下
    消除（剔掉这 10 个病例后异常池只剩患者 00803 的 4 张）。

用法:
    python prepare_brain_mri_slices.py --dry-run          # 只清点，不复制
    python prepare_brain_mri_slices.py                    # 正式复制
"""

import argparse
import hashlib
import os
import shutil
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

PNG_EXT = ".png"
DEFAULT_SRC = r"D:\图神经网络\小样本原型学习\AA-CLIP\data\MedAD\Brain_AD"


def _files(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return sorted(f for f in os.listdir(path) if f.lower().endswith(PNG_EXT))


def _img_dir(split_dir: str) -> str:
    inner = os.path.join(split_dir, "img")
    return inner if os.path.isdir(inner) else split_dir


def _label_dir(split_dir: str) -> str:
    return os.path.join(split_dir, "label")


def _case_of(name: str) -> str:
    """BMAD 命名 00124_60.png：下划线前是病例号。"""
    return os.path.basename(name).split("_")[0]


def _md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(src: str, dst_dir: str, dry: bool) -> bool:
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    if os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return False
    if not dry:
        shutil.copy2(src, dst)
    return True


def _pair(img_dir: str, label_dir: str) -> List[Tuple[str, Optional[str]]]:
    """图像与同名掩码配对；掩码缺失记 None（异常样本缺掩码会在训练前报错）。"""
    rows = []
    for name in _files(img_dir):
        mask = os.path.join(label_dir, name)
        rows.append((os.path.join(img_dir, name), mask if os.path.isfile(mask) else None))
    return rows


def build_plan(args) -> Dict[str, Dict[str, List[Tuple[str, Optional[str]]]]]:
    src = args.src
    plan: Dict[str, Dict[str, List[Tuple[str, Optional[str]]]]] = {
        name: {"normal": [], "abnormal": [], "masks": []} for name in ("train", "val", "test")
    }

    # ---- test：官方测试集，原样搬 ----
    test_good_img = _img_dir(os.path.join(src, "test", "good"))
    plan["test"]["normal"] = [(os.path.join(test_good_img, name), None)
                              for name in _files(test_good_img)]
    plan["test"]["abnormal"] = _pair(_img_dir(os.path.join(src, "test", "Ungood")),
                                     _label_dir(os.path.join(src, "test", "Ungood")))

    # ---- 训练池：正常取自 train/good + valid/good，异常取自 valid/Ungood ----
    normal_rows = _pair(_img_dir(os.path.join(src, "valid", "good")),
                        _label_dir(os.path.join(src, "valid", "good")))
    if args.pool_normal == "train+valid":
        normal_rows += _pair(_img_dir(os.path.join(src, "train", "good")),
                             _label_dir(os.path.join(src, "train", "good")))
    abnormal_rows = _pair(_img_dir(os.path.join(src, "valid", "Ungood")),
                          _label_dir(os.path.join(src, "valid", "Ungood")))

    # ---- 剔除与 test 逐字节相同的切片（同名 + 同 md5）----
    test_names = {os.path.basename(p) for p, _ in plan["test"]["abnormal"]}
    test_md5 = {os.path.basename(p): _md5(p) for p, _ in plan["test"]["abnormal"]}
    dropped_dup = []
    if not args.keep_duplicates:
        kept = []
        for path, mask in abnormal_rows:
            name = os.path.basename(path)
            if name in test_names and _md5(path) == test_md5[name]:
                dropped_dup.append(name)
                continue
            kept.append((path, mask))
        abnormal_rows = kept

    if args.strict_cases:
        test_cases = {_case_of(p) for p, _ in plan["test"]["abnormal"]} | \
                     {_case_of(p) for p, _ in plan["test"]["normal"]}
        abnormal_rows = [(p, m) for p, m in abnormal_rows if _case_of(p) not in test_cases]
        normal_rows = [(p, m) for p, m in normal_rows if _case_of(p) not in test_cases]

    # ---- val：按病例各留出 N 个正常 / N 个异常（异常优先挑不在 test 里的病例）----
    def _split_by_case(rows, val_cases: int, prefer_outside_test: bool):
        by_case: "OrderedDict[str, List[Tuple[str, Optional[str]]]]" = OrderedDict()
        for row in sorted(rows, key=lambda r: os.path.basename(r[0])):
            by_case.setdefault(_case_of(row[0]), []).append(row)
        cases = sorted(by_case)
        if prefer_outside_test:
            test_cases = {_case_of(p) for p, _ in plan["test"]["abnormal"]}
            cases.sort(key=lambda c: (c in test_cases, c))
        if val_cases <= 0:
            return rows, []
        val_cases = min(val_cases, max(0, len(cases) - 1))
        val_case_ids = set(cases[:val_cases])
        val_rows = [r for c in cases if c in val_case_ids for r in by_case[c]]
        train_rows = [r for c in cases if c not in val_case_ids for r in by_case[c]]
        return train_rows, val_rows

    train_normal, val_normal = _split_by_case(normal_rows, args.val_cases, False)
    train_abnormal, val_abnormal = _split_by_case(abnormal_rows, args.val_cases, True)
    plan["train"]["normal"] = train_normal
    plan["train"]["abnormal"] = train_abnormal
    plan["val"]["normal"] = val_normal
    plan["val"]["abnormal"] = val_abnormal
    plan["val"]["masks"] = [(m, None) for _, m in val_abnormal if m]
    plan["test"]["masks"] = [(m, None) for _, m in plan["test"]["abnormal"] if m]
    plan["train"]["masks"] = [(m, None) for _, m in train_abnormal if m]
    plan["_meta"] = {"dropped_dup": dropped_dup}          # type: ignore[assignment]
    return plan


def _describe(plan, args) -> None:
    print("\n划分清点（train 是支持集池，K 从它里面抽）:")
    for split in ("train", "val", "test"):
        normal, abnormal = plan[split]["normal"], plan[split]["abnormal"]
        masks = [m for _, m in abnormal if m]
        cases = {_case_of(p) for p, _ in normal} | {_case_of(p) for p, _ in abnormal}
        print(f"  {split:<5} normal={len(normal):<5} abnormal={len(abnormal):<5} "
              f"掩码={len(masks):<5} 病例={len(cases)}")
    dropped = plan.get("_meta", {}).get("dropped_dup", [])
    if dropped:
        print(f"  已剔除与 test 逐字节相同的异常切片 {len(dropped)} 张：{dropped}")
    test_cases = {_case_of(p) for p, _ in plan["test"]["abnormal"]}
    shared = {_case_of(p) for p, _ in plan["train"]["abnormal"]} & test_cases
    if shared:
        print(f"  ⚠ 训练池 abnormal 与 test 仍有 {len(shared)} 个病例重叠：{sorted(shared)}")
        print("    BMAD 原始划分如此；不加 --case-id-regex 时管线会照跑并记录 "
              "case_disjoint_verified=false。")
        print("    要病例级干净用 --strict-cases（异常池会只剩患者 00803 的切片）。")
    if args.val_cases == 0:
        print("  未建 val：本次只报 AUROC/AP，F1/Dice 需要固定阈值或 --oracle-metrics。")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Brain_AD → brain_mri/{normal,abnormal} 结构转换")
    ap.add_argument("--src", default=DEFAULT_SRC, help="MedAD/Brain_AD 根目录；不存在就直接报错")
    ap.add_argument("--out", default="brain_mri")
    ap.add_argument("--mask-out", default="brain_mri_masks")
    ap.add_argument("--pool-normal", choices=["valid", "train+valid"], default="train+valid",
                    help="支持集池的正常切片来源（默认 train/good + valid/good）")
    ap.add_argument("--val-cases", type=int, default=2,
                    help="从池子里留出多少个正常/异常病例做 val；0 = 不建 val")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="保留与 test 逐字节相同的 10 张异常切片（默认剔除）")
    ap.add_argument("--strict-cases", action="store_true",
                    help="剔除所有与 test 病例重叠的病例（异常池只剩患者 00803）")
    ap.add_argument("--dry-run", action="store_true", help="只清点与校验，不复制文件")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        print(f"[错误] 源目录不存在：{args.src}", file=sys.stderr)
        return 1
    for sub in ("train", "valid", "test"):
        if not os.path.isdir(os.path.join(args.src, sub)):
            print(f"[错误] 源目录缺少 {sub}/：{args.src}", file=sys.stderr)
            return 1

    plan = build_plan(args)
    _describe(plan, args)
    if args.val_cases > 0 and not plan["val"]["abnormal"]:
        print("[提示] val 里没有异常切片，阈值选择会退化为不可用")

    missing = [(p, m) for p, m in plan["train"]["abnormal"] if not m]
    if missing:
        print(f"[错误] 训练池有 {len(missing)} 个异常切片没有掩码，训练会在建模前报错："
              f"{[os.path.basename(p) for p, _ in missing[:5]]}", file=sys.stderr)
        return 1
    missing = [(p, m) for p, m in plan["test"]["abnormal"] if not m]
    if missing:
        print(f"[错误] test 有 {len(missing)} 个异常切片没有掩码", file=sys.stderr)
        return 1

    copied = skipped = 0
    for split in ("train", "val", "test"):
        for label in ("normal", "abnormal"):
            for path, _ in plan[split][label]:
                if _copy(path, os.path.join(args.out, split, label), args.dry_run):
                    copied += 1
                else:
                    skipped += 1
        for path, _ in plan[split]["masks"]:
            if _copy(path, os.path.join(args.mask_out, split, "abnormal"), args.dry_run):
                copied += 1
            else:
                skipped += 1

    verb = "待复制" if args.dry_run else "已复制"
    print(f"\n[完成] {verb} {copied} 个文件，跳过（已存在且大小相同）{skipped} 个")
    print(f"  图像: {args.out}/{split} ...   掩码: {args.mask_out}/<split>/abnormal/")
    if args.dry_run:
        print("  这是 --dry-run，没有写任何文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
