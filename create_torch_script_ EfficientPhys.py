import argparse
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parent
    venv_python = repo_root / ".venv" / "bin" / "python"
    lines = [
        "PyTorch(torch) が見つかりません。",
        "このスクリプトは PyTorch がインストールされた Python で実行してください。",
    ]
    if venv_python.exists():
        lines += [
            "",
            "例（このリポジトリの .venv を使用）:",
            f"  {venv_python} \"{Path(__file__).name}\"",
        ]
    lines += [
        "",
        "セットアップ例:",
        "  python3.12 -m venv .venv",
        "  .venv/bin/python -m pip install torch numpy",
        f"  .venv/bin/python \"{Path(__file__).name}\"",
        "",
        "（補足）Homebrew の python3 が 3.13 の場合、torch が未対応でインストールできないことがあります。python3.12 を推奨します。",
    ]
    print("\n".join(lines), file=sys.stderr)
    raise SystemExit(1)


def _add_efficientphys_to_syspath() -> Path:
    repo_root = Path(__file__).resolve().parent
    efficientphys_src = repo_root / "EfficientPhys" / "EfficientPhys-main"
    if not efficientphys_src.exists():
        raise FileNotFoundError(f"EfficientPhys のコードが見つかりません: {efficientphys_src}")
    sys.path.insert(0, str(efficientphys_src))
    return efficientphys_src


def load_state_dict(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"想定外のチェックポイント形式: {type(ckpt)}")

    # よくあるキー名を優先的に取り出す
    for key in ["model", "state_dict", "ema_state_dict"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]
            break

    # DataParallel の 'module.' を除去
    return {k[7:] if k.startswith("module.") else k: v for k, v in ckpt.items()}


def infer_img_size_from_state_dict(state_dict: dict):
    weight = state_dict.get("final_dense_1.weight")
    if weight is None or not hasattr(weight, "shape") or len(weight.shape) != 2:
        return None
    in_features = int(weight.shape[1])
    return {3136: 36, 16384: 72, 30976: 96}.get(in_features)


def main() -> None:
    parser = argparse.ArgumentParser(description="EfficientPhys の .pth を TorchScript に変換します")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("EfficientPhys/UBFC-rPPG_EfficientPhys.pth"),
        help="変換するチェックポイント(.pth)へのパス",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="出力 TorchScript(.pt) のパス（未指定なら ckpt と同じ場所に .ts.pt を出力）",
    )
    parser.add_argument("--frame-depth", type=int, default=10, help="TSM の frame_depth（学習時と合わせる）")
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="入力画像サイズ（未指定なら ckpt から推定。学習時と合わせる）",
    )
    parser.add_argument(
        "--input-channel",
        type=str,
        default="diff",
        choices=["diff", "raw"],
        help="入力の種類（diff: 差分フレーム, raw: 生フレーム）",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="script",
        choices=["script", "trace"],
        help="TorchScript 化の方法（script 推奨。失敗時は trace を選択）",
    )
    args = parser.parse_args()

    ckpt_path: Path = args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"チェックポイントが見つかりません: {ckpt_path}")

    out_path = args.out if args.out is not None else ckpt_path.with_suffix(".ts.pt")

    _add_efficientphys_to_syspath()
    from model import EfficientPhys_Conv  # noqa: E402

    # 1) 重みを読み込む（img_size 推定のため先にロード）
    state_dict = load_state_dict(ckpt_path)
    img_size = args.img_size
    if img_size is None:
        img_size = infer_img_size_from_state_dict(state_dict)
        if img_size is None:
            raise ValueError("ckpt から img_size を推定できませんでした。--img-size を指定してください。")
        print(f"img_size を ckpt から推定しました: {img_size}")

    # 2) モデル作成（UBFC-rPPG の公開重みは EfficientPhys_Conv を想定）
    model = EfficientPhys_Conv(frame_depth=args.frame_depth, img_size=img_size, channel=args.input_channel)

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("警告: load_state_dict に不整合があります")
        if incompatible.missing_keys:
            print(f"  missing_keys: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"  unexpected_keys: {len(incompatible.unexpected_keys)}")

    model.eval()

    # 3) TorchScript へ変換
    with torch.no_grad():
        method = args.method
        scripted = None
        if method == "script":
            print("torch.jit.script を試行中...")
            try:
                scripted = torch.jit.script(model)
                print("torch.jit.script 成功!")
            except Exception as e:
                print(f"torch.jit.script 失敗: {e}")
                print("trace にフォールバックします...")
                method = "trace"

        if method == "trace":
            # TSM の都合で、raw の場合は diff 後に frame_depth になるよう frame_depth+1 を与える
            example_nt = args.frame_depth + 1 if args.input_channel == "raw" else args.frame_depth
            example_input = torch.randn(example_nt, 3, img_size, img_size)
            scripted = torch.jit.trace(model, example_input)
            print("torch.jit.trace 成功（入力形状は固定）")

    assert scripted is not None

    # 4) 保存
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.jit.save(scripted, out_path)
    except Exception as e:
        # 環境によっては script は保存時に失敗するケースがあるため、trace に切り替える
        if method == "script":
            print(f"torch.jit.save が失敗しました（script）: {e}")
            print("trace で再試行します...")
            example_nt = args.frame_depth + 1 if args.input_channel == "raw" else args.frame_depth
            example_input = torch.randn(example_nt, 3, img_size, img_size)
            traced = torch.jit.trace(model, example_input)
            torch.jit.save(traced, out_path)
        else:
            raise
    print(f"変換完了: {out_path.resolve()}")


if __name__ == "__main__":
    main()
