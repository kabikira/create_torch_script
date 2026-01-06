# EfficientPhys 調査レポート（このディレクトリ内の論文PDF＋コード参照）

本レポートは、`EfficientPhys/` 配下に置かれている以下を根拠にまとめています。

- 論文PDF:  
  - `EfficientPhys/EfficientPhys-main/ICLR_EfficientPhys_Final.pdf`  
  - `EfficientPhys/Liu_EfficientPhys_Enabling_Simple_Fast_and_Accurate_Camera-Based_Cardiac_Measurement_WACV_2023_paper.pdf`
- コード: `EfficientPhys/EfficientPhys-main/*.py`

また、**この `EfficientPhys/EfficientPhys-main/` には「動画から顔を検出して切り出す」前処理コードが含まれていない**ため、質問項目のうち「顔検出方法」「フレームレートに応じた顔検出頻度」については、**このディレクトリ内の一次情報だけでは特定できない**点があります（後述）。

---

## 1. EfficientPhys はどのようなものか（概要）

EfficientPhys は、カメラ映像（主に顔映像）から心拍などの生体信号（rPPG/BVP）を推定するための、**前処理を極力不要にした end-to-end なニューラルネットワーク**です。従来は顔検出・ランドマーク抽出・ROI分割・色空間変換などの前処理が実装/計算コストのボトルネックになりやすいのに対し、EfficientPhys はそれらを不要化して**実装と実行を簡素化し、高フレームレートでの動作やオンデバイス実行を目指す**、という位置づけです。

### 原文（論文）

> models for camera-based physiological measurement called EfﬁcientPhys that  
> remove the need for face detection, segmentation, normalization, color space  
> transformation or any other preprocessing steps. Using an input of raw video  
> frames, our models achieve state-of-the-art accuracy on three public datasets.

出典: `EfficientPhys/EfficientPhys-main/ICLR_EfficientPhys_Final.pdf` p.1（Abstract）

> We propose a truly end-to-end network, EfficientPhys,  
> for which the input is unprocessed video frames without  
> requiring accurate face cropping (see Fig. 1).

出典: `EfficientPhys/Liu_EfficientPhys_Enabling_Simple_Fast_and_Accurate_Camera-Based_Cardiac_Measurement_WACV_2023_paper.pdf` p.2

---

## 2. 入力テンソル（モデルに入るテンソル形状・チャネル）

### 2.1 論文が示す「入力」の考え方（概念）

WACV論文中の図中ラベルでは、畳み込み版・Transformer版ともに入力が **`Nx72x72x3`（RGB）**として示されています（`N` はフレーム数/バッチ方向の表記）。

### 原文（論文図中ラベル）

> Nx72x72x3

出典: `EfficientPhys/Liu_EfficientPhys_Enabling_Simple_Fast_and_Accurate_Camera-Based_Cardiac_Measurement_WACV_2023_paper.pdf` p.4（図中ラベル）

また、ICLR版のAbstractでは「raw video frames」を入力とする、と明記されています。

> Using an input of raw video  
> frames, our models achieve state-of-the-art accuracy on three public datasets.

出典: `EfficientPhys/EfficientPhys-main/ICLR_EfficientPhys_Final.pdf` p.1（Abstract）

### 2.2 このディレクトリのコードが想定する入力テンソル（実装）

このディレクトリ内の学習/評価コードは、動画ファイルそのものではなく、**`dXsub`（入力）と `dysub`（教師波形）を含む `.mat` / `.h5` を読み込む**実装になっています。

- `dXsub` は、チャネル数が **3（diffのみ or rawのみ）** または **6（diff+raw）**のテンソルとして扱われています。  
- `img_size` が入力の空間解像度（`H=W`）として使われ、既定は `36` です（ただし論文では `72×72` にリサイズする記述あり。後述）。

#### 原文（コード: `dXsub` とチャネル構成）

`EfficientPhys/EfficientPhys-main/data_generator.py:33` 付近（`dXsub` の読み込み・整形）

```python
if f1["dXsub"].shape[0] == 6:
    dXsub = np.transpose(np.array(f1["dXsub"]), [3, 0, 2, 1])
...
```

`EfficientPhys/EfficientPhys-main/data_generator.py:69` 付近（`both/diff/raw` の分岐と 6ch=diff+raw の扱い）

```python
if self.input_channel == 'both':
    motion_data = output[:, :3, :, :]
    apperance_data = output[:, 3:, :, :]
    ...
    output = np.concatenate((motion_data, apperance_data), axis=1)
elif self.input_channel == 'diff':
    output = output[:, :3, :, :]
elif self.input_channel == 'raw':
    output = output[:, 3:, :, :]
```

#### 原文（コード: 学習時にモデルへ渡すテンソル形状）

`EfficientPhys/EfficientPhys-main/train_txt.py:143` 付近（`in_chans`=3/6 の決定）

```python
if args.swin_input_channel == 'diff' or args.swin_input_channel == 'raw':
    in_chans = 3
else:
    in_chans = 6
```

`EfficientPhys/EfficientPhys-main/train_txt.py:224` 付近（畳み込み系モデルへ渡す直前の reshape）

```python
inputs = inputs.view(-1, in_chans, args.img_size, args.img_size)
```

この形（`(N, C, H, W)`）で与える前提になっているのは、Temporal Shift Module (TSM) が **先頭次元 `nt` を “時間×バッチ” として解釈して `n_segment=frame_depth` に再整形**する実装になっているためです。

`EfficientPhys/EfficientPhys-main/model.py:25` 付近（TSM の入力解釈）

```python
nt, c, h, w = x.size()
n_batch = nt // self.n_segment
x = x.view(n_batch, self.n_segment, c, h, w)
```

---

## 3. 顔クロップサイズ（= 入力解像度/ROIサイズ）

### 3.1 論文の記述

WACV論文では「正確な face cropping を要求しない」としつつ、実験で用いる入力について「72×72 にリサイズ」と明記しています。  
このため、少なくとも論文記述上は **最終的にモデルへ入る空間解像度は 72×72**です（“顔を必ずクロップしているか” 自体は、この一文だけでは断定できません）。

#### 原文（“face cropping を要求しない”）

> ... input is unprocessed video frames without  
> requiring accurate face cropping ...

出典: `EfficientPhys/Liu_EfficientPhys_Enabling_Simple_Fast_and_Accurate_Camera-Based_Cardiac_Measurement_WACV_2023_paper.pdf` p.2

#### 原文（72×72にリサイズ）

> All the video data  
> are resized into a resolution of 72×72 with resampling using  
> pixel area relation (cv2.INTER AREA).

出典: `EfficientPhys/Liu_EfficientPhys_Enabling_Simple_Fast_and_Accurate_Camera-Based_Cardiac_Measurement_WACV_2023_paper.pdf` p.5

### 3.2 コードの既定値・サポート範囲

このディレクトリ内の学習スクリプト既定は `img_size=36` ですが（`train_txt.py`）、評価スクリプト既定は `img_size=72` で、さらにモデル実装は `36/72/96` を明示的に分岐してサポートしています。

#### 原文（コード: 既定の img_size）

- 学習: `EfficientPhys/EfficientPhys-main/train_txt.py:77`
  ```python
  parser.add_argument('--img_size', type=int, default=36)
  ```
- 評価: `EfficientPhys/EfficientPhys-main/eval_only_loop.py:78`
  ```python
  parser.add_argument('--img_size', type=int, default=72)
  ```

#### 原文（コード: サポートする画像サイズ）

`EfficientPhys/EfficientPhys-main/model.py:84` 付近（`36/72/96` 以外は例外）

```python
if img_size == 36:
    self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
elif img_size == 72:
    self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
elif img_size == 96:
    self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
else:
    raise Exception('Unsupported image size')
```

また、評価スクリプトのテキストファイル名に `72x72` や `nocrop` が含まれており、**「クロップあり/なし」両方のデータバリアントを想定**していることが読み取れます（ただし、その生成方法の実装はこのディレクトリにはありません）。

#### 原文（コード: `Crop` / `NoCrop` の切り替え）

`EfficientPhys/EfficientPhys-main/eval_only_loop.py:97` 付近

```python
if args.test_file == 'UBFCPPV2Crop':
    args.test_txt = './filelists/UBFC/UBFC_72_all_video_PPV2.txt'
elif args.test_file == 'UBFCPPV2NoCrop':
    args.test_txt = './filelists/UBFC/UBFC_72_all_video_PPV2_nocrop.txt'
...
elif args.test_file == 'UBFCNoCrop':
    args.test_txt = './filelists/UBFC/UBFC_72_all_video_nocrop.txt'
```

---

## 4. フレームレートによる顔検出（= fps に応じて顔検出頻度を変えるか？）

### 結論（このディレクトリ内の一次情報ベース）

- **論文（少なくとも ICLR版Abstract）では「face detection を不要化する」と明記**されており、fpsに応じて顔検出頻度を変える、という前提そのものを置いていません。  
- **この `EfficientPhys-main` のPythonコードにも顔検出処理は含まれていません**（`cv2` 等の顔検出器呼び出し、ランドマーク推定、トラッキング等が見当たりません）。  
- `--fs` は “顔検出の頻度” ではなく、**信号処理（バンドパス等）に使うサンプリング周波数**として使われています。

### 原文（論文: face detection 不要）

> remove the need for face detection ...

出典: `EfficientPhys/EfficientPhys-main/ICLR_EfficientPhys_Final.pdf` p.1（Abstract）

### 原文（コード: `fs` の用途はバンドパス等）

`EfficientPhys/EfficientPhys-main/train_txt.py:92`

```python
[b, a] = butter(1, [0.75 / args.fs * 2, 2.5 / args.fs * 2], btype='bandpass')
```

`EfficientPhys/EfficientPhys-main/eval_only_loop.py:105`（MMSE のみ `fs=25` に変更）

```python
elif args.test_file == 'MMSEPPV2':
    args.test_txt = './filelists/MMSE/MMSE72x72PPV2.txt'
    args.fs = 25
```

---

## 5. 顔検出方法（どの検出器を使うか？）

### 結論（このディレクトリ内の一次情報ベース）

- 論文は「顔検出を不要化する」ことを主張しているため、EfficientPhys自体に **顔検出器（MTCNN/RetinaFace/MediaPipe等）を前提とした記述が見当たりません**。  
- このディレクトリ内コードは **`dXsub` として既に整形済みの入力テンソルを読み込むだけ**で、元動画から ROI を切り出す処理がありません。  
  - そのため、「顔検出方法（アルゴリズム名）」「顔検出を何フレームごとに実行するか」は、**このディレクトリ内からは特定できません**。

参考として、`data_generator.py` では `generate_scaled_face` を呼ぶ分岐があり（`background_aug` が必要）、入力が “face” として扱われる前提は読み取れますが、その実装ファイル自体が `EfficientPhys-main/` 内に存在しません。

#### 原文（コード: 背景合成/顔スケール変更らしき呼び出し）

`EfficientPhys/EfficientPhys-main/data_generator.py:8` と `EfficientPhys/EfficientPhys-main/data_generator.py:46`

```python
from background_aug import generate_scaled_face
...
dXsub = generate_scaled_face(dXsub, self.img_size, self.img_size, target_dim_list=[32, 40, 48, 56, 64,72],
                             background_path=self.background_path)
```

---

## 6. まとめ（質問項目ごとの要点）

- **EfficientPhysとは**: 前処理（face detection 等）を不要化し、raw video frames を入力にして生体信号推定を行う end-to-end モデル（Conv版/Transformer版）  
  - 根拠: `EfficientPhys/EfficientPhys-main/ICLR_EfficientPhys_Final.pdf` p.1、`EfficientPhys/Liu_EfficientPhys_Enabling_Simple_Fast_and_Accurate_Camera-Based_Cardiac_Measurement_WACV_2023_paper.pdf` p.2
- **入力テンソル**: 論文図では `Nx72x72x3`。コード上は `.mat/.h5` の `dXsub` を読み、`diff/raw/both` で `C=3/6`、学習時は `view(-1, C, H, W)` にして投入  
  - 根拠: `EfficientPhys/Liu_EfficientPhys_Enabling_Simple_Fast_and_Accurate_Camera-Based_Cardiac_Measurement_WACV_2023_paper.pdf` p.4、`EfficientPhys/EfficientPhys-main/data_generator.py:69`、`EfficientPhys/EfficientPhys-main/train_txt.py:224`
- **顔クロップサイズ**: 論文は「正確な face cropping 不要」とし、入力は 72×72 にリサイズと記述。コードは `img_size` で 36/72/96 を扱う  
  - 根拠: WACV p.2, p.5、`EfficientPhys/EfficientPhys-main/train_txt.py:77`、`EfficientPhys/EfficientPhys-main/model.py:84`
- **フレームレートによる顔検出**: 論文・このディレクトリのコードでは “顔検出” 自体を前提にしない/実装がないため、fps連動の検出頻度は特定不可。`fs` は信号処理のサンプリング周波数  
  - 根拠: ICLR p.1、`EfficientPhys/EfficientPhys-main/train_txt.py:92`
- **顔検出方法**: このディレクトリ内のコードには顔検出器が登場せず、方法は特定不可（論文主張は “不要”）  
  - 根拠: ICLR p.1、`EfficientPhys/EfficientPhys-main/*.py` を検索しても顔検出実装が見当たらない

