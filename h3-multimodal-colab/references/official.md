# 官方模型管道（`model_set: "official"`）

使用者在 SKILL.md 第 0 節選了「官方模型」才讀這份。預設的客製化模型組不用看這裡。

## 兩組的差別：表現力與參考力的取捨

兩組不是好壞之分。

- **官方組＝參考力**：主模型是**完整的 Ref2VA／FL2VA 權重**，沒有經過混合，對參考照的服從度最高，**人物會比較像**。
- **預設組＝表現力**：社群合併主模型加上加速與風格 LoRA，權重經過混合，換來更有戲的動作、表情與運鏡，
  代價是參考力被稀釋一些（人物偶爾會往模型自己的臉靠）。

## 什麼時候選

- **人物要像**：真人講者、品牌代言人、同一個角色要在很多段裡保持一致。
- 要寫進**公開文件**、教學講義、機構或教會的專案：預設組的模型名稱（社群合併主模型、MysticXXX）不適合出現在那裡。
- 只想用 **Comfy-Org 發布的權重**。

要大動作、強烈表情、武打追逐，而且可以接受人物稍微不那麼像 → 用預設組。

## 怎麼寫

manifest 頂層加一行，整批的影片都走官方：

```json
{ "model_set": "official", "jobs": [ ... ] }
```

單一 job 也可以自己寫 `"model_set": "default"` 或 `"official"`，會覆蓋頂層。

## 換了什麼：權重跟著任務走

| 影片模式 | 主模型 | 加速 LoRA | flex LoRA |
|---|---|---|---|
| `ref`、`text`（Ref2VA／T2VA） | `minimax_h3_ref2va_pruned_int8_convrot` | `minimax_h3_ref2v_turbo_4step_v0.1` | `pika_h3_flex_ref2va_v1` |
| `frames`（I2VA／L2VA／FL2VA） | `minimax_h3_fl2va_pruned_int8_convrot` | `minimax_h3_fl2v_turbo_8step_v1.0` | `pika_h3_flex_fl2va_v1` |

- 全部來自 Hugging Face `Comfy-Org/MiniMax-H3`。
- **MysticXXX 整顆拿掉**（不是設成強度 0），所以只跑官方的批次不需要那個檔案。
- 頭尾格一定要用 FL2VA 這一組。拿 Ref2VA 的權重跑頭尾格，等於用錯任務的模型，`check_video` 會擋下來。

## 沒有換的（要誠實告訴使用者）

- 文字編碼器、影像 VAE、聲音 VAE：本來就是 Comfy-Org 的。
- **語意橋接（BUNNY ActionLogic Bridge）與 latent upscaler V2**：仍是社群權重（dseditor 鏡像），工作流的骨架需要它們。
- `intensity` 不是 `off` 時接上的動作 LoRA（Combat、Motion Repair、Speed slider）：仍是社群權重。
- **生圖（`image`）用獨立圖片模型**，不靠社群影片主模型。**配音（`tts`）沒有官方版**：批次裡有配音，就會照樣下載社群主模型。

## 下載量

| 批次內容 | 多下載什麼 |
|---|---|
| 只有官方 `ref`／`text` 影片 | Ref2VA 組約 23 GB，**不下載**社群主模型、社群 turbo、MysticXXX |
| 官方影片含 `frames` | 再加 FL2VA 組約 25 GB |
| 官方影片＋圖片 | 官方組＋圖片模型，**不下載**社群影片主模型 |
| 官方影片＋配音 | 官方組＋社群主模型都要下載 |

本地版：`h3local.py setup --official-only`（只裝官方組；生圖另加 `--with-pictures`，配音仍需社群主模型）或 `--with-official`（兩組都裝）。

## 實測（Colab L4，480P，同提示詞同 seed）

| 題材 | 官方組 vs 預設組 |
|---|---|
| 鎖音主持人（`ref`＋`drive_audio`，12 秒） | 差不多，聲音與原錄音一致 |
| 邊走邊講 | 官方走動較少、表情較誇張 |
| 武打（`ref`，12 秒） | 官方動態模糊、鏡頭旋轉較多 → 大動作是預設組的強項 |

說話、主持、介紹這類以人物為主的畫面，官方組很適合；動作量越大，越能看出預設組在表現力上的調校。

⚠️ Ref2VA 的官方加速 LoRA 是 **4 步版（v0.1）**，這個技能用 8 步（`fast`）／12 步在跑它，步數還沒有替官方組調過；
上面的動作模糊可能有一部分來自這裡。FL2VA 的官方加速 LoRA 是 8 步版，和 `fast` 對得上。

給使用者的一句話：「官方組用的是完整的參考權重，**人物會比較像參考照**，也適合公開使用；
預設組是為表現力調過的，動作和表情比較有戲。這是像不像和有沒有戲之間的取捨。生圖用獨立圖片模型；配音仍會用到社群模型。」
