# このプロジェクトについて
書籍の翻訳をするコード

## 基本的な使用方法
data/inputに翻訳したいテキストをmdとして置こう。(ちゃんとpdfの文章がmdに取り込まれているか手動で確認すること)


## E2E翻訳（DOCX → DOCX）
DOCXファイルから翻訳済みDOCXファイルを直接作成する:

```bash
# 単一ファイルを翻訳
python e2e_translation.py -M gemini -F my_folder --input-file document.docx

# フォルダ内の全DOCXファイルを翻訳
python e2e_translation.py -M gpt -F my_folder --all-docx

# 翻訳後に中間ファイルを削除
python e2e_translation.py -M gemini -F my_folder --all-docx --cleanup
```

## 従来の方法（段階的実行）

### 1. ファイル形式変換
```bash
# DOCX → Markdown
python change_extention.py -M docx2md -F input/my_folder

# Markdown → DOCX
python change_extention.py -M md2docx -F output/my_folder
```

### 2. 翻訳実行
```bash
# Gemini使用
python ep_translation.py -M gemini -F my_folder -R 5

# GPT使用
python ep_translation.py -M gpt -F my_folder
```