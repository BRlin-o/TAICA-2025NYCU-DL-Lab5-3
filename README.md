# Lab6

## 環境設定

### 虛擬環境建立與啟用

使用 Python 虛擬環境可避免套件衝突。請根據作業平台依序執行以下步驟：

- **MacOS / Linux:**

    ```bash
    ## env create
    python -m venv .venv
    source ./.venv/bin/activate

    ## pip update
    pip install --upgrade pip

    ## Normal
    pip install -r requirements.txt
    ## Using CUDA
    pip install -r requirements_CUDA.txt
    ```

- **Windows**

    ```bash
    ## env create
    python -m venv .venv
    .\.venv\Scripts\activate.bat

    ## pip update
    pip install --upgrade pip

    ## Normal
    pip install -r requirements.txt
    ## Using CUDA
    pip install -r requirements_CUDA.txt
    ```

## 訓練模型

```bash
python main.py --mode train --batch_size 128 --epochs 100

python main.py --mode train --batch_size 64 --epochs 100
```

## 生成圖像

```bash
python main.py --mode sample
```

## 使用特定檢查點生成圖像

```bash
python main.py --mode sample --checkpoint path/to/checkpoint.pth
```