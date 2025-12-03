import yfinance as yf
import torch
import numpy as np
import pandas as pd
import scipy

print("✅ yfinance:", yf.__version__)
print("✅ PyTorch:", torch.__version__)
print("✅ CUDA available:", torch.cuda.is_available())
print("✅ numpy:", np.__version__)
print("✅ pandas:", pd.__version__)
print("✅ scipy:", scipy.__version__)

# Test pobierania danych
data = yf.download("SPY", period="1mo")
print("✅ Yahoo Finance test:", data.shape)

# Test PyTorch
x = torch.randn(3, 3)
print("✅ PyTorch test:", x.shape)