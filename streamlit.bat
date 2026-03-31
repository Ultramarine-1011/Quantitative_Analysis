@echo off
setlocal
REM 使用 PATH 中的 Python 解释器，避免硬编码本机路径。
python -m streamlit %*
