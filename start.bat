@echo off
:: 获取当前批处理脚本所在的目录
cd /d %~dp0

:: 激活 conda 环境
call conda activate poserate

:: 运行 python 脚本
python api.py

:: 如果脚本运行完后保持命令行窗口打开
pause
