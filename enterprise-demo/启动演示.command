#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo '运行环境尚未安装，请联系 KlingWork 完成离线依赖安装。'
  read -r -p '按回车退出…'
  exit 1
fi
echo '企业知识检索演示：http://127.0.0.1:8765'
echo '请保持此窗口打开；关闭窗口或按 Control+C 停止。'
.venv/bin/python launcher.py
