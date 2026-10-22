#!/bin/bash
set -e
cd "$(dirname "$0")"
echo '企业知识检索演示：下载离线依赖（不会上传文档）'
PYTHON="$(command -v python3.12 || true)"
if [ -z "$PYTHON" ]; then
  echo '需要 Python 3.12。请先安装 Python 3.12 的 macOS universal2 版本。'
  read -r -p '按回车退出…'
  exit 1
fi
"$PYTHON" -m venv .download-env
mkdir -p offline_packages
if .download-env/bin/python -m pip download --only-binary=:all: --dest offline_packages -r requirements.txt --timeout 120 --retries 5; then
  echo ''
  echo '下载完成！请将本项目里的 offline_packages 文件夹提供给 KlingWork。'
else
  echo ''
  echo '下载未完成。已下载的包会保留；可再次双击此文件继续。'
  echo '若仍失败，请将失败信息发给 KlingWork，不要自行换未知软件源。'
fi
read -r -p '按回车关闭…'
