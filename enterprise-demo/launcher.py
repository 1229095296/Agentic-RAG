"""Foreground desktop launcher; the terminal owns the service lifetime."""
import socket
import threading
import webbrowser


def main():
    probe = socket.socket()
    try:
        probe.bind(('127.0.0.1', 8765))
    except OSError:
        print('端口 8765 已占用。请关闭之前的演示窗口后再启动。')
        input('按回车退出…')
        return
    finally:
        probe.close()
    import uvicorn
    from app import app
    threading.Timer(2, lambda: webbrowser.open('http://127.0.0.1:8765')).start()
    uvicorn.run(app, host='127.0.0.1', port=8765)


if __name__ == '__main__':
    main()
