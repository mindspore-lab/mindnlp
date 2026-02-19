import socket
import struct
import threading
import time


_CMD_SET = 0
_CMD_GET = 1
_CMD_WAIT = 2

_RESP_OK = 0
_RESP_VALUE = 1

_DEFAULT_TIMEOUT = 300


def _send_bytes(sock, data):
    sock.sendall(struct.pack("!I", len(data)))
    sock.sendall(data)


def _recv_bytes(sock):
    raw = _recvall(sock, 4)
    if raw is None:
        raise ConnectionError("connection closed")
    length = struct.unpack("!I", raw)[0]
    return _recvall(sock, length)


def _recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


class _StoreServer:
    def __init__(self, port, world_size, timeout=_DEFAULT_TIMEOUT):
        self._data = {}
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._world_size = world_size
        self._timeout = timeout
        self._closed = False
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("", port))
        self._server.listen(world_size * 4)
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self):
        while not self._closed:
            try:
                conn, _ = self._server.accept()
            except OSError:
                break
            t = threading.Thread(target=self._handle, args=(conn,), daemon=True)
            t.start()

    def _handle(self, conn):
        try:
            while not self._closed:
                hdr = _recvall(conn, 1)
                if hdr is None:
                    break
                cmd = hdr[0]
                if cmd == _CMD_SET:
                    key = _recv_bytes(conn).decode("utf-8")
                    value = _recv_bytes(conn)
                    with self._cond:
                        self._data[key] = value
                        self._cond.notify_all()
                    conn.sendall(bytes([_RESP_OK]))
                elif cmd == _CMD_GET:
                    key = _recv_bytes(conn).decode("utf-8")
                    deadline = time.monotonic() + self._timeout
                    with self._cond:
                        while key not in self._data:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0 or self._closed:
                                raise TimeoutError(
                                    f"TCPStore server: GET '{key}' timed out")
                            self._cond.wait(timeout=min(remaining, 1.0))
                        value = self._data[key]
                    conn.sendall(bytes([_RESP_VALUE]))
                    _send_bytes(conn, value)
                elif cmd == _CMD_WAIT:
                    raw = _recv_bytes(conn)
                    n = struct.unpack("!I", raw[:4])[0]
                    keys = []
                    offset = 4
                    for _ in range(n):
                        klen = struct.unpack("!I", raw[offset:offset + 4])[0]
                        offset += 4
                        keys.append(raw[offset:offset + klen].decode("utf-8"))
                        offset += klen
                    deadline = time.monotonic() + self._timeout
                    with self._cond:
                        while not all(k in self._data for k in keys):
                            remaining = deadline - time.monotonic()
                            if remaining <= 0 or self._closed:
                                raise TimeoutError(
                                    f"TCPStore server: WAIT timed out")
                            self._cond.wait(timeout=min(remaining, 1.0))
                    conn.sendall(bytes([_RESP_OK]))
        except (ConnectionError, OSError, TimeoutError):
            pass
        finally:
            conn.close()

    def close(self):
        self._closed = True
        with self._cond:
            self._cond.notify_all()
        try:
            self._server.close()
        except OSError:
            pass


class TCPStore:
    def __init__(self, host, port, world_size, is_master, timeout=_DEFAULT_TIMEOUT):
        self._host = host
        self._port = port
        self._world_size = world_size
        self._timeout = timeout
        self._server = None
        self._lock = threading.Lock()
        if is_master:
            self._server = _StoreServer(port, world_size, timeout=timeout)
            time.sleep(0.1)
        self._sock = self._connect(host, port, timeout)

    def _connect(self, host, port, timeout):
        deadline = time.monotonic() + timeout
        while True:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(max(deadline - time.monotonic(), 1.0))
                sock.connect((host, port))
                sock.settimeout(self._timeout)
                return sock
            except (ConnectionRefusedError, OSError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"TCPStore: could not connect to {host}:{port} "
                        f"within {timeout}s"
                    )
                time.sleep(0.1)

    def set(self, key, value):
        if isinstance(value, str):
            value = value.encode("utf-8")
        with self._lock:
            self._sock.sendall(bytes([_CMD_SET]))
            _send_bytes(self._sock, key.encode("utf-8"))
            _send_bytes(self._sock, value)
            resp = _recvall(self._sock, 1)
        if resp is None or resp[0] != _RESP_OK:
            raise RuntimeError("TCPStore.set failed")

    def get(self, key):
        with self._lock:
            self._sock.sendall(bytes([_CMD_GET]))
            _send_bytes(self._sock, key.encode("utf-8"))
            resp = _recvall(self._sock, 1)
            if resp is None or resp[0] != _RESP_VALUE:
                raise RuntimeError("TCPStore.get failed")
            return _recv_bytes(self._sock)

    def wait(self, keys, timeout=None):
        buf = struct.pack("!I", len(keys))
        for k in keys:
            kb = k.encode("utf-8")
            buf += struct.pack("!I", len(kb)) + kb
        with self._lock:
            self._sock.sendall(bytes([_CMD_WAIT]))
            _send_bytes(self._sock, buf)
            old_timeout = self._sock.gettimeout()
            if timeout is not None:
                self._sock.settimeout(timeout)
            try:
                resp = _recvall(self._sock, 1)
                if resp is None or resp[0] != _RESP_OK:
                    raise RuntimeError("TCPStore.wait failed")
            finally:
                self._sock.settimeout(old_timeout)

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass
        if self._server is not None:
            self._server.close()
