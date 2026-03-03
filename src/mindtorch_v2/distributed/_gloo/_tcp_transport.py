"""Full-mesh TCP transport for Gloo backend.

Each rank establishes a direct TCP connection to every other rank.
Connection ordering: rank i connects to rank j (j > i) to avoid deadlock.
"""

import socket
import struct
import threading


def _recvall(sock, n):
    """Read exactly n bytes from a socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer connection closed")
        buf.extend(chunk)
    return bytes(buf)


class TcpTransport:
    """Full-mesh TCP connection manager for peer-to-peer communication."""

    def __init__(self, store, rank, world_size, prefix="", timeout=300):
        self._rank = rank
        self._world_size = world_size
        self._timeout = timeout
        self._closed = False
        self._peers = {}       # peer_rank -> socket
        self._locks = {}       # peer_rank -> threading.Lock

        if world_size < 2:
            return

        # Bind ephemeral server socket for accepting incoming connections
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.settimeout(timeout)
        self._server.bind(("", 0))
        self._server.listen(world_size)
        host = socket.gethostname()
        port = self._server.getsockname()[1]

        # Publish our address to the store so peers can find us
        addr_key = f"gloo_addr_rank_{rank}"
        if prefix:
            addr_key = f"{prefix}/{addr_key}"
        store.set(addr_key, f"{host}:{port}".encode("utf-8"))

        # Collect all peer addresses
        peer_addrs = {}
        for i in range(world_size):
            if i == rank:
                continue
            peer_key = f"gloo_addr_rank_{i}"
            if prefix:
                peer_key = f"{prefix}/{peer_key}"
            store.wait([peer_key])
            raw = store.get(peer_key)
            addr_str = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            h, p = addr_str.rsplit(":", 1)
            peer_addrs[i] = (h, int(p))

        # Establish connections with deterministic ordering to avoid deadlock:
        # rank i connects to rank j where j > i;
        # rank j accepts from rank i where i < j.
        #
        # Process lower ranks first (accept from them), then higher ranks
        # (connect to them).

        # Accept connections from ranks < self
        for i in sorted(r for r in range(world_size) if r < rank):
            conn, _ = self._server.accept()
            conn.settimeout(timeout)
            # Peer sends its rank as a 4-byte int so we know who connected
            peer_rank_bytes = _recvall(conn, 4)
            peer_rank = struct.unpack("!I", peer_rank_bytes)[0]
            self._peers[peer_rank] = conn
            self._locks[peer_rank] = threading.Lock()

        # Connect to ranks > self
        for j in sorted(r for r in range(world_size) if r > rank):
            h, p = peer_addrs[j]
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((h, p))
            # Send our rank so the peer knows who we are
            sock.sendall(struct.pack("!I", rank))
            self._peers[j] = sock
            self._locks[j] = threading.Lock()

    def send_to(self, rank, data):
        """Send data bytes to a peer rank. Thread-safe per socket."""
        lock = self._locks[rank]
        sock = self._peers[rank]
        with lock:
            # Wire format: 8-byte big-endian length + data
            sock.sendall(struct.pack("!Q", len(data)))
            sock.sendall(data)

    def recv_from(self, rank):
        """Receive data bytes from a peer rank. Thread-safe per socket."""
        lock = self._locks[rank]
        sock = self._peers[rank]
        with lock:
            length_bytes = _recvall(sock, 8)
            length = struct.unpack("!Q", length_bytes)[0]
            return _recvall(sock, length)

    def close(self):
        """Close all peer sockets and the server socket."""
        if self._closed:
            return
        self._closed = True
        for sock in self._peers.values():
            try:
                sock.close()
            except OSError:
                pass
        self._peers.clear()
        self._locks.clear()
        if hasattr(self, "_server"):
            try:
                self._server.close()
            except OSError:
                pass
