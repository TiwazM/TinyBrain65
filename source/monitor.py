"""Minimal Xemu UART transport for the local regression runner."""
import socket
class Monitor:
    def __init__(self,port):
        self.sock=socket.create_connection(('127.0.0.1',port),timeout=2)
        self.sock.settimeout(2);self.command('r')
    def command(self,command):
        self.sock.sendall((command+'\r\n').encode());data=bytearray()
        while True:
            chunk=self.sock.recv(65536)
            if not chunk:raise RuntimeError('Xemu closed the monitor connection')
            data.extend(chunk);s=data.decode('latin1').replace('\r','')
            if s.rstrip().endswith('.'):return s
