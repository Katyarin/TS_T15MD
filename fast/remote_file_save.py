import socket
import struct
import json

ADDRESS = '192.168.10.41'
PORT = 8001
ENCODING = 'utf-8'
req = 'get fast'
REQUEST_TEXT = req.encode(ENCODING)
print(REQUEST_TEXT)
RESPONCE_PREFIX = ''

BUFFER_SIZE = 60

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(100)
sock.connect((ADDRESS, PORT))


def saveData():
    try:
        if not sock.send(REQUEST_TEXT) == len(REQUEST_TEXT):
            print('Failed to send!')
            return -1
        return sock.recv(BUFFER_SIZE).decode(ENCODING)
    except socket.timeout:
        print('Connection timeout!')
        return -3

print('connected.')