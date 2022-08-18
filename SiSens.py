import socket
import struct
#import binascii

ADDRESS = '192.168.10.42'
PORT = 4001
BUFFER_SIZE = 64
ENCODING = 'utf-8'
REQUEST_TEXT = b'\x01\x03\x00\x00\x00\x10\x44\x06'
RESPONCE_PREFIX = ''

PACKET_END = '\n'

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1)
sock.connect((ADDRESS, PORT))


def getValue():
    try:
        if not sock.send(REQUEST_TEXT) == len(REQUEST_TEXT):
            print('Failed to send!')
            return -1
        data = []
        while(len(data) < 31):
            #print('r')
            data.extend(sock.recv(BUFFER_SIZE))

        if len(data) == 0:
            print('Warning! No response!')
            return -2
        else:
            if len(data) > 30:
                packet = data[14:16]
                packet += data[12:14]
                return struct.unpack('f', bytearray(packet))[0]
            else:
                print(len(data))
                return -4
    except socket.timeout:
        print('Connection timeout!')
        return -3

print('connected.')
