import socket
import struct
import binascii
import json

ADDRESS = '192.168.10.107'
PORT = 8001
BUFFER_SIZE = 20+2048*4+2048*2*8
print(BUFFER_SIZE)
ENCODING = 'utf-8'
req = 'get'
REQUEST_TEXT = req.encode(ENCODING)
print(REQUEST_TEXT)
RESPONCE_PREFIX = ''

packetSize = 20+2048*4+2048*2*8
size_int = 4
size_f = 2
ch_size = 8
header = 20
data_size = 2048

read_data = {'header': {}, 'data': []}

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1)
sock.connect((ADDRESS, PORT))
def getValue(shotn):
    try:
        if not sock.send(REQUEST_TEXT) == len(REQUEST_TEXT):
            print('Failed to send!')
            return -1
        data = []
        read_data = {'header': {}, 'data': []}
        while(len(data) < BUFFER_SIZE):
            data.extend(sock.recv(BUFFER_SIZE))
        print(len(data))
        if not (data[0] == 92 and data[1] == 35):
            print('this data strange!')
            return 1

        read_data['header']['day'] = data[2]
        read_data['header']['h'] = data[3]
        read_data['header']['m'] = data[4]
        read_data['header']['s'] = data[5]

        read_data['header']['FreqH'] = data[6]
        read_data['header']['FreqL'] = data[7]
        read_data['header']['DataWidthH'] = data[8]
        read_data['header']['DataWidthL'] = data[9]
        day = data[2]
        hour = data[3]
        minut = data[4]
        sec = data[5]
        print('time:', day, 'day %i:%i:%i' %(hour, minut, sec))

        timestamp = []
        for num in range(data_size):
            local_data = []
            timestamp.append(struct.unpack('i', bytearray(data[header + num*size_int: header + (num + 1)*size_int]))[0])
            local_data.append(struct.unpack('i', bytearray(data[header + num*size_int: header + (num + 1)*size_int]))[0])
            for ch in range(ch_size):
                local_data.append(struct.unpack('H', bytearray(data[header + size_int*data_size + num*size_f + ch*data_size*size_f: header + size_int*data_size + num*size_f + ch*data_size*size_f + size_f]))[0])
            read_data['data'].append(local_data)
        with open('%i.json' %shotn, 'w') as file:
            json.dump(read_data, file)
        return 0
    except socket.timeout:
        print('Connection timeout!')
        return -3

print('connected.')
