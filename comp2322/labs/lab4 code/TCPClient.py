import socket
clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientSocket.bind(('', 40134))
serverName = '127.0.0.1'
serverPort = 12345
clientSocket.connect((serverName, serverPort))

srcIP, srcPort = clientSocket.getsockname()
destIP, destPort = clientSocket.getpeername()

print(f'Tuple: (SrcIP: {srcIP}, SrcPort: {srcPort}, DestIP: {destIP}, DestPort: {destPort})')

sentence = clientSocket.recv(1024).decode()
print ("from server:", sentence)

passwd = input('Input 4 digit password: ')
clientSocket.send(passwd.encode())

auth_result = clientSocket.recv(1034).decode()
print("from server: ", auth_result)

clientSocket.close()