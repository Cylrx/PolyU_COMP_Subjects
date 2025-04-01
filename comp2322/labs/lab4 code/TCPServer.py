import socket
serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

print ("socket successfully created")
serverPort = 12345
serverSocket.bind(('', serverPort))
print ("socket binded to %s" %(serverPort))
serverSocket.listen(5)
print ("socket is listening")

while True:
    connectionSocket, addr = serverSocket.accept()
    print ('got connection from', addr)
    sentence='thank you for connecting'
    connectionSocket.send(sentence.encode())
    passwd = connectionSocket.recv(1024).decode()
    connectionSocket.send(("Your password is correct!" if passwd == '0134' else "Your password is incorrect!").encode())

    connectionSocket.close()
    break
