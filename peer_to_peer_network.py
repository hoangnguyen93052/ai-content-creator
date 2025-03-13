import socket
import threading
import os
import json

class Peer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.peers = {}
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.bind((self.host, self.port))
        self.running = True

    def listen_for_messages(self):
        while self.running:
            try:
                message, address = self.server_socket.recvfrom(1024)
                self.handle_message(message.decode(), address)
            except Exception as e:
                print(f"Error receiving message: {e}")

    def handle_message(self, message, address):
        print(f"Message from {address}: {message}")
        if message.startswith("JOIN"):
            self.peers[address] = True
            self.broadcast(f"Peer {address} has joined the network.")
            self.send_peer_list(address)
        elif message.startswith("LEAVE"):
            if address in self.peers:
                del self.peers[address]
                self.broadcast(f"Peer {address} has left the network.")
        elif message.startswith("FILE"):
            filename = message.split(" ")[1]
            self.send_file(filename, address)
        # Additional message handling can be added here

    def send_peer_list(self, address):
        peer_list = json.dumps(list(self.peers.keys()))
        self.server_socket.sendto(f"PEERS {peer_list}".encode(), address)

    def broadcast(self, message):
        print(f"Broadcasting message: {message}")
        for peer in self.peers.keys():
            self.server_socket.sendto(message.encode(), peer)

    def send_file(self, filename, address):
        if os.path.isfile(filename):
            with open(filename, 'rb') as file:
                data = file.read()
                self.server_socket.sendto(data, address)
            print(f"Sent file {filename} to {address}")
        else:
            print(f"File {filename} does not exist")

    def start(self):
        print(f"Starting peer at {self.host}:{self.port}")
        threading.Thread(target=self.listen_for_messages, daemon=True).start()

    def stop(self):
        self.running = False
        self.server_socket.close()
        print("Peer stopped.")

def main():
    host = "localhost"
    port = 5000
    peer = Peer(host, port)
    peer.start()

    try:
        while True:
            command = input("Enter command (JOIN, LEAVE, FILE [filename]): ").strip()
            if command.startswith("JOIN"):
                continue
            elif command.startswith("LEAVE"):
                peer.stop()
                break
            elif command.startswith("FILE "):
                filename = command.split(" ")[1]
                peer.broadcast(f"FILE {filename}")
            else:
                print("Unknown command.")
    except KeyboardInterrupt:
        peer.stop()

if __name__ == "__main__":
    main()
