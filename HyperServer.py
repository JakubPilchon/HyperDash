from http.server import BaseHTTPRequestHandler, HTTPServer, SimpleHTTPRequestHandler
import os
from functools import partial

class HyperDashServer(HTTPServer):
    def __init__(self, host:str, port:str, directory:str, **kwargs) -> None:
        print("HyperDash 2024")
        handler = partial(SimpleHTTPRequestHandler, directory = directory)
        self.host = host
        self.port = port
        
        super().__init__((self.host, self.port), handler , **kwargs)

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        print(r"Server online at /{}:{}/ \n Ctrl+C to end".format(self.host, self.port))
        try:
            return super().serve_forever(poll_interval)
        except KeyboardInterrupt:
            print("Server Stopped due to manual shutdown")


if __name__ == "__main__":
    print('HyperDash 2024')
    host = input("Provide localhost :")
    portname = int(input('Provide port: '))
    path = input('Provide path to Hyperdash directory: ')
    server = HyperDashServer(host, portname, path)
    server.serve_forever()
    
