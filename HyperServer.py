from http.server import BaseHTTPRequestHandler, HTTPServer, SimpleHTTPRequestHandler
import os
from functools import partial

class HyperDashServer(HTTPServer):
    def __init__(self, host:str, port:str, directory:str, **kwargs) -> None:
        print("HyperDash 2024")
        handler = partial(SimpleHTTPRequestHandler, directory = directory)
        
        super().__init__((host, port), handler , **kwargs)


if __name__ == "__main__":
    print('HyperDash 2024')
    host = input("Provide localhost :")
    portname = int(input('Provide port: '))
    path = input('Provide path to Hyperdash directory: ')
    server = HyperDashServer(host, portname, path)
    server.serve_forever()
    
