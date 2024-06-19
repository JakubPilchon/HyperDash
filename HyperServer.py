from http.server import BaseHTTPRequestHandler, HTTPServer


class __HyperDashRequests(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_header()


class HyperDashServer(HTTPServer):
    def __init__(self, **kwargs) -> None:
        print("HyperDash 2024")
        host = input("Enter host name: ")
        port = int(input("Enter port number: "))

        super().__init__((host, port), __HyperDashRequests, **kwargs)


if __name__ == "__main__":
    HyperDashServer()
    
