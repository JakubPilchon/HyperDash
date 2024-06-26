from http.server import BaseHTTPRequestHandler, HTTPServer, SimpleHTTPRequestHandler
import os
from functools import partial

class HyperDashServer(HTTPServer):
    """
        Implementation of server for HyperDash.
        This class inherits from `http.server.HTTPServer`.

        Parameters:
            directory : str
                 directory where Dashboard sites are located.
            host : str, optional
                 server address. Default is "localhost"
            port : int, optional
                 port on which server is run. Default is `8050`
    """
    
    def __init__(self, directory:str, host:str ="localhost", port:int = 8050, **kwargs) -> None:
        #check for missing files
        missed = []

        for file in ["viz_site.html", "index.html"]:
            if  not os.path.isfile(os.path.join(directory, file)):
                missed.append(file)
        
        if missed:
            raise FileNotFoundError(f"files missing: {missed}")

        print("HyperDash 2024")
        handler = partial(SimpleHTTPRequestHandler, directory = directory)
        self.host = host
        self.port = port
        
        super().__init__((self.host, self.port), handler , **kwargs)

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        """
            Parent method override, that adds information and  manages server shutdown.

            Parameters:
                poll_interval : float
                     Default is `0.5`
        """
        print("Server online at http://{}:{}/ \n Ctrl+C to end".format(self.host, self.port))
        try:
            return super().serve_forever(poll_interval)
        except KeyboardInterrupt:
            self.server_close()
            print("Server Stopped due to manual shutdown")


if __name__ == "__main__":
    print('HyperDash 2024')
    host = input("Provide host adress:")
    #portname = int(input('Provide port: '))
    path = input('Provide path to Hyperdash directory: ')
    server = HyperDashServer(path, host)
    server.serve_forever()
    
