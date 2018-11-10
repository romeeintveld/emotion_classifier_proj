import tornado.ioloop
import tornado.web
import tornado.websocket
import classifier
import os
import json
 
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")
 
class SimpleWebSocket(tornado.websocket.WebSocketHandler):
    connections = set()
 
    def open(self):
        self.connections.add(self)
 
    def on_message(self, message):
        msg = json.loads(message)
        print(msg)
        if msg['type'] == "predict":
            results = classifier.predict(msg['path'])
            
            # All connected clients receive the result.
            [client.write_message(results) for client in self.connections]
        else:
            # All connected clients receive the error.
            [client.write_message('Unhandled message') for client in self.connections]

    def on_close(self):
        self.connections.remove(self)
 
def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/websocket", SimpleWebSocket)
    ])
 
if __name__ == "__main__":
    app = make_app()
    app.listen(8989)
    ## TODO: if classifier exist, load it. Else train it.
    classifier_file_name = "my_model.h5"
    if not os.path.isfile(classifier_file_name):
        classifier.train()

    tornado.ioloop.IOLoop.current().start()