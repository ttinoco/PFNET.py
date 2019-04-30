import json
from .cpfnet import Network

class NetworkJSONEncoder(json.JSONEncoder):

    def default(self, obj):

        if isinstance(obj, Network):

            return json.loads(obj.__getstate__())

        else:
            return json.JSONEncoder.default(self, obj)

class NetworkJSONDecoder(json.JSONDecoder):

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        
        if 'base_power' in obj and 'buses' in obj:
            
            net = Network()
            net.__setstate__(json.dumps(obj))
            
            return net

        else:
            return obj
