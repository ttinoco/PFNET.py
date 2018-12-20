from pfnet import CustomFunction

class DummyFunction(CustomFunction):

    def init(self):
        self.name = "dummy function"
        
    def count_step(self, bus, busdc, t):
        pass
                
    def analyze_step(self, bus, busdc, t):
        pass
    
    def eval_step(self, bus, busdc, t, x):
        pass
