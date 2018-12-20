from pfnet import CustomConstraint

class DummyConstraint(CustomConstraint):

    def init(self):        
        self.name = "dummy constraint"

    def count_step(self, bus, busdc, t):
        pass

    def analyze_step(self, bus, busdc, t):
        pass

    def eval_step(self, bus, busdc, t, x, y):
        pass
        
    def store_sens_step(self, bus, busdc, t, sA, sf, sGu, sGl):
        pass

    
