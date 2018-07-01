

class PyParserMAT(object):

    def __init__(self):

        pass

    def parse(self, filename, num_periods=1):

        import grg_mpdata as mp

        case = mp.io.parse_mp_case_file(filename)

        print(case)

    def show(self):

        pass

    def write(self, net, filename):

        pass
