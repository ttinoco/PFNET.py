#Rapid test
import pfnet

parser=pfnet.PyParserRAW()
net=parser.parse('C:\\Users\\Barberia Juan Luis\\Desktop\\3w_trafo.raw')


net.show_components(output_level=2)

'''
for i in range(3):
    
    print net.branches[i].bus_m.name
    print net.branches[i].bus_k.name
    
    print net.branches[i].b
    print net.branches[i].g

'''




