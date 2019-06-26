#Rapid test
import pfnet

parser=pfnet.PyParserRAW()

net=parser.parse('C:\Users\Barberia Juan Luis\Desktop\IEEE_14_bus.raw')

net.show_components(output_level=1)

