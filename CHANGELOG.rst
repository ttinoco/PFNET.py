Unreleased
----------
* Linked with appveyor.
* Added fully functional heuristic class and unittest.
* Added "constrained function" unittest.
* Added fix that makes constraints keep functions alive after set_parameter.    
* Exposed contingency name and added unittests.
* Direct memory mapping for array attributes.
* Support for simplified constraint api.
* Support for simplified function api.  
* Added PyParserMAT that uses grg-mpdata and can parse and write Matpower .m files.
* Wrote generic Parser that can use CParser (.mat, .json, etc) or PyParser (.m) based on extension.    
* Updated code to be compatible with bus-loop-based C pfnet library.
* Voltage dependent loads.
* Changed parent class of AttributeInt in order to make it work better with numpy.
* Load voltage dependence constraint.
* Unittest for raw write.
* DC buses, DC branches, and HVDC VSC converters.
* Bus "v set regulated" property and query.
* HVDC CSC converters.
* FACTS devices.    

Version 1.3.3
-------------
* Made Python wrapper its own repo.
