Unreleased
----------
* Constant current constraint for CSC HVDC.
* Net update reg Q participations.
* Branch num ratios.
* Branch method to check whether it has zero impedance.    
* Can create copy of network that has equivalent buses merged.
* Can update original network from merged network.
* Net can give num of ZI lines.   
* Localize gen regulation.
* CSC DC current control function.
* Gen redispatch penalty function.
* Branch __str__ and __repr__.
* Network method for clipping switched shunt b.
* Network JSON encoder and decoder.
* Support for net components being in or out of service.  
* Added "v_set_refrence" boolean parameter to voltage magnitude regularization function.
* Added utils submodule with routine to create PTDFs.    
* EPC parser support: pfnet module has_epc_parser method, and ParserEPC class.    
  
Version 1.3.4
-------------
* Linked with appveyor.
* Added fully functional heuristic class and unittest.
* Added constrained function unittest.
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
* Bus v_set_regulated" property and query.
* HVDC CSC converters.
* FACTS devices.
* Updated count/analyze/eval to also loop through dc buses, and updated python-based dummy func/constr.    
* Changed bus sens_v_reg_by_gen to sens_v_set_reg.
* Changed "voltage regulation by generators" constraints to "voltage set point regulation".
* Exposed VSC HVDC and FACTS constraints/functions and added unittests.
* Added "is_in_service" method and "in_service" attribute to load.
* Exposed routines for updated load P and Q components according to provided weights.    
* Exposed CSC HVDC constraints/functions.
* Extended test utils to compare dc buses, dc branches, FACTS, csc converters, and vsc converters.    
* Added "set" method to PyParserMAT to follow C parser API.
* Added shunt adjustment mode (continuous/discrete) and rounding capability.
* Exposed branch power flow Jacobian routine.
* Removed graph wrapper.
* Removed module info dictionary.
* Added package __version__.
* Added routine for getting any network component from key.    
* Added bus function get_v_max(code) get_v_min(code).
* Added output_level to network show_components.
* Added unittest for redundant buses.
* Fixed build_lib.sh to handle PWD with white spaces.

Version 1.3.3
-------------
* Made Python wrapper its own repo.
