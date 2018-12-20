.. include:: defs.hrst

.. _reference:

*************
API Reference
*************

.. _ref_parser:

Parser
======

.. autoclass:: pfnet.ParserBase
   :members: 

.. autoclass:: pfnet.Parser
.. autoclass:: pfnet.ParserJSON
.. autoclass:: pfnet.ParserMAT
.. autoclass:: pfnet.PyParserMAT

.. _ref_bus:

Bus
===

.. _ref_bus_prop:

Bus Properties
--------------

================================ ========
================================ ========
``"any"``
``"slack"``
``"regulated by generator"``
``"regulated by transformer"``
``"regulated by shunt"``
``"not slack"``
``"not regulated by generator"``
================================ ========

.. _ref_bus_q:

Bus Quantities
--------------

================================= ========
================================= ========
``"all"``
``"voltage angle"``
``"voltage magnitude"``
================================= ========

.. _ref_bus_class:

Bus Class
---------

.. autoclass:: pfnet.Bus
   :members:

.. _ref_branch:

Branch
======

.. _ref_branch_prop:

Branch Properties
-----------------

===================== ============================
===================== ============================
``"any"``
``"tap changer"``
``"tap changer - v"`` Controls voltage magnitude
``"tap changer - Q"`` Controls reactive flow
``"phase shifter"``
``"not on outage"``
===================== ============================

.. _ref_branch_q:

Branch Quantities
-----------------

========================= =======
========================= =======
``"all"``
``"phase shift"``
``"tap ratio"``
========================= =======

.. _ref_branch_class:

Branch Class
------------

.. autoclass:: pfnet.Branch
   :members:

.. _ref_branch_y_corr_class:

Branch Y Correction Class
-------------------------

.. autoclass:: pfnet.BranchYCorrection
   :members:

.. _ref_gen:

Generator
=========

.. _ref_gen_prop:

Generator Properties
--------------------

============================= ===========================
============================= ===========================
``"any"``
``"slack"``
``"regulator"``
``"not slack"``
``"not regulator"``
``"not on outage"``
``"adjustable active power"`` :math:`P_{\min} < P_{\max}`
============================= ===========================

.. _ref_gen_q:

Generator Quantities
--------------------

==================== =======
==================== =======
``"all"``
``"active power"``
``"reactive power"``
==================== =======

.. _ref_gen_class:

Generator Class
---------------

.. autoclass:: pfnet.Generator
   :members:

.. _ref_shunt:

Shunt
=====

.. _ref_shunt_prop:

Shunt Properties
----------------

=================== ============================
=================== ============================
``"any"``
``"switching - v"`` Controls voltage magnitude
=================== ============================	
  
.. _ref_shunt_q:

Shunt Quantities
----------------

=========================== =======
=========================== =======
``"all"``
``"susceptance"``
=========================== =======

.. _ref_shunt_class:

Shunt Class
-----------

.. autoclass:: pfnet.Shunt
   :members:

.. _ref_load:

Load
====

.. _ref_load_prop:

Load Properties
---------------

============================= ===========================
============================= ===========================
``"any"``
``"adjustable active power"`` :math:`P_{\min} < P_{\max}`
============================= ===========================

.. _ref_load_q:

Load Quantities
---------------

==================== =======
==================== =======
``"all"``
``"active power"``
``"reactive power"``
==================== =======

.. _ref_load_class:

Load Class
----------

.. autoclass:: pfnet.Load
   :members:

.. _ref_vargen:

Variable Generator
==================

.. _ref_vargen_prop:

Variable Generator Properties
-----------------------------

========= =======
========= =======
``"any"``
========= =======

.. _ref_vargen_q:

Variable Generator Quantities
-----------------------------

==================== =======
==================== =======
``"all"``
``"active power"``
``"reactive power"``
==================== =======

.. _ref_vargen_class:

Variable Generator Class
------------------------

.. autoclass:: pfnet.VarGenerator
   :members:

.. _ref_bat:

Battery
=======

.. _ref_bat_prop:

Battery Properties
------------------

========= =======
========= =======
``"any"``
========= =======

.. _ref_bat_q:

Battery Quantities
------------------

==================== =======
==================== =======
``"all"``
``"charging power"``
``"energy level"``
==================== =======

.. _ref_bat_class:

Battery Class
-------------

.. autoclass:: pfnet.Battery
   :members:

.. _ref_facts:

FACTS
=====

.. _ref_facts_prop:

FACTS Properties
----------------

========= =======
========= =======
``"any"``
========= =======

.. _ref_facts_q:

FACTS Quantities
----------------

==================== =======
==================== =======
``"all"``
==================== =======

.. _ref_facts_class:

FACTS Class
-----------

.. autoclass:: pfnet.Facts
   :members:

.. _ref_vsc:

HVDC VSC
========

.. _ref_vsc_prop:

HVDC VSC Properties
-------------------

========= =======
========= =======
``"any"``
========= =======

.. _ref_vsc_q:

HVDC VSC Quantities
-------------------

==================== =======
==================== =======
``"all"``
==================== =======

.. _ref_vsc_class:

HVDC VSC Class
--------------

.. autoclass:: pfnet.ConverterVSC
   :members:

.. _ref_csc:

HVDC CSC
========

.. _ref_csc_prop:

HVDC CSC Properties
-------------------

========= =======
========= =======
``"any"``
========= =======

.. _ref_csc_q:

HVDC CSC Quantities
-------------------

==================== =======
==================== =======
``"all"``
==================== =======

.. _ref_csc_class:

HVDC CSC Class
--------------

.. autoclass:: pfnet.ConverterCSC
   :members:

.. _ref_busdc:

HVDC Bus
========

.. _ref_busdc_prop:

HVDC Bus Properties
-------------------

========= =======
========= =======
``"any"``
========= =======

.. _ref_busdc_q:

HVDC Bus Quantities
-------------------

==================== =======
==================== =======
``"all"``
==================== =======

.. _ref_busdc_class:

HVDC Bus Class
--------------

.. autoclass:: pfnet.BusDC
   :members:

.. _ref_branchdc:

HVDC Branch
===========

.. _ref_branchdc_prop:

HVDC Branch Properties
----------------------

========= =======
========= =======
``"any"``
========= =======

.. _ref_branchdc_q:

HVDC Branch Quantities
----------------------

==================== =======
==================== =======
``"all"``
==================== =======

.. _ref_branchdc_class:

HVDC Branch Class
-----------------

.. autoclass:: pfnet.BranchDC
   :members:
      
.. _ref_net:

Network
=======

.. _ref_net_obj:

Component Types
---------------

======================== =======
======================== =======
``"all"``
``"bus"``
``"battery"``
``"branch"``
``"generator"``
``"load"``
``"shunt"``
``"variable generator"``
``"csc converter"``
``"vsc converter"``
``"dc bus"``
``"dc branch"``
``"facts"``
======================== =======

.. _ref_net_flag:

Flag Types
----------

============== ==============================================
============== ==============================================
``"variable"`` For selecting quantities to be variables
``"fixed"``    For selecting variables to be fixed
``"bounded"``  For selecting variables to be bounded.
``"sparse"``   For selecting control adjustments to be sparse
============== ==============================================

.. _ref_var_values:

Variable Value Options
----------------------

================== =======
================== =======
``"current"``
``"upper limits"``
``"lower limits"``
================== =======

.. _ref_net_class:

Network Class
-------------

.. autoclass:: pfnet.Network
   :members:

.. _ref_cont:

Contingency
===========

.. autoclass:: pfnet.Contingency
   :members:

.. _ref_func:

Function
========

.. _ref_func_names:

Function Names
--------------

====================================== =======
====================================== =======
``"consumption utility"`` 
``"generation cost"``
``"generator powers regularization"``
``"net consumption cost"``
``"phase shift regularization"``
``"susceptance regularization"``
``"tap ratio regularization"``
``"soft voltage magnitude limits"``
``"sparse controls penalty"``
``"voltage magnitude regularization"`` 
``"voltage angle regularization"``
====================================== =======
 
.. _ref_func_class:

Function Classes
----------------

.. autoclass:: pfnet.FunctionBase
   :members:

.. autoclass:: pfnet.Function
.. autoclass:: pfnet.CustomFunction
   :members:

.. _ref_constr:

Constraint
==========

.. _ref_constr_names:

Constraint Names
----------------

============================================ =======
============================================ =======
``"AC power balance"``
``"DC power balance"``
``"linearized AC power balance"``
``"variable fixing"``
``"variable bounds"``
``"variable nonlinear bounds"``
``"generator active power participation"``
``"generator reactive power participation"``
``"generator ramp limits"``
``"voltage regulation by generators"``
``"voltage regulation by transformers"``
``"voltage regulation by shunts"``
``"AC branch flow limits"``
``"DC branch flow limits"``
``"linearized AC branch flow limits"``
``"battery dynamics"``
``"load constant power factor"``
============================================ =======

.. _ref_constr_class:

Constraint Classes
------------------

.. autoclass:: pfnet.ConstraintBase
   :members:

.. autoclass:: pfnet.Constraint
.. autoclass:: pfnet.CustomConstraint
   :members:

.. _ref_heur:

Heuristic
=========

.. _ref_heur_names:

Heuristic Names
---------------

======================================= =======
======================================= =======
``"PVPQ switching"``
======================================= =======
 
.. _ref_heur_class:

Heuristic Classes
-----------------

.. autoclass:: pfnet.HeuristicBase
   :members:

.. autoclass:: pfnet.Heuristic
      
.. _ref_problem:

Optimization Problem
====================

.. _ref_problem_class:

Problem Class
-------------

.. autoclass:: pfnet.Problem
   :members:   
   
Test Utilities
==============

.. _ref_test_utilities:

.. autofunction:: pfnet.tests.utils.check_constraint_combined_Hessian
.. autofunction:: pfnet.tests.utils.check_constraint_single_Hessian
.. autofunction:: pfnet.tests.utils.check_constraint_Jacobian
.. autofunction:: pfnet.tests.utils.check_function_Hessian
.. autofunction:: pfnet.tests.utils.check_function_gradient
.. autofunction:: pfnet.tests.utils.compare_buses		     
.. autofunction:: pfnet.tests.utils.compare_generators
.. autofunction:: pfnet.tests.utils.compare_loads
.. autofunction:: pfnet.tests.utils.compare_shunts
.. autofunction:: pfnet.tests.utils.compare_branches
.. autofunction:: pfnet.tests.utils.compare_dc_buses
.. autofunction:: pfnet.tests.utils.compare_dc_branches				    
.. autofunction:: pfnet.tests.utils.compare_csc_converters
.. autofunction:: pfnet.tests.utils.compare_vsc_converters
.. autofunction:: pfnet.tests.utils.compare_facts
.. autofunction:: pfnet.tests.utils.compare_networks
.. autofunction:: pfnet.tests.utils.check_network
