.. _ext:

**********
Extensions
**********

This section describes how to add custom functions and constraints to PFNET. The general principle is that PFNET asks functions and constraints to ``count`` the number of matrix rows and non-zero entries, then to ``analyze`` the structural and constant parts, and finally to ``evaluate`` the parts that depend on variable values. For each of these operations (``count``, ``analyze``, and ``evaluate``), PFNET loops through buses and time periods once, and asks constraints to perform these operations gradually in ``steps`` for each bus and time.  

.. _ext_func:

Adding a Function
=================

To add a new function to PFNET, one should create a subclass of the :class:`CustomFunction <pfnet.CustomFunction>` class and provide four methods:

* :func:`init <pfnet.CustomFunction.init>`

  * This method initializes any custom function data. 

* :func:`count_step <pfnet.CustomFunction.count_step>`

  * This method is called for every bus and time period, and is responsible for updating the counter :data:`Hphi_nnz <pfnet.FunctionBase.Hphi_nnz>` of non-zero entries of the Hessian matrix of the function (only lower or upper triangular part). 

* :func:`analyze_step <pfnet.CustomFunction.analyze_step>`

  * This method is called for every bus and time period, and is responsible for storing the structural or constant part of the Hessian matrix :data:`Hphi <pfnet.FunctionBase.Hphi>`.  Only one element of each off-diagonal pair should be stored. After all buses and time periods have been processed, PFNET automatically makes the Hessian lower triangular by swapping elements as necessary.

* :func:`eval_step <pfnet.CustomFunction.eval_step>`

  * This method is called for every bus and time period, and is responsible for updating the values of :data:`phi <pfnet.FunctionBase.phi>`, :data:`gphi <pfnet.FunctionBase.gphi>`, and :data:`Hphi <pfnet.FunctionBase.Hphi>` using a given vector of variable values ``x``. 

A template for creating a custom function is provided below. The argument ``bus`` is an :class:`AC bus <pfnet.Bus>` whereas the argument ``busdc`` is an :class:`HVDC bus <pfnet.BusDC>`. For each function call, exactly one of them is given and the other is ``None``.

.. literalinclude:: ../examples/custom_function_template.py

An example of a custom function that computes the quadratic active power generation cost can be found `here <https://github.com/ttinoco/PFNET.py/blob/master/pfnet/functions/dummy_function.py>`__. 

.. _ext_constr:

Adding a Constraint
===================

To add a new constraint to PFNET, one should create a subclass of the :class:`CustomConstraint <pfnet.CustomConstraint>` class. The subclass needs to define the following five methods:

* :func:`init <pfnet.CustomConstraint.init>`

  * This method initializes any custom constraint data.

* :func:`count_step <pfnet.CustomConstraint.count_step>`

  * This method is called for every bus and time period, and is responsible for updating the counters :data:`A_row <pfnet.ConstraintBase.A_row>`, :data:`A_nnz <pfnet.ConstraintBase.A_nnz>`, :data:`G_row <pfnet.ConstraintBase.G_row>`, :data:`G_nnz <pfnet.ConstraintBase.G_nnz>`, :data:`J_row <pfnet.ConstraintBase.J_row>`, and :data:`J_nnz <pfnet.ConstraintBase.J_nnz>`, which count the number of rows and non-zero elements of the matrices :data:`A <pfnet.ConstraintBase.A>`, :data:`G <pfnet.ConstraintBase.G>`, and Jacobian :data:`J <pfnet.ConstraintBase.J>`, respectively, and for updating the entries of the array :data:`H_nnz <pfnet.ConstraintBase.H_nnz>`, which count the number of non-zeros of each nonlinear constraint Hessian (only lower or upper triangular part). If the constraint has extra variables, this method is responsible for updating the counter :data:`num_extra_vars <pfnet.ConstraintBase.num_extra_vars>`.

* :func:`analyze_step <pfnet.CustomConstraint.analyze_step>`

  * This method is called for every bus and time period, and is responsible for storing the structural or constant parts of the matrices :data:`A <pfnet.ConstraintBase.A>`, :data:`G <pfnet.ConstraintBase.G>`, Jacobian :data:`J <pfnet.ConstraintBase.J>`, and the Hessians of the nonlinear equality constraint functions. The latter can be extracted using the method :func:`get_H_single() <pfnet.ConstraintBase.get_H_single>`. For these Hessian matrices, only one element of each off-diagonal pair should be stored. After all buses and time periods have been processed, PFNET automatically makes these constraint Hessians lower triangular by swapping elements as necessary. If the constraint has extra variables, this method is responsible for updating the contents of the vectors :data:`init_extra_vars <pfnet.ConstraintBase.init_extra_vars>`, :data:`u_extra_vars <pfnet.ConstraintBase.u_extra_vars>`, and :data:`l_extra_vars <pfnet.ConstraintBase.l_extra_vars>`, which provide initial values, upper bounds, and lower bounds for the extra variables. 

* :func:`eval_step <pfnet.CustomConstraint.eval_step>`

  * This method is called for every bus and time period, and is used for updating the values of the nonlinear constraint functions :data:`f <pfnet.ConstraintBase.f>`, their Jacobian :data:`J <pfnet.ConstraintBase.J>`, and Hessians using a given vector of variable values ``x`` and a vector of extra constraint variable values ``y``.

* :func:`store_sens_step <pfnet.CustomConstraint.store_sens_step>`

  * This method is called for every bus and time period, and is used for storing constraint sensitivity information in the :ref:`network components <net_components>` and will be documented in the near future. It can be left empty for now.

A template for creating a custom constraint is provided below. The argument ``bus`` is an :class:`AC bus <pfnet.Bus>` whereas the argument ``busdc`` is an :class:`HVDC bus <pfnet.BusDC>`. For each function call, exactly one of them is given and the other is ``None``.

.. literalinclude:: ../examples/custom_constraint_template.py

An example of a custom constraint that constructs the DC power balance equations can be found `here <https://github.com/ttinoco/PFNET.py/blob/master/pfnet/constraints/dummy_constraint.py>`__.

.. note:: Nonlinear constraints implemented in Python will likely be very slow. Therefore, it is recommended to write such constraints directly in C. The procedure of adding constraints in C is similar to the one outlined above. In particular, the same methods need to be provided. Examples of constraints written in C can be found `in this folder <https://github.com/ttinoco/PFNET/tree/master/src/problem/constr>`_.
