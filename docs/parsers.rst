.. include:: defs.hrst

.. _parsers:

************
Data Parsers
************

This section describes the different data parsers available in PFNET. 

.. _parsers_overview:

Overview
========

Parsers in PFNET are subclasses of the |ParserBase| class. They can be used to read a power network data file and create a |Network|. For convenience, a format-specific parser can be instantiated from the |Parser| class by specifying the file extension or a sample filename::

  >>> import pfnet
  >>> parser = pfnet.Parser('m')
  >>> network = parser.parse('ieee14.m')

For this and subsequent examples, is it assumed that the Python interpreter is started in a directory where the sample case |ieee14| can be found.

.. _parsers_json:

JSON
====

PFNET networks can be constructed from data files in the popular lightweight data-interchange format |JSON|. These network data files have extension ``.json`` and parsers for them can be instantiated from the class |ParserJSON|. These JSON parsers also allow writing a given network to a file, as the example below shows::

  >>> parser_json = pfnet.ParserJSON()
  >>> parser_json.write(network, 'new_network.json')
  >>> network = parser_json.parse('new_network.json')

For creating, visualizing, or modifying these JSON network files, online editors such as the following may be used for convenience:

* `<http://jsoneditoronline.org>`_
* `<http://www.cleancss.com/json-editor>`_

In the top-level object of the JSON data, *i.e.*, the network, the field ``version`` indicates the PFNET version associated with the data.
  
.. _parsers_m:

MATPOWER
========

|MATPOWER| is a popular |MATLAB| package for solving power flow and optimal power flow problems. It contains several power flow and optimal power flow cases defined in |MATLAB| files. These files, which have extension ``.m``, can be used to create power networks in PFNET using parsers of type |PyParserMAT|, which leverage the package `grg-mpdata <https://github.com/lanl-ansi/grg-mpdata>`_::

  >>> parser_mat = pfnet.PyParserMAT()
  >>> parser_mat.write(network, 'new_network.m')
  >>> network = parser_mat.parse('new_network.m')

A similar parser of type |ParserMAT| is also available for parsing CSV files of extension ``.mat`` created from |MATPOWER| ``.m`` data files using the tool `mpc2mat.m <https://github.com/ttinoco/PFNET/blob/master/tools/mpc2mat.m>`_.
