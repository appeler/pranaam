Quick Start
===========

This guide will get you up and running with pranaam in just a few minutes.

Basic Usage
-----------

The main function in pranaam is ``pred_rel``, which predicts religion based on names.

Single Name Prediction
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pranaam
   
   # Predict for a single name
   result = pranaam.pred_rel("Shah Rukh Khan")
   print(result)

Output:

.. code-block:: text

            name pred_label  pred_prob_muslim
   0  Shah Rukh Khan     muslim              73.0

Multiple Names (English)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pranaam
   
   # List of English names
   names = ["Shah Rukh Khan", "Amitabh Bachchan", "Abdul Kalam"]
   result = pranaam.pred_rel(names, lang="eng")
   print(result)

Output:

.. code-block:: text

            name   pred_label  pred_prob_muslim
   0  Shah Rukh Khan       muslim              73.0
   1  Amitabh Bachchan  not-muslim              27.0
   2      Abdul Kalam       muslim              85.5

Hindi Names
~~~~~~~~~~~

.. code-block:: python

   import pranaam
   
   # Hindi names
   hindi_names = ["शाहरुख खान", "अमिताभ बच्चन"]
   result = pranaam.pred_rel(hindi_names, lang="hin")
   print(result)

Output:

.. code-block:: text

           name   pred_label  pred_prob_muslim
   0    शाहरुख खान       muslim              73.0
   1  अमिताभ बच्चन  not-muslim              27.0

Working with Pandas
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import pranaam
   
   # Create a DataFrame with names
   df = pd.DataFrame({
       'names': ['Shah Rukh Khan', 'Amitabh Bachchan', 'A.P.J. Abdul Kalam'],
       'profession': ['Actor', 'Actor', 'Scientist']
   })
   
   # Predict religion for the names column
   predictions = pranaam.pred_rel(df['names'], lang="eng")
   
   # Merge with original data
   result = pd.concat([df, predictions[['pred_label', 'pred_prob_muslim']]], axis=1)
   print(result)

Command Line Interface
----------------------

You can also use pranaam from the command line:

.. code-block:: bash

   # Single name prediction
   predict_religion --input "Shah Rukh Khan" --lang eng
   
   # Hindi name prediction
   predict_religion --input "शाहरुख खान" --lang hin

Understanding the Output
------------------------

The function returns a pandas DataFrame with these columns:

* **name**: The input name
* **pred_label**: Predicted religion ('muslim' or 'not-muslim')  
* **pred_prob_muslim**: Probability score (0-100) that the person is Muslim

Accuracy and Limitations
------------------------

* **High Accuracy**: 98% accuracy on unseen names for both Hindi and English models
* **Binary Classification**: Currently predicts Muslim vs. not-Muslim only
* **Training Data**: Based on Bihar Land Records (4M+ unique records)
* **Context**: Nearly 95% of India's population are Hindu or Muslim

Next Steps
----------

* Check out the :doc:`api` for detailed function documentation
* See :doc:`examples` for more advanced usage patterns
* Learn about the training data in our `notebooks <https://github.com/appeler/pranaam/tree/main/pranaam/notebooks>`_