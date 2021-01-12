#!/usr/bin/env python
# coding: utf-8

# # Apriori

# In[23]:


dataset = [["Onion", "Potato", "Burger"],
           ["Potato","Burger", "Milk"],
           ["Milk", "Beer"],
           ["Onion", "Potato", "Milk"],
           ["Onion", "Potato", "Burger", "Beer"],
           ["Onion", "Potato", "Burger", "Milk"]
          ]


# In[24]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder


# In[25]:


dataset


# In[26]:


te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)


# In[27]:


df = pd.DataFrame(te_ary, columns=te.columns_)
df


# In[28]:


from mlxtend.frequent_patterns import apriori

frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
frequent_itemsets


# # Association rule mining
# 

# In[29]:


from mlxtend.frequent_patterns import association_rules
res = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
res

Antecedent support variable tells us probability of antecedent products alone
Consequents support variable tells us probability of consequents products alone
The support value is the value of the two products (Antecedents and Consequents)
Confidence is an indication of how often the rule has been found to be true.
The ratio of the observed support to that expected if X and Y were independent.

# In[30]:


res1 = res[["antecedents", "consequents","support", "confidence", "lift"]]
res1


# In[31]:


res2 = res1[res1["confidence"]>=0.6]
res2


# In[32]:


res3 = res2[res2["lift"]>=1]
res3


# In[35]:


res4 = res3[res3["confidence"]>=1]
res4


# In[ ]:




