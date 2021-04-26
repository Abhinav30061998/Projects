#!/usr/bin/env python
# coding: utf-8

# In[1]:


num= int(input("Enter !st num :"))
if(num%2==0) :
    print("It is Even Number : ")
else:
    print("it is Odd Number :")
    


# In[9]:


num = float(input("Enter a number: "))
if num > 0:
   print("Positive number")
elif num == 0:
   print("Zero")
else:
   print("Negative number")


# In[10]:


kerala =["kozhikode","palakkad","thrissur"]


# In[11]:


kerala


# In[12]:


kerala.append("kollam")


# In[13]:


kerala


# In[14]:


kerala.remove("kollam")


# In[15]:


kerala


# In[16]:


for x in kerala(len(a)) :
    print a[\n],


# In[23]:


keralas =["kozhikode","palakkad","thrissur"]
for val in keralas :
    print(val)


# In[1]:


def add(x,y):
    return x+y
def sub(x,y):
    return x-y
def mul(x,y):
    return x*y
def div(x,y):
    return x/y
print("Select Operation")
print("1. Add")
print("2. Subtract")
print("3. Multiply")
print("4. Divide")
choice = input("Enter choice(1/2/3/4): ")
x= int(input("Enter the first number : "))
y= int(input("Enter the second number :"))
if   choice == 1:
    print("The output of the Operation is {0}".format(add(x,y)))
elif choice == 2:
    print("The output of the Operation is {0}".format(sub(x,y)))
elif choice == 3:
    print("The output of the Operation is {0}".format(mul(x,y)))
elif choice == 4:
    print("The output of the Operation is {0}".format(div(x,y)))
else:
    print("The choice is incorrect")


# In[2]:


# Python program to check if year is a leap year or not


year= int(input("Enter the Year :"))

# To get year (integer input) from the user
# year = int(input("Enter a year: "))

if (year % 4) == 0:
   if (year % 100) == 0:
       if (year % 400) == 0:
           print("{0} is a leap year".format(year))
       else:
           print("{0} is not a leap year".format(year))
   else:
       print("{0} is a leap year".format(year))
else:
   print("{0} is not a leap year".format(year))


# In[ ]:





# In[19]:


year= int(input("Enter the Year :"))

if (year % 4) == 0:
    if (year % 10) == 0:
        if (year % 400) == 0:
    print("{0} is a leap year".format(year))
elif:
    print("{0} is not a leap year".format(year))
elif:
    print("{0} is a leap year".format(year))
else:
    print("{0} is not a leap year".format(year))


# In[2]:


## Import Library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


pd


# In[7]:


x=np.linspace(10,20,30)
y=np.linspace(10,20,30)
plt.plot(x,y)
plt.show()


# In[8]:


plt.plot([1,4,6,8],[3,8,3,5])
plt.show()


# In[12]:


plt.xlabel("current")
plt.ylabel("voltage")
plt.title("Ohm's Law")
plt.plot([5,4,15],[10,8,30])
plt.show()


# In[13]:


x= np.linspace(0,10,20)
y=x*2

plt.plot(x,y,'g^')

plt.xlabel("current")
plt.ylabel("voltage")
plt.title("ohm's law")
plt.show()


# In[30]:


x=np.linspace(0,5,10)
y=np.linspace(3,6,10)

plt.plot(x,y,'b-', x,y**2,'y.',x,y**3,'g^',x,y++200,'r-')
plt.show()


# In[42]:


x=np.linspace(1,10,100)
y=np.log(x)

plt.figure(1)

plt.subplot(2,2,1)
plt.title("liner")
plt.plot(x,x,'g^')

plt.subplot(2,2,2)
plt.title("cubic")
plt.plot(x,x**3,'r.')

plt.figure(2)
plt.subplot(2,2,1)
plt.title("log")
plt.plot(x,np.log(x),'b-')

plt.subplot(222)
plt.title("exponential")
plt.plot(x,x**2,'y+')

 plt.show()


# In[4]:


df = pd.read_csv("C:/Users/Studet/Documents/Downloads/market_fact.csv")
df.head()   


# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.boxplot(df['Order_Quantity'])
plt.show()


# In[8]:


plt.boxplot(df['Sales'])
plt.show()


# In[10]:


df['Sales'].describe()


# In[12]:


plt.subplot(1,2,1)
plt.boxplot(df['Sales'])
plt.subplot(1,2,2)
plt.boxplot(df['Sales'])
plt.yscale('log')
plt.show()


# In[17]:


## outliers
##  split in 4 quadrands


# In[18]:


#Histogram
plt.hist(df['Sales'])
plt.show()


# In[19]:


plt.hist(df['Sales'])
plt.yscale('log')
plt.show()


# In[20]:


#Scatterplot
plt.scatter(df['Sales'],df['Profit'])
plt.show()


# In[21]:


df = pd.read_csv("C:/Users/Studet/Documents/Downloads/bitcoin_price.csv")
df.head()   


# In[22]:


plt.boxplot(df['Low'])
plt.show()


# In[23]:


plt.boxplot(df['High'])
plt.show()


# In[24]:


df['Low'].describe()


# In[33]:


plt.subplot(1,2,1)
plt.boxplot(df['Low'])

plt.subplot(1,2,2)
plt.boxplot(df['Low'])
plt.yscale('log')
plt.show()


# In[35]:


plt.hist(df['High'])
plt.show()


# In[36]:


plt.hist(df['High'])
plt.yscale('log')
plt.show()


# In[45]:


#scatterplot
plt.scatter(df['High'],df['Low'])
plt.show()


# In[53]:


plt.bar(df['High'],df['Low'])
plt.yscale('log')
plt.show()


# In[54]:


plt.bar(df['High'],df['Low'])
plt.show()


# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[30]:


sns


# In[31]:


df = pd.read_csv("C:/Users/Studet/Documents/Downloads/market_fact.csv")
df.head()


# #simple density plot
# 
# sns.distplot(df['Shipping_Cost'])
# plt.show()

# In[56]:


#simple density plot

sns.distplot(df['Shipping_Cost'])
sns.set_style("ticks")
plt.show()


# In[33]:


df['Shipping_Cost'].describe()


# In[34]:


sns.distplot(df['Shipping_Cost'])
sns.set_style("ticks")
plt.yscale('log')
plt.show()


# In[36]:


#rug = True
# plotting only a few points since rug takes a long while

sns.distplot(df['Shipping_Cost'][:200],rug=True)
plt.show()


# In[37]:


sns.distplot(df['Sales'],hist=False)
plt.show()


# In[38]:


#subplots

#subplot 1
plt.subplot(221)
plt.title('Sales')
sns.distplot(df['Sales'])

#subplot 2
plt.subplot(222)
plt.title('Profit')
sns.distplot(df['Profit'])

#subplot 3
plt.subplot(223)
plt.title('Order Quantity')
sns.distplot(df['Sales'])

#subplot 4
plt.subplot(224)
plt.title('Shipping Cost')
sns.distplot(df['Shipping_Cost'])

plt.show()


# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


sns


# In[51]:


sns.boxplot(df['Order_Quantity'])
plt.title('Order Quantity')
plt.show()


# In[52]:


sns.boxplot(y=df['Order_Quantity'])
plt.title('Order Quantity')
plt.show()


# In[53]:


# joint plots of Profit and Sales

sns.jointplot('Sales', 'Profit', df)
plt.show()

# same as sns.jointplot(df['Sales'], df['Profit'])


# In[54]:


# remove points having extreme values
df = df[(df.Profit < 10000) & (df.Sales < 20000)]

sns.jointplot('Sales', 'Profit', df)
plt.show()


# In[9]:


# plotting low Sales value orders
# hex plot
df = pd.read_csv("C:/Users/Studet/Documents/Downloads/market_fact.csv")
df = df[(df.Profit < 100) & (df.Profit > -100) & (df.Sales < 200)]
sns.jointplot('Sales', 'Profit', df, kind="hex", color="b")
plt.show()


# In[11]:


# reading cryptocurrency files
btc = pd.read_csv("C:/Users/Studet/Documents/crypto_data/bitcoin_price.csv")
ether = pd.read_csv("C:/Users/Studet/Documents/crypto_data/ethereum_price.csv")
ltc = pd.read_csv("C:/Users/Studet/Documents/crypto_data/litecoin_price.csv")
monero = pd.read_csv("C:/Users/Studet/Documents/crypto_data/monero_price.csv")
neo = pd.read_csv("C:/Users/Studet/Documents/crypto_data/neo_price.csv")
quantum = pd.read_csv("C:/Users/Studet/Documents/crypto_data/qtum_price.csv")
ripple = pd.read_csv("C:/Users/Studet/Documents/crypto_data/ripple_price.csv")

# putting a suffix with column names so that joins are easy
btc.columns = btc.columns.map(lambda x: str(x) + '_btc')
ether.columns = ether.columns.map(lambda x: str(x) + '_et')
ltc.columns = ltc.columns.map(lambda x: str(x) + '_ltc')
monero.columns = monero.columns.map(lambda x: str(x) + '_mon')
neo.columns = neo.columns.map(lambda x: str(x) + '_neo')
quantum.columns = quantum.columns.map(lambda x: str(x) + '_qt')
ripple.columns = ripple.columns.map(lambda x: str(x) + '_rip')

btc.head()


# In[12]:


# merging all the files by date
m1 = pd.merge(btc, ether, how="inner", left_on="Date_btc", right_on="Date_et")
m2 = pd.merge(m1, ltc, how="inner", left_on="Date_btc", right_on="Date_ltc")
m3 = pd.merge(m2, monero, how="inner", left_on="Date_btc", right_on="Date_mon")
m4 = pd.merge(m3, neo, how="inner", left_on="Date_btc", right_on="Date_neo")
m5 = pd.merge(m4, quantum, how="inner", left_on="Date_btc", right_on="Date_qt")
crypto = pd.merge(m5, ripple, how="inner", left_on="Date_btc", right_on="Date_rip")

crypto.head()


# In[13]:


# Subsetting only the closing prices column for plotting
curr = crypto[["Close_btc", "Close_et", 'Close_ltc', "Close_mon", "Close_neo", "Close_qt"]]
curr.head()


# In[15]:


# pairplot
sns.pairplot(curr)
plt.show()


# In[16]:


# You can also observe the correlation between the currencies 
# using df.corr()
cor = curr.corr()
round(cor, 3)


# In[17]:


# figure size
plt.figure(figsize=(10,8))

# heatmap
sns.heatmap(cor, cmap="YlGnBu", annot=True)
plt.show()


# In[2]:


#NUMPY
import numpy as np


# In[4]:


np


# In[6]:


array_1d = np.array([2, 4, 5, 6, 7, 9])
print(array_1d)
print(type(array_1d))


# In[7]:


array_3d = np.array([[2, 3, 4], [5, 8, 7],[5,6,9]])
print(array_3d)


# In[10]:


#lambda

list_1 = [3,6,9,1]
list_2 = [2,4,6,8]
list_3 = [3,5,7,9]

product_list = list(map(lambda x, y: x*y, list_1,list_2))
print(product_list)


# In[11]:


array_1 = np.array(list_1)
array_2 = np.array(list_2)
array_3 = np .array(list_3)

array_3 = array_1*array_2*array_3
print(array_3)
print(type(array_3))


# In[12]:


# Square a list
list_squared = [i**2 for i in list_1]

# Square a numpy array
array_squared = array_1**2

print(list_squared)
print(array_squared)


# In[13]:


array_from_list = np.array([2,5,6,7])
array_from_tuple = np.array((4,5,8,9))

print(array_from_list)
print(array_from_tuple)


# In[14]:


np.ones((5, 3))


# In[16]:


numbers = np.arange(10,100,3)
print(numbers)


# In[17]:


np.random.random([3, 4])


# In[18]:


np.linspace(15, 18, 25)

