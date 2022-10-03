#Last amended: 3rd October, 2022
#My folder: C:\Users\Ashok\OneDrive\Documents\streamlit\lessons
# Objective: 
#           Experiments in streamlit
#
##******* Change F9 ********##
#a. Tools-->Preferences-->Keyboard Shortcuts
#b. Search for 'run'
#c. We will interchange Ctrl+Return and F9
#d. Hit the box where text Ctrl+Return is written
#   and hit ctrl+k. Click OK.
#e. Next hit the text box where F9 is written 
#   and press Ctrl+Enter then click OK.
#f. You may restore Ctrl+K to F9
#g. Click Apply and OK
##******* Change F9 ends ********##


#Ref: Main Concepts:
#    https://docs.streamlit.io/library/get-started/main-concepts
#st.write()
#st.dataframe()
#st.line_chart()
#st.area_chart()
#st.bar_chart()---ToDo
#st.pyplot()
#st.slider()




#%%
# 1.0 Call libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%



#########################################################
##                   A. Magic Commands                 ##
#########################################################

# 1.1 Magic commands
#     All variables will be displayed in streamlit


df = pd.DataFrame({ 'age' : [ 23,40, 50,25],
                    'income' : [10, 20, 15,30]})

st.write("Printing df:")
df

# 1.1 Change the above dataframe by adding a IIIrd column:
    
df['exp']  = ['ml', 'bio', 'edu', 'ml']

st.write("Printing df after adding a column:")
df

#########################################################
##                   B. st.write                       ##
#########################################################

# st.write() syntax:
#   It can be passed one or more arguments and different types of objects  
#   Text uses markdown syntax  
# Refer: For Markdown syntax:
#    https://www.markdownguide.org/basic-syntax/
 


# 2.0 st.write() with markdown:
st.write( "<b><ul>print df again:</ul></b> ", df, "*Has three columns*", unsafe_allow_html = True)    

# 2.1 st.write with a mix of html and markdown:
st.write( "<b><ul>print df again:</ul></b> ", df, "*Has three columns*", unsafe_allow_html = True)


# 2.2 Display nympy array with .write()
ar = np.random.randn(10, 20)
st.write("Numpy array", ar)


#########################################################
##                   C. Pandas tables                  ##
#########################################################



# 3.1 Display pandas tables with styles
# Ref: https://pandas.pydata.org/docs/user_guide/style.html

data = np.random.normal(loc = 1.0, scale = 0.78, size=(20,10))
cols= ["cols{}".format(i) for i in range(data.shape[1])]
dx = pd.DataFrame(data, columns = cols)

# 3.2
dx.style

# 3.3 highlight_max() is a streamlit function
#     See: https://docs.streamlit.io/library/api-reference/data/st.dataframe
#     Press ctrl+F to search table
#     Press ctrl+c to copy
st.dataframe(dx.style.highlight_max(axis = 0))


#########################################################
##                   D. Charts                         ##
#########################################################
# https://docs.streamlit.io/library/api-reference/charts

# 4.0 Some data first
fx = np.random.rand(10,3)
st.dataframe(fx)

# 4.1 Now draw a line chart with
#      index as X-axis. There will 
#       be three charts:
st.line_chart(fx)


# 4.2 Transform to DataFrame:
dust = pd.DataFrame(fx, columns = ['X','Y', 'Z'])



# 2.0 Plot line chart
# Ref: https://docs.streamlit.io/library/api-reference/charts/st.line_chart
# 'X' axis values must be set as index
dust = dust.set_index('X')
st.line_chart(dust['Y'])
st.line_chart(dust['Z'])


# 2.1 Plot area chart
#     DataFrame index is X-axis
st.area_chart(dust.Y)



# 3.0 Use 'figure' object in streamlit:
# Ref: https://docs.streamlit.io/library/api-reference/charts/st.pyplot    
arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)
st.pyplot(fig)


# 4.2 Lineplot of matplotlib in streamlit:
du = pd.DataFrame(fx, columns = ['X','Y', 'Z'])
fig,ax = plt.subplots(1,1)
ax.plot('X','Y', data = du)
st.pyplot(fig)


#########################################################
##                     E. Map                          ##
#########################################################

# 5.0 Create some longitude/latitude map data:
map_data = pd.DataFrame(
    (np.random.randn(1000, 2) / [50, 50]) + [37.76, -122.4],
    columns=['lat', 'lon'])

map_data

# 5.1 Some experiments to understand above 
#     transformation:
x = np.random.randn(3,2)
x
x / [50,50]
(x/ [50,50]) + [10,10]
st.map(map_data)


#########################################################
##                     F. Widgets                      ##
#########################################################

## 6.0 Slider
# Ref: https://docs.streamlit.io/library/api-reference/widgets/st.slider


df = pd.DataFrame( np.random.normal(loc= 0.5, scale=0.3, size = (20,10)))
df.columns = [ f"cols{i}" for i in range(10)   ]
df


# 6.1 As you slide the slider, the script runs again from top-to-bottom
#     In this process, df being random in nature, changes again
y = st.slider('x' , min_value= min(df['cols0']), max_value = max(df['cols1']))

# 6.2 Note the flexible syntax of st.write having
#      multiple inputs:
st.write(y, " squared is ",  y * y)

df1 = pd.DataFrame([(23,60,'a'),(24,71,'a'),(21,39,'b'),(10,10,'a')],columns = ['x','y','z']) 
df1

# 6.3 
e = st.slider('What is your age: ',
              min_value = np.min(df1.x),
              max_value = np.max(df1.x),
              step = 2,
              value = 20)

# 6.4
st.write("Value of e * e is : ", e * e )



#####################################################

