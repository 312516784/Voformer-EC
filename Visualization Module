import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
import pandas_bokeh
from pyecharts import options as opts
from pyecharts.charts import BMap
import json
from pyecharts.commons.utils import JsCode

# Ploting
# Load data
path = '/weather data/'
result = pd.read_excel(path+'result analysis.xlsx',sheet_name='Sheet')

p1 = result[['count']].plot_bokeh.bar(title='Bar_Clustering Category/Amount', legend='top_left',zooming=False,panning=False,
                                      toolbar_location=None,figsize=(600,550),
                                      show_figure=False)
p2 = result[['clusters','count']].plot_bokeh.pie(title='Pie_Clustering Result Comparison',
                                                 legend='top_right',x='clusters',y='count',ylabel='Count',
                                                 colormap=[ '#6495ED', '#ADFF2F', '#FF0000', '#FFDAB9', '#6A5ACD', '#B0C4DE', '#2E8B57',
                                                           '#FFFFE0', '#F5DEB3', '#FF7F50', '#C71585', '#008B8B', '#8B4513', '#008000',
                                                           '#556B2F', '#9932CC', '#FFE4C4', '#87CEEB', '#B8860B', '#CD5C5C'],
                                                 line_color="grey",figsize=(600,550),show_figure=False)
p3 = result.plot_bokeh.scatter(figsize=(1200,600),legend='top_left',
                             x='clusters',
                             y='count',
                             size='size',
                             title='Scatter_Clustering Category Distribution',
                             zooming=False,
                             panning=False,
                             toolbar_location=None,
                            show_figure=False)

layout = pandas_bokeh.column(pandas_bokeh.row(p1,p2),p3)
pandas_bokeh.show(layout)

# Map Visualisation
# Data preprocessing
# Load data
# Create map coordinate
data = pd.read_excel(path+'result analysis.xlsx',sheet_name='Sheet1')
data_re = data[['Name','longitude','latitude']].set_index('Name').T.to_dict('list')
result1 = json.dumps(data_re,ensure_ascii=False)

CoordMap = data[['Name','longitude','latitude']].set_index('Name').T.to_dict('list')
clusters_data = data[['Name','clusters']].values.tolist()

# Plot map
# Set number of clusters
num_clusters = 6

# Generate list of unique colors for each cluster
cluster_colors = ['#'+''.join(random.choice('0123456789ABCDEF') for j in range(6)) for i in range(num_clusters)]

# Set visual map options
visual_map_opts = opts.VisualMapOpts(is_piecewise=True, min_=-1, max_=num_clusters, pos_right='10%', pos_bottom='10%', type_='color',
                                     pieces=[{"value": i, "label": f"Cluster {i}", "color": cluster_colors[i-1]} for i in range(1, num_clusters+1)])v

# Instantiate BMap
mapp = BMap(init_opts=opts.InitOpts(width="1400px",height="800px"))
# Add points by json, aiming to create a map coordinate
mapp.add_coordinate_json(json_file=path+"map.json")

# Data Setting
mapp.add(type_="scatter",
         series_name="",
         data_pair=[list(z) for z in zip(data['Name'],data['clusters'])],
         symbol_size= 7)

# Setting parameters, center position is China center
mapp.add_schema(baidu_ak= 'no7Ooaf84XtUzaGRYybWgVyBBOFGD7k6',center=[104.114129, 37.550339],zoom=11)

# Globle Option
mapp.set_global_opts(title_opts=opts.TitleOpts(title="Temperature Clustering Result Distribution",
                                              pos_left="center",
                                              title_textstyle_opts=opts.TextStyleOpts(color='#ff0000')),
                     tooltip_opts=opts.TooltipOpts(trigger="item",
                                                  formatter=[JsCode(
                                                      """function(params) {
                                                      if ('value' in params.data) {
                                                          return params.data.name + ':' + params.data.value[2];
                                                      }
                                                    }""")]),
                     visualmap_opts=opts.VisualMapOpts(is_piecewise=True, min_=-1, max_=20, pos_right='10%', pos_bottom='10%', type_='color',
                                                       pieces=
                                                       [#{"value":20,'lable':'Cluster 20','color':'#FFC0CB'},
                                                         #{"value":19,'lable':'Cluster 19','color':'#DC143C'},
                                                         #{"value":18,'lable':'Cluster 18','color':'#D8BFD8'},
                                                        #{"value":17,'lable':'Cluster 17','color':'#DB7093'},
                                                         #{"value":16,'lable':'Cluster 16','color':'#FF69B4'},
                                                         #{"value":15,'lable':'Cluster 15','color':'#278EA5'},
                                                         #{"value":14,'lable':'Cluster 14','color':'#21E6C1'},
                                                         #{"value":13,'lable':'Cluster 13','color':'#FFBA5A'},
                                                         #{"value":12,'lable':'Cluster 12','color':'#FF7657'},
                                                         #{"value":11,'lable':'Cluster 11','color':'#C56868'},
                                                         #{"value":10,'lable':'Cluster 10','color':'#6b76ff'},
                                                         #{"value":9,'lable':'Cluster 9','color':'#ff0000'},
                                                         {"value":8,'lable':'Cluster 8','color':'#9e579d'},
                                                         {"value":7,'lable':'Cluster 7','color':'#f85f73'},
                                                         {"value":6,'lable':'Cluster 6','color':'#928a97'},
                                                         {"value":5,'lable':'Cluster 5','color':'#b6f7c1'},
                                                         {"value":4,'lable':'Cluster 4','color':'#0b409c'},
                                                         {"value":3,'lable':'Cluster 3','color':'#d22780'},
                                                         {"value":2,'lable':'Cluster 2','color':'#882042'},
                                                         {"value":1,'lable':'Cluster 1','color':'#071E3D'},
                                                         {"value":-1,'lable':'Cluster 0','color':'#B40404'},
                                                     ]))
mapp.set_series_opts(label_opts=opts.LabelOpts(is_show=False))

# Whether data Label shows
mapp.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
mapp.render(path+'Temperature_Map Visualisation.html')
