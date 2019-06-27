import numpy as np
import pandas as pd
pd.set_option('max_colwidth',50)
pd.set_option('max_columns',250)
pd.set_option('max_rows',500)

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import adfuller, kpss, acf, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf,month_plot,quarter_plot
from scipy import signal

#visualizations
import folium
from folium import plugins
from folium.plugins import HeatMap
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns 
import cufflinks as cf
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools as tls
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
from mpl_toolkits.basemap import Basemap
from pylab import rcParams
cf.go_offline()
cf.set_config_file(world_readable=True,theme='white') # Make all charts public and set a global theme
sns.set_style("whitegrid") #possible choices: white, dark, whitegrid, darkgrid, ticks

### Define function for dataset statistics ###
def get_stats(df, sort_by='Different Values', sort_how=False, exclude=[], include_only=[], target='target'):
        columns = [c for c in df.columns.values if c not in exclude and c in include_only]
        df=df[columns]
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Data types
        data_types = df.dtypes
        
        # Different Values
        other_values = pd.DataFrame(columns=['Different Values','Most Common','% of Most Common','Skewness',
                                            'Kurtosis','Mean','Min','25% quantile','Median','75% quantile','Max'])
        for c in columns:
            if (df[c].dtype != 'object'):
                other_values = other_values.append({
                    'Name' : c,
                    'Different Values' : df[c].value_counts().count(),
                    'Most Common' : df[c].value_counts().idxmax(),
                    '% of Most Common' : 100*df[c].value_counts().max() / df[c].value_counts().sum(),
                    'Skewness' : df[c].skew(),
                    'Kurtosis' : df[c].kurt(),
                    'Mean' : df[c].mean(),
                    'Min' : df[c].min(),
                    '25% quantile' : df[c].quantile(0.25),
                    'Median' : df[c].median(),
                    '75% quantile' : df[c].quantile(0.75),
                    'Max' : df[c].max(),
                                                    }, ignore_index=True)
            else:
                other_values = other_values.append({
                    'Name' : c,
                    'Different Values' : df[c].value_counts().count(),
                    'Most Common' : df[c].value_counts().idxmax(),
                    '% of Most Common' : 100*df[c].value_counts().max() / df[c].value_counts().sum()
                                                    }, ignore_index=True)                
        other_values = other_values.set_index('Name')
        
        
        # Make a table with the results
        mis_val_table = pd.concat([data_types, other_values, mis_val, mis_val_percent], axis=1, sort=False)       
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0:'Type',
                   'Different Values':'Different Values',
                   'Most Common':'Most Common',
                   '% of Most Common':'% of Most Common',
                   'Skewness':'Skewness', 
                   'Kurtosis':'Kurtosis', 
                   1:'Missing Values', 
                   2:'% of Missing Values',
                   'Mean' : 'Mean',
                   'Min' : 'Min',
                   '25% quantile' : '25% quantile',
                   'Median' : 'Median',
                   '75% quantile' : '75% quantile',
                   'Max' : 'Max',
        })
        
        # Sort the table 
        mis_val_table_ren_columns = mis_val_table_ren_columns.sort_values(sort_by, ascending=sort_how)
        df = mis_val_table_ren_columns
        
        # Re-arrange columns
        #cols = df.columns.tolist()
        #cols = cols[0:1] + cols[7:8]# + cols[0:1] + cols[2:3] + cols[3:4] + cols[5:6] + cols[4:5] + cols[6:7] +cols[7:8] +cols[8:9] + cols[9:10]
        #df = df[cols]
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[0]) + " columns. (" +
        str(df[df['Type']=='object'].shape[0])+" categorical and "+str(df[df['Type']!='object'].shape[0])+" numerical).\n"
        "There are " + str(df[df['Missing Values']>0].shape[0]) +" columns that have missing values, of which " +
        str(df[df['% of Missing Values']>=90].shape[0]) + " have over 90%!\n" +
        str(df[(df['Type']=='object')&(df['Different Values']<5)].shape[0]) + 
        " object type columns have less than 5 different values and you can consider one-hot encoding, while " + 
        str(df[(df['Type']=='object')&(df['Different Values']>=5)].shape[0]) +
        " have more than 5 colums and you can consider label encoding.\n" +
        str(df[df['Skewness']>1].shape[0]) + " columns are highly positively skewed (skewness>1), while " +
        str(df[df['Skewness']<-1].shape[0]) + " columns are highly negatively skewed (skewness<-1).\n" +
        str(df[(df['Skewness']>-0.5)&(df['Kurtosis']<0.5)].shape[0]) + " columns are symmetrical (-0.5<skewness<0.5).\n" +
        str(df[df['Kurtosis']>3].shape[0]) + " columns have high kurtosis (kurtosis>3) and should be check for outliers, while " + 
        str(df[df['Kurtosis']<3].shape[0]) + " columns have low kurtosis (kurtosis<3). "
        )        
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
        
### Define function for Univariate Plots
def plot_univariate(df, plot_title):
    hist_data = []
    hist_data.extend([df])
        
    group_labels = ['Surface Temperature']
    fig = ff.create_distplot(hist_data, group_labels, show_hist=True)
    distplot1=fig['data']
    
    trace = go.Box(y=df, name='', pointpos = -1.8, jitter = 0.3, boxpoints = 'all') 

    my_fig = tls.make_subplots(rows=1, cols=2, print_grid=False)
    my_fig.layout.update(title=plot_title, barmode = 'overlay', bargap=0.04, height=360, width=940, showlegend=False)

    my_fig.append_trace(distplot1[0], 1, 1)
    my_fig.append_trace(distplot1[1], 1, 1)
    my_fig.append_trace(trace, 1, 2)
    iplot(my_fig)

### Define function for Map Plot         
def plot_map(df,variable,title,saveFig=False):
    # How much to zoom from coordinates (in degrees)
    zoom_scale = 0
    lats = df['Center Lat'].tolist()
    lons = df['Center Long'].tolist()
    prob = df[variable].tolist()

    # Setup the bounding box for the zoom and bounds of the map
    bbox = [np.min(lats)-zoom_scale,np.max(lats)+zoom_scale,\
            np.min(lons)-zoom_scale,np.max(lons)+zoom_scale]

    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_title(title, fontsize = 20, loc='center')
    matplotlib.rcParams.update({'font.size': 14})
    # Define the projection, scale, the corners of the map, and the resolution.
    m = Basemap(projection='cyl',llcrnrlat=29.75,urcrnrlat=46.25, llcrnrlon=-6,urcrnrlon=36.5,lat_ts=10,resolution='h')
    m.bluemarble()

    # Draw coastlines and fill continents and water with color
    m.drawcoastlines()
    #m.fillcontinents(color='#CCCCCC',lake_color='lightblue')

    # draw parallels, meridians, and color boundaries
    m.drawparallels(np.arange(29.75,46.25,(46.25-30)/5),labels=[1,0,0,0])
    m.drawmeridians(np.arange(-6,36.5,(36.5+6)/5),labels=[0,0,0,1],rotation=15)
    m.drawmapboundary(fill_color='lightblue')

    # format colors for elevation range
    alt_min = np.min(prob)
    alt_max = np.max(prob)
    cmap = plt.get_cmap('jet') #jet,nipy_spectral,YlOrRd,hot_r
    normalize = matplotlib.colors.Normalize(vmin=alt_min, vmax=alt_max)

    from matplotlib.patches import Rectangle
    from matplotlib.patches import FancyBboxPatch

    # the range [50,250] can be changed to create different colors and ranges
    for ii in range(0,len(prob)):
        x,y = m(lons[ii],lats[ii])
        color_interp = np.interp(prob[ii],[alt_min,alt_max],[50,250])   
        rect=plt.Rectangle(xy=(x-0.25,y-0.25),linewidth=0.6, width=0.5,height=0.5,fill=True, alpha=0.8,color=cmap(int(color_interp)))
        border=plt.Rectangle(xy=(x-0.25,y-0.25),linewidth=0.2, width=0.5,height=0.5,fill=False, alpha=0.7,color="#000088")
        ax.add_patch(rect);ax.add_patch(border)

    # format the colorbar 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,norm=normalize)

    # save the figure and show it
    if (saveFig == True):
        plt.savefig(title+'.png', format='png', dpi=500,transparent=True)
    plt.show()  

### Define Function for Time Series Line Plot
def time_plot(df,variable,title,y_axis_title):
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.lineplot(df.index, df[variable] )

    ax.set_title(title, fontsize = 20, loc='center', fontdict=dict(weight='bold'))
    ax.set_xlabel('Year', fontsize = 16, fontdict=dict(weight='bold'))
    ax.set_ylabel(y_axis_title, fontsize = 16, fontdict=dict(weight='bold'))
    plt.tick_params(axis='y', which='major', labelsize=16)
    plt.tick_params(axis='x', which='major', labelsize=16)
    ax.yaxis.tick_left()  

### Define Function for Time Series Seasonality Plots       
def seasonality_plot(df,variable,title,y_axis_title=''):
    #variable = 'salinitySurface'
    fig, ax = plt.subplots(figsize=(15, 6))

    palette = sns.color_palette("ch:2.5,-.2,dark=.3", 10)
    sns.lineplot(df['month'], df[variable], hue=df['year'], palette=palette)
    ax.set_title(title, fontsize = 20, loc='center', fontdict=dict(weight='bold'))
    ax.set_xlabel('Month', fontsize = 16, fontdict=dict(weight='bold'))
    ax.set_ylabel(y_axis_title, fontsize = 16, fontdict=dict(weight='bold'))


    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    sns.boxplot(df['year'], df[variable], ax=ax[0])
    ax[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
    ax[0].set_xlabel('Year', fontsize = 16, fontdict=dict(weight='bold'))
    ax[0].set_ylabel(y_axis_title, fontsize = 16, fontdict=dict(weight='bold'))

    sns.boxplot(df['month'], df[variable], ax=ax[1])
    ax[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
    ax[1].set_xlabel('Month', fontsize = 16, fontdict=dict(weight='bold'))
    ax[1].set_ylabel(y_axis_title, fontsize = 16, fontdict=dict(weight='bold'))    

### Decomposition and strength of trend and seasonality###    
def decomposition(df,variable):    

    y = df[[variable]]
    rcParams['figure.figsize'] = 15, 12
    rcParams['axes.labelsize'] = 20
    rcParams['ytick.labelsize'] = 16
    rcParams['xtick.labelsize'] = 16
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')
    decomp = decomposition.plot()
    decomp.suptitle('', fontsize=22)

    trend_strength = max(0,(1-decomposition.resid.var()/(decomposition.resid+decomposition.trend).var())[0])
    seasonality_strength = max(0,(1-decomposition.resid.var()/(decomposition.resid+decomposition.seasonal).var())[0])
    print("Trend strength:",trend_strength)
    print("Seasonality strength:",seasonality_strength)
    
### Augmented Dickey-Fuller Test to check for Stationarity
def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print('Augmented Dickey-Fuller Test: {}'.format(title))
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out['critical value ({})'.format(key)]=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")