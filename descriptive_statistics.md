

```python
import pandas as pd
import numpy as np
from scipy import stats

# read the data

data = pd.read_csv('/Users/vincentshields/Desktop/econometrics/caschool.csv')
```


```python
# describe all numerical variables

data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Observation Number</th>
      <th>dist_cod</th>
      <th>enrl_tot</th>
      <th>teachers</th>
      <th>calw_pct</th>
      <th>meal_pct</th>
      <th>computer</th>
      <th>testscr</th>
      <th>comp_stu</th>
      <th>expn_stu</th>
      <th>str</th>
      <th>avginc</th>
      <th>el_pct</th>
      <th>read_scr</th>
      <th>math_scr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>210.500000</td>
      <td>67472.809524</td>
      <td>2628.792857</td>
      <td>129.067376</td>
      <td>13.246042</td>
      <td>44.705237</td>
      <td>303.383333</td>
      <td>654.156548</td>
      <td>0.135927</td>
      <td>5312.407541</td>
      <td>19.640425</td>
      <td>15.316588</td>
      <td>15.768155</td>
      <td>654.970477</td>
      <td>653.342619</td>
    </tr>
    <tr>
      <th>std</th>
      <td>121.387808</td>
      <td>3466.994655</td>
      <td>3913.104985</td>
      <td>187.912679</td>
      <td>11.454821</td>
      <td>27.123381</td>
      <td>441.341298</td>
      <td>19.053348</td>
      <td>0.064956</td>
      <td>633.937053</td>
      <td>1.891812</td>
      <td>7.225890</td>
      <td>18.285927</td>
      <td>20.107980</td>
      <td>18.754202</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>61382.000000</td>
      <td>81.000000</td>
      <td>4.850000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>605.550049</td>
      <td>0.000000</td>
      <td>3926.069580</td>
      <td>14.000000</td>
      <td>5.335000</td>
      <td>0.000000</td>
      <td>604.500000</td>
      <td>605.400024</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>105.750000</td>
      <td>64307.750000</td>
      <td>379.000000</td>
      <td>19.662499</td>
      <td>4.395375</td>
      <td>23.282200</td>
      <td>46.000000</td>
      <td>640.049988</td>
      <td>0.093767</td>
      <td>4906.180053</td>
      <td>18.582360</td>
      <td>10.639000</td>
      <td>1.940807</td>
      <td>640.400024</td>
      <td>639.375015</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>210.500000</td>
      <td>67760.500000</td>
      <td>950.500000</td>
      <td>48.564999</td>
      <td>10.520450</td>
      <td>41.750700</td>
      <td>117.500000</td>
      <td>654.449982</td>
      <td>0.125464</td>
      <td>5214.516601</td>
      <td>19.723208</td>
      <td>13.727800</td>
      <td>8.777634</td>
      <td>655.750000</td>
      <td>652.449982</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>315.250000</td>
      <td>70419.000000</td>
      <td>3008.000000</td>
      <td>146.350002</td>
      <td>18.981350</td>
      <td>66.864725</td>
      <td>375.250000</td>
      <td>666.662506</td>
      <td>0.164466</td>
      <td>5601.401367</td>
      <td>20.871815</td>
      <td>17.629001</td>
      <td>22.970003</td>
      <td>668.725006</td>
      <td>665.849991</td>
    </tr>
    <tr>
      <th>max</th>
      <td>420.000000</td>
      <td>75440.000000</td>
      <td>27176.000000</td>
      <td>1429.000000</td>
      <td>78.994202</td>
      <td>100.000000</td>
      <td>3324.000000</td>
      <td>706.750000</td>
      <td>0.420833</td>
      <td>7711.506836</td>
      <td>25.799999</td>
      <td>55.327999</td>
      <td>85.539719</td>
      <td>704.000000</td>
      <td>709.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# descriptive statistics for select variables

data[['calw_pct', 'meal_pct', 'computer']].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>calw_pct</th>
      <th>meal_pct</th>
      <th>computer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>13.246042</td>
      <td>44.705237</td>
      <td>303.383333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.454821</td>
      <td>27.123381</td>
      <td>441.341298</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.395375</td>
      <td>23.282200</td>
      <td>46.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>10.520450</td>
      <td>41.750700</td>
      <td>117.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>18.981350</td>
      <td>66.864725</td>
      <td>375.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>78.994202</td>
      <td>100.000000</td>
      <td>3324.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# another method by index

data.iloc[:, 7:10].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>calw_pct</th>
      <th>meal_pct</th>
      <th>computer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>13.246042</td>
      <td>44.705237</td>
      <td>303.383333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.454821</td>
      <td>27.123381</td>
      <td>441.341298</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.395375</td>
      <td>23.282200</td>
      <td>46.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>10.520450</td>
      <td>41.750700</td>
      <td>117.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>18.981350</td>
      <td>66.864725</td>
      <td>375.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>78.994202</td>
      <td>100.000000</td>
      <td>3324.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# in python, 1 is also true, so there is no need to change the column (even though it may appear as true or false)
# descriptive statistics for a subset of variables

data['smallclass'] = data['str'] < 20
data.query("""smallclass==1""").describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Observation Number</th>
      <th>dist_cod</th>
      <th>enrl_tot</th>
      <th>teachers</th>
      <th>calw_pct</th>
      <th>meal_pct</th>
      <th>computer</th>
      <th>testscr</th>
      <th>comp_stu</th>
      <th>expn_stu</th>
      <th>str</th>
      <th>avginc</th>
      <th>el_pct</th>
      <th>read_scr</th>
      <th>math_scr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>238.000000</td>
      <td>238.000000</td>
      <td>238.000000</td>
      <td>238.000000</td>
      <td>238.000000</td>
      <td>238.000000</td>
      <td>238.000000</td>
      <td>238.000000</td>
      <td>238.000000</td>
      <td>238.000000</td>
      <td>238.000000</td>
      <td>238.000000</td>
      <td>238.000000</td>
      <td>238.000000</td>
      <td>238.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>227.407563</td>
      <td>67704.239496</td>
      <td>1507.071429</td>
      <td>80.194204</td>
      <td>12.951997</td>
      <td>41.633079</td>
      <td>193.777311</td>
      <td>657.351259</td>
      <td>0.146658</td>
      <td>5540.316183</td>
      <td>18.383887</td>
      <td>16.335805</td>
      <td>12.534326</td>
      <td>658.826050</td>
      <td>655.876468</td>
    </tr>
    <tr>
      <th>std</th>
      <td>121.421633</td>
      <td>3549.489615</td>
      <td>2510.398070</td>
      <td>131.817380</td>
      <td>11.344613</td>
      <td>27.271027</td>
      <td>318.587674</td>
      <td>19.358012</td>
      <td>0.066752</td>
      <td>670.522035</td>
      <td>1.283886</td>
      <td>8.552967</td>
      <td>16.819417</td>
      <td>20.152532</td>
      <td>19.356639</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>61382.000000</td>
      <td>81.000000</td>
      <td>4.850000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>606.750000</td>
      <td>0.000000</td>
      <td>4136.250977</td>
      <td>14.000000</td>
      <td>5.699000</td>
      <td>0.000000</td>
      <td>604.500000</td>
      <td>609.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>124.250000</td>
      <td>63990.250000</td>
      <td>307.500000</td>
      <td>17.575000</td>
      <td>3.994825</td>
      <td>20.506674</td>
      <td>37.250000</td>
      <td>643.524994</td>
      <td>0.103460</td>
      <td>5131.536499</td>
      <td>17.699945</td>
      <td>11.116000</td>
      <td>1.010545</td>
      <td>644.349991</td>
      <td>642.624985</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>229.500000</td>
      <td>68631.000000</td>
      <td>629.500000</td>
      <td>35.112499</td>
      <td>10.215700</td>
      <td>37.066500</td>
      <td>84.500000</td>
      <td>656.525024</td>
      <td>0.131993</td>
      <td>5399.535157</td>
      <td>18.739755</td>
      <td>14.050125</td>
      <td>4.958952</td>
      <td>657.899994</td>
      <td>654.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>337.500000</td>
      <td>70678.000000</td>
      <td>1786.250000</td>
      <td>94.600000</td>
      <td>18.513400</td>
      <td>63.851748</td>
      <td>247.000000</td>
      <td>669.912506</td>
      <td>0.185726</td>
      <td>5810.176635</td>
      <td>19.344734</td>
      <td>18.319098</td>
      <td>17.128223</td>
      <td>672.125015</td>
      <td>668.574982</td>
    </tr>
    <tr>
      <th>max</th>
      <td>420.000000</td>
      <td>75135.000000</td>
      <td>27176.000000</td>
      <td>1429.000000</td>
      <td>58.752201</td>
      <td>100.000000</td>
      <td>3324.000000</td>
      <td>706.750000</td>
      <td>0.358974</td>
      <td>7711.506836</td>
      <td>19.961538</td>
      <td>55.327999</td>
      <td>85.539719</td>
      <td>704.000000</td>
      <td>709.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# frequency table for countys

data.county.value_counts()
```




    Sonoma             29
    Los Angeles        27
    Kern               27
    Tulare             24
    San Diego          21
    Santa Clara        20
    San Mateo          17
    Humboldt           17
    Shasta             13
    Fresno             12
    Orange             11
    Santa Barbara      11
    Placer             11
    Merced             11
    El Dorado          10
    San Bernardino     10
    Siskiyou            9
    Nevada              9
    Ventura             9
    Kings               9
    Marin               8
    Tehama              8
    Stanislaus          7
    Santa Cruz          7
    Contra Costa        7
    Sacramento          7
    Monterey            7
    San Joaquin         6
    Imperial            6
    Tuolumne            6
    Sutter              6
    Butte               6
    Lassen              5
    Madera              5
    Riverside           4
    San Benito          3
    Glenn               3
    Lake                2
    Yuba                2
    Trinity             2
    San Luis Obispo     2
    Mendocino           1
    Alameda             1
    Calaveras           1
    Inyo                1
    Name: county, dtype: int64




```python
# more frequency tables

print(data.county.value_counts())
print(data.smallclass.value_counts())
```

    Sonoma             29
    Los Angeles        27
    Kern               27
    Tulare             24
    San Diego          21
    Santa Clara        20
    San Mateo          17
    Humboldt           17
    Shasta             13
    Fresno             12
    Orange             11
    Santa Barbara      11
    Placer             11
    Merced             11
    El Dorado          10
    San Bernardino     10
    Siskiyou            9
    Nevada              9
    Ventura             9
    Kings               9
    Marin               8
    Tehama              8
    Stanislaus          7
    Santa Cruz          7
    Contra Costa        7
    Sacramento          7
    Monterey            7
    San Joaquin         6
    Imperial            6
    Tuolumne            6
    Sutter              6
    Butte               6
    Lassen              5
    Madera              5
    Riverside           4
    San Benito          3
    Glenn               3
    Lake                2
    Yuba                2
    Trinity             2
    San Luis Obispo     2
    Mendocino           1
    Alameda             1
    Calaveras           1
    Inyo                1
    Name: county, dtype: int64
    True     238
    False    182
    Name: smallclass, dtype: int64



```python
# independent ttest for difference in means

# first we split the data 

data1 = data.query("smallclass==1")
data2 = data.query("smallclass==0")

stats.ttest_ind(data1['testscr'], data2['testscr'])
```




    Ttest_indResult(statistic=3.9991928467186746, pvalue=7.515397707407867e-05)




```python
data.groupby(['smallclass'])['testscr'].mean()
```




    smallclass
    False    649.978849
    True     657.351259
    Name: testscr, dtype: float64




```python
# ttest assuming unequal variance

stats.ttest_ind(data1['testscr'], data2['testscr'], equal_var = False)

```




    Ttest_indResult(statistic=4.042581850024987, pvalue=6.33255438998875e-05)




```python

```


```python

```
